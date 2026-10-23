// NMRF fused window attention with relative-position embeddings on q, k and v (WindowAttention.forward, Swin-style
// windows of ws x ws cells x n candidates = T tokens, head dim 32). Same math as upstream:
//   logit_ij = q_i . k_j + q_i . Ek[r(ci,cj)] + k_j . Eq[r(ci,cj)] + mask[w % nW][i][j]      (q, Eq scaled)
//   out_i    = sum_j softmax_j(logit_ij) * (v_j + Ev[r(ci,cj)])
// with r = (hi - hj + ws - 1) * (2 ws - 1) + (wi - wj + ws - 1) (Swin relative_position_index), cell c = token / n.
// Inputs fp16, math fp32. One CTA per (group of wpc windows, head), one thread per query token (wpc = blockDim / T, so
// small windows (refinement: T = 16) share a CTA and the embedding-table load); K, V and the head's table slice in
// shared memory, read as half2. Size-independent (T, ws, n, wpc, number of mask windows are runtime arguments).
#include <cuda_fp16.h>

#define D 32
#define D2 16

extern "C" __global__ void win_rpe_attn(const __half* __restrict__ q, const __half* __restrict__ k,
                                        const __half* __restrict__ v, const __half* __restrict__ table, int tstride,
                                        const float* __restrict__ mask, int nW, int heads, int T, int ws, int n,
                                        float scale, __half* __restrict__ out, int nwin, int wpc) {
    extern __shared__ __half2 smem2[];
    const int R = (2 * ws - 1) * (2 * ws - 1);
    __half2* se = smem2;               // [R, 3 * D2]: Eq | Ek | Ev for this head
    __half2* sk = se + R * 3 * D2;     // [wpc, T, D2]
    __half2* sv = sk + wpc * T * D2;   // [wpc, T, D2]
    const int h = blockIdx.x % heads;
    const int w0 = (blockIdx.x / heads) * wpc;
    for (int x = threadIdx.x; x < R * 3 * D2; x += blockDim.x) {
        int r = x / (3 * D2), c = x - r * 3 * D2;
        se[x] = reinterpret_cast<const __half2*>(table + (long long)r * tstride + h * 3 * D)[c];
    }
    for (int x = threadIdx.x; x < wpc * T * D2; x += blockDim.x) {
        int wl = x / (T * D2), rem = x - wl * T * D2;
        int w = w0 + wl;
        if (w < nwin) {
            long long b2 = ((long long)w * heads + h) * T * D2;
            sk[x] = reinterpret_cast<const __half2*>(k)[b2 + rem];
            sv[x] = reinterpret_cast<const __half2*>(v)[b2 + rem];
        }
    }
    __syncthreads();
    const int wl = threadIdx.x / T, i = threadIdx.x - wl * T;
    const int w = w0 + wl;
    if (wl >= wpc || w >= nwin) return;
    const long long base2 = ((long long)w * heads + h) * T * D2;
    float2 qi[D2];
    for (int e = 0; e < D2; ++e) {
        float2 t = __half22float2(reinterpret_cast<const __half2*>(q)[base2 + (long long)i * D2 + e]);
        qi[e] = make_float2(t.x * scale, t.y * scale);
    }
    const int ci = i / n, hi = ci / ws, wi = ci - hi * ws;
    const float* mrow = mask ? mask + ((long long)(w % nW) * T + i) * T : nullptr;
    const __half2* kw = sk + wl * T * D2;
    const __half2* vw = sv + wl * T * D2;
    float m = -INFINITY, s = 0.f;
    float2 acc[D2];
    for (int e = 0; e < D2; ++e) acc[e] = make_float2(0.f, 0.f);
    for (int j = 0; j < T; ++j) {
        float l = mrow ? mrow[j] : 0.f;
        if (l == -INFINITY) continue;
        const int cj = j / n, hj = cj / ws, wj = cj - hj * ws;
        const int r = (hi - hj + ws - 1) * (2 * ws - 1) + (wi - wj + ws - 1);
        const __half2* eq = se + r * 3 * D2;
        const __half2* ek = eq + D2;
        const __half2* kj = kw + j * D2;
        float a = 0.f, b = 0.f;
        for (int e = 0; e < D2; ++e) {
            float2 kv = __half22float2(kj[e]);
            float2 kk = __half22float2(ek[e]);
            float2 qq = __half22float2(eq[e]);
            a += qi[e].x * (kv.x + kk.x) + qi[e].y * (kv.y + kk.y);
            b += kv.x * qq.x + kv.y * qq.y;
        }
        l += a + b * scale;
        float mn = fmaxf(m, l);
        float c = __expf(m - mn), p = __expf(l - mn);
        s = s * c + p;
        const __half2* ev = eq + 2 * D2;
        const __half2* vj = vw + j * D2;
        for (int e = 0; e < D2; ++e) {
            float2 vv = __half22float2(vj[e]);
            float2 ee = __half22float2(ev[e]);
            acc[e].x = acc[e].x * c + p * (vv.x + ee.x);
            acc[e].y = acc[e].y * c + p * (vv.y + ee.y);
        }
        m = mn;
    }
    const float inv = 1.f / s;
    __half2* o = reinterpret_cast<__half2*>(out) + base2 + (long long)i * D2;
    for (int e = 0; e < D2; ++e) o[e] = __floats2half2_rn(acc[e].x * inv, acc[e].y * inv);
}
