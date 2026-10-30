/* Custom INT8 inference for DSC-AMCNet — signed q-domain + SMLAD packing.
 *
 * Reference semantics: re-derived from the ONNX QDQ graph (validated 12/12
 * layers against onnxruntime, see docs/P0_ACCURACY.md).
 * Activation q values are int8; "value = (q - zp) * scale".
 * Weight/bias values come from dsc_model_data.h as int8/int32 quantized ints.
 *
 * Every conv/dense layer uses a per-tensor multiplier m = s_in*s_w/s_out,
 * applied as a Q31 fixed-point multiply (round-to-nearest), then + zp_out.
 */
#include "dsc_infer.h"
#include "dsc_model_data.h"
#include "test_sample.h"
#include "main.h"

#include <stdint.h>
#include <string.h>
#include <math.h>

void uart_printf(const char *fmt, ...);

#define QMIN (-128)
#define QMAX 127

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* two signed 16-bit lanes packed in one 32-bit word */
static inline uint32_t pk2(int lo, int hi)
{
    return (uint16_t)(int16_t)lo | ((uint32_t)(uint16_t)(int16_t)hi << 16);
}

static inline int32_t smlad(uint32_t a, uint32_t b, int32_t acc)
{
    int32_t r;
    __asm volatile("smlad %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(acc));
    return r;
}

/* Q31 requant, expressed with a Q32 multiplier so that the whole rounding
 * step is one SMULL + add-carry:
 *   q = floor((acc*mq31 + 2^30) / 2^31)  ==  floor((acc*(2mq31) + 2^31) / 2^32)
 * The right hand form only needs the 64-bit product's high word plus the
 * carry out of the low word (bit 31 of lo), which is bit-exact. */
static inline int32_t rq(int32_t acc, int32_t mq32)
{
    int32_t hi;
    uint32_t lo;
    __asm volatile("smull %0, %1, %2, %3" : "=r"(lo), "=r"(hi) : "r"(acc), "r"(mq32));
    return hi + (int32_t)(lo >> 31);
}

/* ---- packed weights + multipliers ---- */
static uint32_t wp1[24 * 1];
static uint32_t wp3[24 * 12], wp5[24 * 12], wp7[24 * 12];
static uint32_t wfc1[6 * 12], wfc2[24 * 3];
static uint32_t wcl0[64 * 12], wcl3[11 * 32];

static int32_t b1[24], b3[24], b5[24], b7[24], bfc1[6], bfc2[24], bcl0[64], bcl3[11];
static int32_t b2[24], b4[24], b6[24];              /* dw bias + 128*sumw */
static int32_t sumw1[24], sumw3[24], sumw5[24], sumw7[24];
static int32_t sumfc1[6], sumfc2[24], sumcl0[64], sumcl3[11];
static int32_t sumdw1[24], sumdw2[24], sumdw3[24];

static int32_t m_c1, m_c2, m_c3, m_c4, m_c5, m_c6, m_c7;
static int32_t m_fc1, m_fc2, m_cl0, m_cl3;

/* 2 * round(m * 2^31): Q32 multiplier for rq(); |m| is ~1e-3..3e-2 for this
 * model, so doubling the Q31 multiplier never overflows int32. */
static int32_t q31(float m) { return (int32_t)(2.0 * lrint((double)m * 2147483648.0)); }

static void pack_w(const int8_t *w, int rows, int ic, uint32_t *out)
{
    int r, i;
    for (r = 0; r < rows; r++)
        for (i = 0; i < ic; i += 2)
            out[r * (ic / 2) + (i / 2)] = pk2(w[r * ic + i], w[r * ic + i + 1]);
}

static void sum_rows(const int8_t *w, int rows, int ic, int32_t *sum)
{
    int r, i;
    for (r = 0; r < rows; r++) {
        int32_t s = 0;
        for (i = 0; i < ic; i++) s += w[r * ic + i];
        sum[r] = s;
    }
}

static void sum_dw(const int8_t *w, int ch, int k, int32_t *sum)
{
    int c, i;
    for (c = 0; c < ch; c++) {
        int32_t s = 0;
        for (i = 0; i < k; i++) s += w[c * k + i];
        sum[c] = s;
    }
}

void dsc_infer_init(void)
{
    int i;

    pack_w(dsc_w_c1, 24, 2, wp1);
    pack_w(dsc_w_c3, 24, 24, wp3);
    pack_w(dsc_w_c5, 24, 24, wp5);
    pack_w(dsc_w_c7, 24, 24, wp7);
    pack_w(se_fc1_w, 6, 24, wfc1);
    pack_w(se_fc2_w, 24, 6, wfc2);
    pack_w(clf0_w, 64, 24, wcl0);
    pack_w(clf3_w, 11, 64, wcl3);

    sum_rows(dsc_w_c1, 24, 2, sumw1);
    sum_rows(dsc_w_c3, 24, 24, sumw3);
    sum_rows(dsc_w_c5, 24, 24, sumw5);
    sum_rows(dsc_w_c7, 24, 24, sumw7);
    sum_rows(se_fc1_w, 6, 24, sumfc1);
    sum_rows(se_fc2_w, 24, 6, sumfc2);
    sum_rows(clf0_w, 64, 24, sumcl0);
    sum_rows(clf3_w, 11, 64, sumcl3);
    sum_dw(dsc_w_c2, 24, 3, sumdw1);
    sum_dw(dsc_w_c4, 24, 3, sumdw2);
    sum_dw(dsc_w_c6, 24, 3, sumdw3);

    /* conv biases: q-domain base = b - zp_in*sumw ; zp_in = -4 (input) / -128 (rest) */
    for (i = 0; i < 24; i++) {
        b1[i] = dsc_b_c1[i] + 4 * sumw1[i];
        b3[i] = dsc_b_c3[i] + 128 * sumw3[i];
        b5[i] = dsc_b_c5[i] + 128 * sumw5[i];
        b7[i] = dsc_b_c7[i] + 128 * sumw7[i];
        b2[i] = dsc_b_c2[i] + 128 * sumdw1[i];
        b4[i] = dsc_b_c4[i] + 128 * sumdw2[i];
        b6[i] = dsc_b_c6[i] + 128 * sumdw3[i];
    }
    for (i = 0; i < 6; i++) bfc1[i] = se_fc1_b[i] + 128 * sumfc1[i];
    for (i = 0; i < 24; i++) bfc2[i] = se_fc2_b[i] + 128 * sumfc2[i];
    for (i = 0; i < 64; i++) bcl0[i] = clf0_b[i] + 128 * sumcl0[i];
    for (i = 0; i < 11; i++) bcl3[i] = clf3_b[i] + 128 * sumcl3[i];

    m_c1 = q31(A_IN * W_C1 / A_C1);
    m_c2 = q31(A_C1 * W_C2 / A_C2);
    m_c3 = q31(A_C2 * W_C3 / A_C3);
    m_c4 = q31(A_C3 * W_C4 / A_C4);
    m_c5 = q31(A_C4 * W_C5 / A_C5);
    m_c6 = q31(A_C5 * W_C6 / A_C6);
    m_c7 = q31(A_C6 * W_C7 / A_C7);
    m_fc1 = q31(A_SE_MEAN * W_SE_FC1 / A_SE_FC1);
    m_fc2 = q31(A_SE_FC1 * W_SE_FC2 / A_SE_FC2);
    m_cl0 = q31(A_GAP * W_CLF0 / A_CLF0);
    m_cl3 = q31(A_CLF0 * W_CLF3 / A_LOGITS);
}

static void conv1x1(const int8_t *in, int ic, int len, const uint32_t *wp,
                    const int32_t *base, int32_t m, int8_t *out)
{
    int oc, t, p, ic2 = ic / 2;
    uint32_t buf[32];

    for (t = 0; t < len; t++) {
        for (p = 0; p < ic2; p++)
            buf[p] = pk2(in[(2 * p) * len + t], in[(2 * p + 1) * len + t]);
        for (oc = 0; oc < 24; oc++) {
            const uint32_t *w = wp + oc * ic2;
            int32_t acc = base[oc];
            for (p = 0; p < ic2; p++) acc = smlad(buf[p], w[p], acc);
            out[oc * len + t] = (int8_t)clampi(rq(acc, m) - 128, QMIN, QMAX);
        }
    }
}

/* 3-tap depthwise, two outputs per packed pair:
 *   y[2i]   = w0*x[2i-1] + SMLAD((x[2i],x[2i+1]), (w1,w2))
 *   y[2i+1] = SMLAD((x[2i],x[2i+1]), (w0,w1)) + w2*x[2i+2]
 * Padding in the q domain is q = -128 (handled by the base offset already). */
static void conv_dw3(const int8_t *in, int ch, int len, const int8_t *w,
                     const int32_t *base, int32_t m, int8_t *out)
{
    int c, i;
    for (c = 0; c < ch; c++) {
        const int8_t *x = in + c * len;
        int8_t *o = out + c * len;
        const int8_t *wc = w + c * 3;
        const uint32_t w01 = pk2(wc[0], wc[1]);
        const uint32_t w12 = pk2(wc[1], wc[2]);
        const int32_t b = base[c];
        const int32_t w0 = wc[0], w2 = wc[2];
        const int n = len / 2;
        /* first pair: left padding q = -128 */
        {
            const uint32_t P = pk2(x[0], x[1]);
            const int32_t xp2 = (len > 2) ? x[2] : -128;
            int32_t a0 = smlad(P, w12, b + (int32_t)(-128) * w0);
            int32_t a1 = smlad(P, w01, b + (int32_t)xp2 * w2);
            o[0] = (int8_t)clampi(rq(a0, m) - 128, QMIN, QMAX);
            o[1] = (int8_t)clampi(rq(a1, m) - 128, QMIN, QMAX);
        }
        /* interior pairs: no boundary tests inside the loop */
        for (i = 1; i < n - 1; i++) {
            const int t0 = 2 * i;
            const uint32_t P = pk2(x[t0], x[t0 + 1]);
            int32_t a0 = smlad(P, w12, b + (int32_t)x[t0 - 1] * w0);
            int32_t a1 = smlad(P, w01, b + (int32_t)x[t0 + 2] * w2);
            o[t0] = (int8_t)clampi(rq(a0, m) - 128, QMIN, QMAX);
            o[t0 + 1] = (int8_t)clampi(rq(a1, m) - 128, QMIN, QMAX);
        }
        /* last pair: right padding q = -128 */
        if (n > 1) {
            const int t0 = 2 * (n - 1);
            const uint32_t P = pk2(x[t0], x[t0 + 1]);
            int32_t a0 = smlad(P, w12, b + (int32_t)x[t0 - 1] * w0);
            int32_t a1 = smlad(P, w01, b + (int32_t)(-128) * w2);
            o[t0] = (int8_t)clampi(rq(a0, m) - 128, QMIN, QMAX);
            o[t0 + 1] = (int8_t)clampi(rq(a1, m) - 128, QMIN, QMAX);
        }
    }
}

static void maxpool2(const int8_t *in, int ch, int len, int8_t *out)
{
    int c, t, n = len / 2;
    for (c = 0; c < ch; c++)
        for (t = 0; t < n; t++) {
            int8_t a = in[c * len + 2 * t], b = in[c * len + 2 * t + 1];
            out[c * n + t] = a > b ? a : b;
        }
}

static void se_block(const int8_t *in, int len, int8_t *out)
{
    int8_t qm[24], qf1[6], qf2[24];
    int8_t sig[24];
    int c, t;

    for (c = 0; c < 24; c++) {
        int32_t s = 0;
        for (t = 0; t < len; t++) s += in[c * len + t];
        {
            float v = ((float)s / (float)len + 128.0f) * A_C7;
            qm[c] = (int8_t)clampi((int)lrintf(v / A_SE_MEAN) - 128, QMIN, QMAX);
        }
    }

    for (c = 0; c < 6; c++) {
        int32_t acc = bfc1[c];
        for (t = 0; t < 24; t++) acc += (int32_t)qm[t] * (int32_t)se_fc1_w[c * 24 + t];
        qf1[c] = (int8_t)clampi(rq(acc, m_fc1) - 128, QMIN, QMAX);
    }

    for (c = 0; c < 24; c++) {
        int32_t acc = bfc2[c];
        for (t = 0; t < 6; t++) acc += (int32_t)qf1[t] * (int32_t)se_fc2_w[c * 6 + t];
        qf2[c] = (int8_t)clampi(rq(acc, m_fc2) - 5, QMIN, QMAX);
        {
            float v = (float)((int)qf2[c] + 5) * A_SE_FC2;
            float sg = 1.0f / (1.0f + expf(-v));
            sig[c] = (int8_t)clampi((int)lrintf(sg * 255.0f) - 128, QMIN, QMAX);
        }
    }

    for (c = 0; c < 24; c++) {
        float g = ((float)((int)sig[c] + 128)) * (1.0f / 255.0f);
        for (t = 0; t < len; t++) {
            float x = ((float)((int)in[c * len + t] + 128)) * A_C7;
            out[c * len + t] = (int8_t)clampi((int)lrintf(x * g / A_SE_MUL) - 128, QMIN, QMAX);
        }
    }
}


static void shuffle4(const int8_t *in, int len, int8_t *out)
{
    int g, r, t;
    for (g = 0; g < 4; g++)
        for (r = 0; r < 6; r++)
            for (t = 0; t < len; t++)
                out[(r * 4 + g) * len + t] = in[(g * 6 + r) * len + t];
}

static void gap(const int8_t *in, int len, int8_t *out)
{
    int c, t;
    for (c = 0; c < 24; c++) {
        int32_t s = 0;
        for (t = 0; t < len; t++) s += in[c * len + t];
        {
            float v = ((float)s / (float)len + 128.0f) * A_SE_MUL;
            out[c] = (int8_t)clampi((int)lrintf(v / A_GAP) - 128, QMIN, QMAX);
        }
    }
}

static void dense(const int8_t *in, int ic, int oc_n, const uint32_t *wp,
                  const int32_t *base, int32_t m, int zp_out, int8_t *out, int out_stride)
{
    int o, p, ic2 = ic / 2;
    uint32_t buf[32];

    for (p = 0; p < ic2; p++) buf[p] = pk2(in[2 * p], in[2 * p + 1]);
    for (o = 0; o < oc_n; o++) {
        const uint32_t *w = wp + o * ic2;
        int32_t acc = base[o];
        for (p = 0; p < ic2; p++) acc = smlad(buf[p], w[p], acc);
        out[o * out_stride] = (int8_t)clampi(rq(acc, m) + zp_out, QMIN, QMAX);
    }
}

static int8_t A[24 * 128];
static int8_t B[24 * 128];
static int8_t C[24 * 16 + 64];

void dsc_fill_sample(float *dst)
{
    int i;
    for (i = 0; i < 256; i++) dst[i] = ((float)test_sample[i] + 4.0f) * A_IN;
}

void dsc_infer_run_q(const int8_t *in8, int8_t *logits_out)
{
    int8_t qbuf[64];
    conv1x1(in8, 2, 128, wp1, b1, m_c1, A);
    conv_dw3(A, 24, 128, dsc_w_c2, b2, m_c2, B);
    conv1x1(B, 24, 128, wp3, b3, m_c3, A);
    maxpool2(A, 24, 128, B);
    conv_dw3(B, 24, 64, dsc_w_c4, b4, m_c4, A);
    conv1x1(A, 24, 64, wp5, b5, m_c5, B);
    maxpool2(B, 24, 64, A);
    conv_dw3(A, 24, 32, dsc_w_c6, b6, m_c6, B);
    conv1x1(B, 24, 32, wp7, b7, m_c7, A);
    maxpool2(A, 24, 32, B);
    se_block(B, 16, A);
    shuffle4(A, 16, B);
    gap(B, 16, C);
    dense(C, 24, 64, wcl0, bcl0, m_cl0, -128, qbuf, 1);
    dense(qbuf, 64, 11, wcl3, bcl3, m_cl3, 63, logits_out, 1);
}

void dsc_infer_run(const float *input_f32, int8_t *logits_out)
{
    int i;
    int8_t in8[2 * 128];

    for (i = 0; i < 256; i++)
        in8[i] = (int8_t)clampi((int)lrintf(input_f32[i] / A_IN) - 4, QMIN, QMAX);

    dsc_infer_run_q(in8, logits_out);
}