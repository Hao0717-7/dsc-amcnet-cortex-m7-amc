#ifndef DSC_INFER_H
#define DSC_INFER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* one-time weight packing (call before first run) */
void dsc_infer_init(void);

/* input: 2x128 float32, normalized; output: 11 int8 logits (scale A_LOGITS) */
void dsc_infer_run(const float *input_f32, int8_t *logits_out);

#ifdef __cplusplus
}
#endif

#endif /* DSC_INFER_H */
