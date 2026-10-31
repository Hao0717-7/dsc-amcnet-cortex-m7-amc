/* Host harness: run the production INT8 engine (custom_engine/dsc_infer.c) on
 * quantized windows and dump the 11 q-domain logits per window, so the exact
 * same C code that runs on the MCU can be checked bit-for-bit against
 * onnxruntime on a PC.
 *
 * Build (needs any C compiler; zig is installed with `pip install ziglang`):
 *   python -m ziglang cc -O2 -DDSC_HOST_SIM \
 *       -Istm32_deployment/tools/host_sim -Istm32_deployment/custom_engine \
 *       stm32_deployment/tools/host_sim/host_engine.c \
 *       stm32_deployment/custom_engine/dsc_infer.c -o .tmp/host_sim/engine.exe
 *
 * usage: engine.exe <in.bin> <out.bin> <n_windows>
 *   in.bin  : n_windows * 256 int8 (input q values, zp = -4)
 *   out.bin : n_windows * 11 int8 (logits q values, zp = +63)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "dsc_infer.h"

void uart_printf(const char *fmt, ...) { (void)fmt; }

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s <in.bin> <out.bin> <n_windows>\n", argv[0]);
        return 2;
    }
    const long n = strtol(argv[3], NULL, 10);
    FILE *fi = fopen(argv[1], "rb");
    if (!fi) { perror("open in"); return 1; }
    FILE *fo = fopen(argv[2], "wb");
    if (!fo) { perror("open out"); return 1; }

    int8_t in[256], out[11];
    dsc_infer_init();
    for (long w = 0; w < n; w++) {
        if (fread(in, 1, sizeof in, fi) != sizeof in) {
            fprintf(stderr, "short read at window %ld\n", w);
            return 1;
        }
        dsc_infer_run_q(in, out);
        fwrite(out, 1, sizeof out, fo);
    }
    fclose(fi);
    fclose(fo);
    return 0;
}
