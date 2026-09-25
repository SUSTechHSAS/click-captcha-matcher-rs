/* Solve captchas through the C ABI.
 *
 *   cc -O2 examples/solve.c -Iinclude -Ltarget/release -lccm -o solve
 *   LD_LIBRARY_PATH=target/release ./solve captcha.jpg
 */
#include <stdio.h>

#include "ccm.h"

int main(int argc, char **argv) {
    static uint8_t buf[1 << 20];
    int err = 0;
    ccm_solver *s = ccm_new(NULL, 0, &err);
    if (!s) {
        fprintf(stderr, "ccm_new: error %d\n", err);
        return 2;
    }
    printf("libccm %s, %s kernel\n", ccm_version(), ccm_isa());
    for (int i = 1; i < argc; i++) {
        FILE *f = fopen(argv[i], "rb");
        if (!f) {
            perror(argv[i]);
            continue;
        }
        size_t n = fread(buf, 1, sizeof buf, f);
        fclose(f);
        int32_t p[8];
        float margin;
        int rc = ccm_solve(s, buf, n, p, &margin);
        if (rc != CCM_OK) {
            fprintf(stderr, "%s: error %d\n", argv[i], rc);
            continue;
        }
        printf("%s %d-%d,%d-%d,%d-%d,%d-%d %.6f\n", argv[i], p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], margin);
    }
    ccm_free(s);
    return 0;
}
