/* click-captcha-matcher: 250x80 click captcha JPEG -> 4 click points.
 *
 *   ccm_solver *s = ccm_new(NULL, 0, NULL);          // embedded w16 model
 *   int32_t pts[8]; float margin;
 *   if (ccm_solve(s, jpeg, jpeg_len, pts, &margin) == CCM_OK)
 *       ...                                          // (pts[0],pts[1]) .. (pts[6],pts[7])
 *   ccm_free(s);
 *
 * Points are in prompt order. `margin` (best minus runner-up assignment score)
 * is the confidence: below a threshold, fetch a new captcha instead of submitting.
 * Every function is thread-safe, and one solver may be shared between threads.
 */
#ifndef CCM_H
#define CCM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CCM_WIDTH 250
#define CCM_HEIGHT 80

enum {
    CCM_OK = 0,
    CCM_EARG = -1,         /* NULL pointer */
    CCM_EJPEG = -2,        /* not a (valid) JPEG */
    CCM_EUNSUPPORTED = -3, /* progressive, 12-bit, CMYK, ... */
    CCM_ESIZE = -4,        /* not 250x80 */
    CCM_EMODEL = -5        /* bad .ccm model file */
};

typedef struct ccm_solver ccm_solver;

/* Load a .ccm model (tools/export_ccm.py); model = NULL loads the embedded one.
 * Returns NULL on failure, with the error code in *err when err is not NULL.
 * The bytes are copied; they need not outlive the call. */
ccm_solver *ccm_new(const uint8_t *model, size_t len, int *err);
void ccm_free(ccm_solver *s);

/* JPEG bytes -> points[8] = x0,y0,x1,y1,x2,y2,x3,y3 and *margin (may be NULL). */
int ccm_solve(const ccm_solver *s, const uint8_t *jpeg, size_t len, int32_t points[8], float *margin);

/* The same for a decoded 250x80 gray image (row-major, one byte per pixel). */
int ccm_solve_gray(const ccm_solver *s, const uint8_t *gray, int32_t points[8], float *margin);

/* JPEG -> 250x80 gray, identical to Pillow's Image.open(f).convert("L"). */
int ccm_decode(const uint8_t *jpeg, size_t len, uint8_t *gray);

/* Gray image -> sim[24] (4x6 cosine similarity, row-major) and centers[12]
 * (x, y of the 6 candidate slots; may be NULL). For debugging and evaluation. */
int ccm_similarity(const ccm_solver *s, const uint8_t *gray, float sim[24], float centers[12]);

const char *ccm_version(void);
/* Compute kernel in use: "avx2", "neon" or "generic". */
const char *ccm_isa(void);
/* Use the portable kernel from now on (testing / benchmarking). */
void ccm_force_generic(void);

#ifdef __cplusplus
}
#endif

#endif /* CCM_H */
