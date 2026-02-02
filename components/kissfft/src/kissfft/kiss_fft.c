/* kiss_fft.h
 * Basic KissFFT header (C)
 *
 * Minimal version for building kiss_fft and kiss_fftr.
 *
 * Original KissFFT by Mark Borgerding - http://github.com/mborgerding/kissfft
 * License: BSD/MIT style. Use freely.
 */

#ifndef KISS_FFT_H
#define KISS_FFT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "kiss_fft_scalar.h"
#include <stddef.h>
#include "kiss_fft.h"



typedef struct {
    kiss_fft_scalar r;
    kiss_fft_scalar i;
} kiss_fft_cpx;

/* Type for config object */
typedef struct kiss_fft_state* kiss_fft_cfg;

/* Kiss FFT API */
kiss_fft_cfg kiss_fft_alloc(int nfft,int inverse_fft,void* mem,size_t* lenmem);
void kiss_fft(kiss_fft_cfg cfg,const kiss_fft_cpx *fin,kiss_fft_cpx *fout);
void kiss_fft_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* KISS_FFT_H */
