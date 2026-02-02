/* kiss_fftr.h
 * Real FFT wrapper API for kiss_fft
 */

#ifndef KISS_FFTR_H
#define KISS_FFTR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "kiss_fft.h"

/* config for real FFT */
typedef struct kiss_fftr_state* kiss_fftr_cfg;

/* allocate real FFT config */
kiss_fftr_cfg kiss_fftr_alloc(int nfft,int inverse_fft,void* mem,size_t* lenmem);

/* compute real forward (or inverse if inverse_fft) transform
 * in: real time samples (float) length nfft
 * out: complex output length nfft/2+1
 */
void kiss_fftr(kiss_fftr_cfg cfg,const kiss_fft_scalar *timedata,kiss_fft_cpx *freqdata);

#ifdef __cplusplus
}
#endif

#endif /* KISS_FFTR_H */
