#ifndef KISS_FFT_H
#define KISS_FFT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>          // <<< REQUIRED
#include "kiss_fft_scalar.h"

typedef struct {
    kiss_fft_scalar r;
    kiss_fft_scalar i;
} kiss_fft_cpx;

typedef struct kiss_fft_state* kiss_fft_cfg;

kiss_fft_cfg kiss_fft_alloc(int nfft,int inverse_fft,void* mem,size_t* lenmem);
void kiss_fft(kiss_fft_cfg cfg,const kiss_fft_cpx *fin,kiss_fft_cpx *fout);
void kiss_fft_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* KISS_FFT_H */

