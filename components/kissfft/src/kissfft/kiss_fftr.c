/* kiss_fftr.c
 *
 * Minimal real FFT wrapper that uses the complex FFT above.
 * Computes half-complex spectrum for real input.
 *
 * Note: This is a canonical simple implementation sufficient for MFCC.
 */

#include <stdlib.h>
#include <string.h>
#include "kiss_fftr.h"

struct kiss_fftr_state {
    int nfft;
    kiss_fft_cfg kfft; /* complex FFT state */
};

kiss_fftr_cfg kiss_fftr_alloc(int nfft,int inverse_fft,void* mem,size_t* lenmem) {
    (void)mem; (void)lenmem;
    struct kiss_fftr_state* st = (struct kiss_fftr_state*)malloc(sizeof(struct kiss_fftr_state));
    if (!st) return NULL;
    st->nfft = nfft;
    st->kfft = kiss_fft_alloc(nfft, inverse_fft, NULL, NULL);
    if (!st->kfft) { free(st); return NULL; }
    return st;
}

void kiss_fftr(kiss_fftr_cfg cfg,const kiss_fft_scalar *timedata,kiss_fft_cpx *freqdata) {
    int nfft = cfg->nfft;
    /* pack real data into complex buffer with imag=0 */
    kiss_fft_cpx* buf = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * nfft);
    if (!buf) return;
    for (int i=0;i<nfft;i++) { buf[i].r = timedata[i]; buf[i].i = 0; }
    /* run complex FFT */
    kiss_fft(cfg->kfft, buf, buf);
    /* output first nfft/2+1 bins */
    int nbins = nfft/2 + 1;
    for (int k=0;k<nbins;k++) freqdata[k] = buf[k];
    free(buf);
}
