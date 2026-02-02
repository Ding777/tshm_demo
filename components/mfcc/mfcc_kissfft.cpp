// minimal MFCC wrapper that uses kiss_fftr (very basic energy/mel approx).
#include "mfcc_kissfft.hpp"
#include "kiss_fftr.h"
#include <cmath>
#include <cstring>
#include <cstdlib>

static float hann_window_sum = -1.0f;

void compute_mfcc_from_pcm_kissfft(const int16_t* pcm, int pcm_len, float* out, int out_dim) {
    // Very small placeholder MFCC-like routine using magnitude spectrum energy mapped to out_dim bands.
    // NOT a production MFCC — replace this with a real MFCC using mel filters + DCT for accuracy.
    if (pcm_len <= 0 || out_dim <= 0) {
        for (int i=0;i<out_dim;i++) out[i] = 0.0f;
        return;
    }

    // choose FFT size = next power of two >= pcm_len
    int nfft = 1;
    while (nfft < pcm_len) nfft <<= 1;
    // allocate buffers (on heap; small sizes)
    kiss_fftr_cfg cfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    if (!cfg) {
        for (int i=0;i<out_dim;i++) out[i]=0.0f;
        return;
    }
    kiss_fft_cpx* freq = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx)*(nfft/2+1));
    if (!freq) { free(cfg); for (int i=0;i<out_dim;i++) out[i]=0.0f; return; }

    // prepare float input (zero pad)
    float* in = (float*)malloc(sizeof(float)*nfft);
    if (!in) { free(freq); free(cfg); for (int i=0;i<out_dim;i++) out[i]=0.0f; return; }
    for (int i=0;i<nfft;i++) in[i] = (i < pcm_len) ? (float)pcm[i] : 0.0f;

    // run real FFT
    kiss_fftr(cfg, in, freq);

    // compute magnitude spectrum
    int nbins = nfft/2 + 1;
    float* mag = (float*)malloc(sizeof(float)*nbins);
    if (!mag) { free(in); free(freq); free(cfg); for (int i=0;i<out_dim;i++) out[i]=0.0f; return; }
    for (int k=0;k<nbins;k++) {
        mag[k] = sqrtf(freq[k].r*freq[k].r + freq[k].i*freq[k].i);
    }

    // Simple mel-like bucketing: map nbins -> out_dim by averaging ranges
    for (int b=0;b<out_dim;b++) {
        int start = (int)((long long)b * nbins / out_dim);
        int end = (int)((long long)(b+1) * nbins / out_dim);
        if (end <= start) end = start+1;
        float s=0.0f;
        for (int k=start;k<end;k++) s += mag[k];
        float avg = s / (end-start);
        // compress with log, normalize roughly
        out[b] = logf(1.0f + avg);
    }

    // cleanup
    free(mag); free(in); free(freq); free(cfg);
}
