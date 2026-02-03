#include "mfcc_kissfft.hpp"
#include <cmath>
#include <cstddef>
#include "kiss_fft.h"


// Minimal placeholder MFCC: normalized log energy repeated to fill mfcc_dim.
// This compiles and runs for testing. 
bool mfcc_compute_from_pcm(const int16_t* pcm_samples, int pcm_len, float* mfcc_out, int mfcc_dim, int sample_rate) {
    if (!pcm_samples || pcm_len <= 0 || !mfcc_out || mfcc_dim <= 0) return false;
    double energy = 1e-9;
    for (int i = 0; i < pcm_len; ++i) {
        double v = pcm_samples[i];
        energy += v * v;
    }
    double l = 10.0 * log10(energy / pcm_len + 1e-12);
    // crude normalization to [0,1]
    float v = (float)((l + 100.0) / 100.0);
    if (v < 0.0f) v = 0.0f;
    for (int i = 0; i < mfcc_dim; ++i) mfcc_out[i] = v;
    return true;
}
