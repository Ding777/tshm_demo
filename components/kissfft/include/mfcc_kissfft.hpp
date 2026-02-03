#pragma once
#include <cstdint>
#include <cstdbool>
#include "mfcc_kissfft.hpp"


#ifdef __cplusplus
extern "C" {
#endif

// Simple MFCC API used by main.cpp.
// NOTE: this placeholder implementation returns a trivial energy-based feature.
bool mfcc_compute_from_pcm(const int16_t* pcm_samples, int pcm_len, float* mfcc_out, int mfcc_dim, int sample_rate);

#ifdef __cplusplus
}
#endif


