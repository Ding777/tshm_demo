#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Compute MFCC-like features from PCM samples.
// pcm: int16_t samples (mono), pcm_len samples.
// out: float buffer of length out_dim (caller ensures out_dim matches D_MODEL or desired dim).
void compute_mfcc_from_pcm_kissfft(const int16_t* pcm, int pcm_len, float* out, int out_dim);

#ifdef __cplusplus
}
#endif

