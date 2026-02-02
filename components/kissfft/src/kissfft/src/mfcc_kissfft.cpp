// mfcc_kissfft.cpp
#include "mfcc_kissfft.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include "kiss_fftr.h"   // from kissfft
#include "kiss_fft.h"

static inline float hz_to_mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static inline float mel_to_hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

// Hamming window
static void hamming_window(float* w, int N) {
    for (int n = 0; n < N; ++n) {
        w[n] = 0.54f - 0.46f * cosf(2.0f * M_PI * n / (N - 1));
    }
}

bool mfcc_compute_from_pcm(const int16_t* pcm_samples, int pcm_len, float* out, int out_dim, int sample_rate) {
    if (!pcm_samples || !out) return false;
    // Typical MFCC params (tweak as needed):
    const int frame_len = 512;      // FFT length (power of two). Hop can be frame_len for simplicity.
    const int n_fft = frame_len;
    if (pcm_len < frame_len) return false;

    const int n_bins = n_fft/2 + 1;
    const int n_mels = out_dim;    // produce same count of mel bands as out_dim
    const int n_dct = out_dim;     // DCT output dim

    // allocate on stack-ish / vector
    std::vector<float> frame(frame_len);
    std::vector<float> win(frame_len);
    std::vector<float> power_spec(n_bins);
    std::vector<float> mel_energies(n_mels);
    std::vector<float> mel_filterbank(n_mels * n_bins);
    std::vector<float> dct_out(n_dct);

    // create window
    hamming_window(win.data(), frame_len);

    // copy + window first frame (zero-crossing already handled)
    for (int i = 0; i < frame_len; ++i) frame[i] = (float)pcm_samples[i] * win[i];

    // Prepare KissFFT config
    kiss_fftr_cfg cfg = kiss_fftr_alloc(n_fft, 0, NULL, NULL);
    if (!cfg) return false;
    std::vector<kiss_fft_cpx> freq_cpx(n_bins);

    // Compute FFT
    kiss_fftr(cfg, frame.data(), freq_cpx.data());

    // Compute power spectrum
    power_spec[0] = (freq_cpx[0].r * freq_cpx[0].r); // DC
    for (int k = 1; k < n_bins-1; ++k) {
        float r = freq_cpx[k].r;
        float i = freq_cpx[k].i;
        power_spec[k] = r*r + i*i;
    }
    power_spec[n_bins-1] = (freq_cpx[n_bins-1].r * freq_cpx[n_bins-1].r); // Nyquist (real)

    // Build mel filterbank (triangular filters) ONCE per invocation (cost negligible here)
    // compute mel scale limits
    float nyquist = sample_rate * 0.5f;
    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel(nyquist);
    // mels coordinates
    std::vector<float> mel_centers(n_mels + 2);
    for (int i=0;i<(int)mel_centers.size();++i)
        mel_centers[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (n_mels + 1));

    // precompute bin freq centers
    std::vector<float> fft_freqs(n_bins);
    for (int k=0;k<n_bins;k++) fft_freqs[k] = (sample_rate * (float)k) / (float)n_fft;

    // fill mel_filterbank (row-major: mel x bin)
    for (int m = 0; m < n_mels; ++m) {
        float left = mel_centers[m];
        float center = mel_centers[m+1];
        float right = mel_centers[m+2];
        for (int k=0;k<n_bins;++k) {
            float f = fft_freqs[k];
            float val = 0.0f;
            if (f >= left && f <= center) {
                val = (f - left) / (center - left);
            } else if (f > center && f <= right) {
                val = (right - f) / (right - center);
            } else val = 0.0f;
            mel_filterbank[m * n_bins + k] = val;
        }
    }

    // Apply mel filters to power spectrum -> mel energies
    for (int m=0;m<n_mels;++m) {
        double sum = 1e-12;
        for (int k=0;k<n_bins;++k) sum += (double)power_spec[k] * (double)mel_filterbank[m * n_bins + k];
        mel_energies[m] = (float)sum;
    }

    // Log compress
    for (int m=0;m<n_mels;++m) mel_energies[m] = logf(mel_energies[m] + 1e-12f);

    // DCT-II matrix naive: out[u] = sum_{m=0..M-1} mel_energies[m] * cos(pi*u*(2m+1)/(2M))
    for (int u=0; u<n_dct; ++u) {
        double s = 0.0;
        for (int m=0;m<n_mels;++m) {
            s += (double)mel_energies[m] * cos(M_PI * (double)u * (2.0*(double)m + 1.0) / (2.0 * (double)n_mels));
        }
        // optionally scale by sqrt(2/M) etc; here we keep raw DCT amplitude
        dct_out[u] = (float)s;
    }

    // copy to out (if out_dim != n_dct, we either truncate or zero-pad)
    int to_copy = std::min(n_dct, out_dim);
    for (int i=0;i<to_copy;++i) out[i] = dct_out[i];
    for (int i=to_copy;i<out_dim;++i) out[i] = 0.0f;

    free(cfg);
    return true;
}
