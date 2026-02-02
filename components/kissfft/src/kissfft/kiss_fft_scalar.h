/* kiss_fft_scalar.h
 * Minimal kissfft scalar header
 * Public domain / MIT-style usage (same as original KissFFT)
 */

#ifndef KISS_FFT_SCALAR_H
#define KISS_FFT_SCALAR_H

#ifdef FIXED_POINT
#include <stdint.h>
typedef int16_t kiss_fft_scalar;
#else
typedef float kiss_fft_scalar;
#endif

#endif /* KISS_FFT_SCALAR_H */
