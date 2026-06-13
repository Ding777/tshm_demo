/*
  main.cpp
  Complete file — functions defined before app_main, i2s init uses memset + explicit fields.

*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cinttypes>

extern "C" {
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_timer.h"
#include "driver/i2s.h"
}
#include "mfcc_kissfft.hpp"
#include "tshm_weights.hpp" // <<-- must exist in components/tshm/include/

// <-- MFCC asset header (must define yes_01_mfcc, yes_01_mfcc_frames, yes_01_mfcc_dim)
#include "assets_yes_01.h"

static const char* TAG = "TSHM_DEVICE";

// ------------------------------------------------------------------
// PSRAM helpers
// ------------------------------------------------------------------
void* ps_malloc(size_t bytes) {
    void* p = heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!p) {
        ESP_LOGE(TAG, "Internal RAM alloc failed for %zu bytes", bytes);
    }
    return p;
}
void ps_free(void* p) { if (p) heap_caps_free(p); }

// ------------------------------------------------------------------
// Simple math & util helpers
// ------------------------------------------------------------------
inline float fast_tanh(float x) { return tanhf(x); }

void matvec(const float* W, int out_dim, int in_dim, const float* x, float* y) {
    for (int i = 0; i < out_dim; ++i) {
        const float* wrow = W + (size_t)i * in_dim;
        float s = 0.0f;
        for (int j = 0; j < in_dim; ++j) s += wrow[j] * x[j];
        y[i] = s;
    }
}

void layernorm_forward(const float* x, int dim, const float* gamma, const float* beta, float* out, float eps=1e-5f) {
    float mean = 0.0f;
    for (int i = 0; i < dim; ++i) mean += x[i];
    mean /= dim;
    float var = 0.0f;
    for (int i = 0; i < dim; ++i) { float t = x[i] - mean; var += t * t; }
    var /= dim;
    float denom = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < dim; ++i) out[i] = (x[i] - mean) * denom * gamma[i] + beta[i];
}

static bool is_all_zero(const float* buf, int len, float eps=1e-6f) {
    for (int i = 0; i < len; ++i) if (fabsf(buf[i]) > eps) return false;
    return true;
}

// ------------------------------------------------------------------
// Globals: dequantized params (PSRAM)
// NOTE: names such as encoder_embed_weight_q etc. must come from tshm_weights.hpp
// ------------------------------------------------------------------
static float* embed_weight_f = nullptr; static int embed_shape0=0, embed_shape1=0;

static float* U_weight_f = nullptr; static int U_shape0=0, U_shape1=0;
static float* V_weight_f = nullptr; static int V_shape0=0, V_shape1=0;
static float* A_weight_f = nullptr; static int A_shape0=0, A_shape1=0;
static float* B_weight_f = nullptr; static int B_shape0=0, B_shape1=0;

static float* pre_ln_gamma = nullptr; static float* pre_ln_beta = nullptr; static int ln_dim = 0;

static float* ffn_fc1_weight = nullptr; static int ffn_fc1_shape0=0, ffn_fc1_shape1=0; static float* ffn_fc1_bias=nullptr;
static float* ffn_fc2_weight = nullptr; static int ffn_fc2_shape0=0, ffn_fc2_shape1=0; static float* ffn_fc2_bias=nullptr;

static float* layer_c = nullptr; static int layer_c_len = 0;

static float* head_weight_f = nullptr; static int head_rows=0, head_cols=0;
static float* head_bias_f = nullptr;

static int D_MODEL = 0;
static int R_DIM = 0;
static int K_DIM = 0;

// input (MFCC) feature dimension (n_mfcc)
static int INPUT_DIM = 0;

static float const_w = 0.1f;
static float const_s = 0.1f;
static float res_scale_val = 0.2f;

// Streaming state
static float* S_running = nullptr;  // [R_DIM]
static float* M_pref = nullptr;     // [R_DIM]
static float* N_pref = nullptr;     // [R_DIM]

// Preallocated scratch buffers
static float *scratch_x_norm = nullptr, *scratch_phi = nullptr;
static float *scratch_Pt = nullptr, *scratch_Qt = nullptr;
static float *scratch_tmpA = nullptr, *scratch_tmpB = nullptr, *scratch_e = nullptr, *scratch_G = nullptr;
static float *scratch_Mt = nullptr, *scratch_Nt = nullptr;
static float *scratch_phi_prime = nullptr, *scratch_term1 = nullptr, *scratch_term2 = nullptr;
static float *scratch_g = nullptr, *scratch_residual = nullptr, *scratch_x_next = nullptr, *scratch_ffn_out = nullptr;
static float *scratch_logits = nullptr;

// internal ffn tmp
static float* ffn_internal_tmp = nullptr;
static int ffn_internal_tmp_capacity = 0;

// ------------------------------------------------------------------
// Dequant helper
// ------------------------------------------------------------------
float* dequant_int8_to_float(const int8_t* qptr, size_t nelems, float scale) {
    float* dst = (float*)ps_malloc(nelems * sizeof(float));
    if (!dst) return nullptr;
    for (size_t i = 0; i < nelems; ++i) dst[i] = float(qptr[i]) * scale;
    return dst;
}

// ------------------------------------------------------------------
// embed_apply (INPUT_DIM -> D_MODEL)
// ------------------------------------------------------------------
static void embed_apply(const float* mfcc_in, float* out_emb) {
    for (int i = 0; i < D_MODEL; ++i) {
        const float* wrow = embed_weight_f + (size_t)i * INPUT_DIM;
        float s = 0.0f;
        for (int j = 0; j < INPUT_DIM; ++j) s += wrow[j] * mfcc_in[j];
        out_emb[i] = s;
    }
}

// ------------------------------------------------------------------
// Load and dequantize all weights (from tshm_weights.hpp int8 arrays)
// ------------------------------------------------------------------
void load_and_dequant_all() {
    ESP_LOGI(TAG, "Dequantizing weights into PSRAM...");

    // embed
    {
        int s0 = encoder_embed_weight_shape[0];
        int s1 = encoder_embed_weight_shape[1];
        size_t n = (size_t)s0 * s1;
        embed_weight_f = dequant_int8_to_float(encoder_embed_weight_q, n, encoder_embed_weight_scale);
        embed_shape0 = s0; embed_shape1 = s1;
        ESP_LOGI(TAG, "Loaded embed weight: %d x %d", s0, s1);
    }

    // U,V,A,B
    {
        U_shape0 = encoder_layers_0_U_weight_shape[0];
        U_shape1 = encoder_layers_0_U_weight_shape[1];
        size_t nU = (size_t)U_shape0 * U_shape1;
        U_weight_f = dequant_int8_to_float(encoder_layers_0_U_weight_q, nU, encoder_layers_0_U_weight_scale);

        V_shape0 = encoder_layers_0_V_weight_shape[0];
        V_shape1 = encoder_layers_0_V_weight_shape[1];
        size_t nV = (size_t)V_shape0 * V_shape1;
        V_weight_f = dequant_int8_to_float(encoder_layers_0_V_weight_q, nV, encoder_layers_0_V_weight_scale);

        A_shape0 = encoder_layers_0_A_weight_shape[0];
        A_shape1 = encoder_layers_0_A_weight_shape[1];
        size_t nA = (size_t)A_shape0 * A_shape1;
        A_weight_f = dequant_int8_to_float(encoder_layers_0_A_weight_q, nA, encoder_layers_0_A_weight_scale);

        B_shape0 = encoder_layers_0_B_weight_shape[0];
        B_shape1 = encoder_layers_0_B_weight_shape[1];
        size_t nB = (size_t)B_shape0 * B_shape1;
        B_weight_f = dequant_int8_to_float(encoder_layers_0_B_weight_q, nB, encoder_layers_0_B_weight_scale);

        ESP_LOGI(TAG, "Loaded U(%d x %d) V(%d x %d) A(%d x %d) B(%d x %d)", U_shape0,U_shape1,V_shape0,V_shape1,A_shape0,A_shape1,B_shape0,B_shape1);
    }

    // pre layernorm
    {
        ln_dim = encoder_layers_0_pre_ln_weight_shape[0];
        pre_ln_gamma = dequant_int8_to_float(encoder_layers_0_pre_ln_weight_q, (size_t)ln_dim, encoder_layers_0_pre_ln_weight_scale);
        pre_ln_beta  = dequant_int8_to_float(encoder_layers_0_pre_ln_bias_q,  (size_t)ln_dim, encoder_layers_0_pre_ln_bias_scale);
        ESP_LOGI(TAG, "Loaded pre-ln params dim=%d", ln_dim);
    }

    // ffn
    {
        ffn_fc1_shape0 = encoder_layers_0_ffn_1_weight_shape[0];
        ffn_fc1_shape1 = encoder_layers_0_ffn_1_weight_shape[1];
        size_t n1 = (size_t)ffn_fc1_shape0 * ffn_fc1_shape1;
        ffn_fc1_weight = dequant_int8_to_float(encoder_layers_0_ffn_1_weight_q, n1, encoder_layers_0_ffn_1_weight_scale);
        ffn_fc1_bias = dequant_int8_to_float(encoder_layers_0_ffn_1_bias_q, (size_t)encoder_layers_0_ffn_1_bias_shape[0], encoder_layers_0_ffn_1_bias_scale);

        ffn_fc2_shape0 = encoder_layers_0_ffn_3_weight_shape[0];
        ffn_fc2_shape1 = encoder_layers_0_ffn_3_weight_shape[1];
        size_t n2 = (size_t)ffn_fc2_shape0 * ffn_fc2_shape1;
        ffn_fc2_weight = dequant_int8_to_float(encoder_layers_0_ffn_3_weight_q, n2, encoder_layers_0_ffn_3_weight_scale);
        ffn_fc2_bias = dequant_int8_to_float(encoder_layers_0_ffn_3_bias_q, (size_t)encoder_layers_0_ffn_3_bias_shape[0], encoder_layers_0_ffn_3_bias_scale);
        ESP_LOGI(TAG, "Loaded ffn shapes fc1=%d x %d fc2=%d x %d", ffn_fc1_shape0,ffn_fc1_shape1,ffn_fc2_shape0,ffn_fc2_shape1);
    }

    // layer c
    {
        layer_c_len = encoder_layers_0_c_shape[0];
        layer_c = dequant_int8_to_float(encoder_layers_0_c_q, (size_t)layer_c_len, encoder_layers_0_c_scale);
        ESP_LOGI(TAG, "Loaded layer c (K=%d)", layer_c_len);
    }

    // head
    {
        head_rows = head_2_weight_shape[0];
        head_cols = head_2_weight_shape[1];
        size_t hn = (size_t)head_rows * head_cols;
        head_weight_f = dequant_int8_to_float(head_2_weight_q, hn, head_2_weight_scale);
        head_bias_f = dequant_int8_to_float(head_2_bias_q, (size_t)head_2_bias_shape[0], head_2_bias_scale);
        ESP_LOGI(TAG, "Loaded head weights: %d x %d", head_rows, head_cols);
    }

    // dims
    D_MODEL = embed_shape0;
    R_DIM = U_shape0;
    K_DIM = A_shape0;
    INPUT_DIM = embed_shape1;

    ESP_LOGI(TAG, "Model dims: D=%d, R=%d, K=%d, INPUT_DIM=%d", D_MODEL, R_DIM, K_DIM, INPUT_DIM);
    ESP_LOGI(TAG, "Dequantization done.");
}

// ------------------------------------------------------------------
// Scratch allocation
// ------------------------------------------------------------------
bool alloc_scratch_once() {
    scratch_x_norm = (float*)ps_malloc(sizeof(float)*D_MODEL);
    scratch_phi = (float*)ps_malloc(sizeof(float)*D_MODEL);
    scratch_Pt = (float*)ps_malloc(sizeof(float)*R_DIM);
    scratch_Qt = (float*)ps_malloc(sizeof(float)*R_DIM);
    scratch_tmpA = (float*)ps_malloc(sizeof(float)*K_DIM);
    scratch_tmpB = (float*)ps_malloc(sizeof(float)*K_DIM);
    scratch_e = (float*)ps_malloc(sizeof(float)*K_DIM);
    scratch_G = (float*)ps_malloc(sizeof(float)*K_DIM);
    scratch_Mt = (float*)ps_malloc(sizeof(float)*R_DIM);
    scratch_Nt = (float*)ps_malloc(sizeof(float)*R_DIM);
    scratch_phi_prime = (float*)ps_malloc(sizeof(float)*D_MODEL);
    scratch_term1 = (float*)ps_malloc(sizeof(float)*D_MODEL);
    scratch_term2 = (float*)ps_malloc(sizeof(float)*D_MODEL);
    scratch_g = (float*)ps_malloc(sizeof(float)*D_MODEL);
    scratch_residual = (float*)ps_malloc(sizeof(float)*D_MODEL);
    scratch_x_next = (float*)ps_malloc(sizeof(float)*D_MODEL);
    scratch_ffn_out = (float*)ps_malloc(sizeof(float)*D_MODEL);
    scratch_logits = (float*)ps_malloc(sizeof(float)*head_rows);

    if (!scratch_x_norm || !scratch_phi || !scratch_Pt || !scratch_Qt || !scratch_tmpA || !scratch_tmpB || !scratch_e || !scratch_G
        || !scratch_Mt || !scratch_Nt || !scratch_phi_prime || !scratch_term1 || !scratch_term2 || !scratch_g || !scratch_residual
        || !scratch_x_next || !scratch_ffn_out || !scratch_logits) {
        ESP_LOGE(TAG, "Scratch allocation failed");
        return false;
    }
    return true;
}

// ------------------------------------------------------------------
// FFN forward (robust, uses tmp buffers or internal persist)
// ------------------------------------------------------------------
void ffn_forward(const float* fc1_w, int fc1_out, int fc1_in, const float* fc1_b,
                 const float* fc2_w, int fc2_out, int fc2_in, const float* fc2_b,
                 const float* x_in, float* out, float* tmp_buf, int tmp_buf_len) {
    if (tmp_buf_len >= fc1_out) {
        matvec(fc1_w, fc1_out, fc1_in, x_in, tmp_buf);
        for (int i = 0; i < fc1_out; ++i) tmp_buf[i] = fast_tanh(tmp_buf[i] + fc1_b[i]);
        matvec(fc2_w, fc2_out, fc2_in, tmp_buf, out);
        for (int i = 0; i < fc2_out; ++i) out[i] += fc2_b[i];
        return;
    }

    if (ffn_internal_tmp_capacity < fc1_out) {
        if (ffn_internal_tmp) { ps_free(ffn_internal_tmp); ffn_internal_tmp = nullptr; ffn_internal_tmp_capacity = 0; }
        size_t bytes = (size_t)fc1_out * sizeof(float);
        ffn_internal_tmp = (float*)ps_malloc(bytes);
        if (ffn_internal_tmp) {
            ffn_internal_tmp_capacity = fc1_out;
            ESP_LOGI(TAG, "ffn_forward: allocated persistent internal tmp (%d floats)", ffn_internal_tmp_capacity);
        } else {
            ESP_LOGW(TAG, "ffn_forward: failed to allocate persistent internal tmp (%d floats), will use chunked fallback", fc1_out);
        }
    }

    if (ffn_internal_tmp && ffn_internal_tmp_capacity >= fc1_out) {
        float* buf = ffn_internal_tmp;
        matvec(fc1_w, fc1_out, fc1_in, x_in, buf);
        for (int i = 0; i < fc1_out; ++i) buf[i] = fast_tanh(buf[i] + fc1_b[i]);
        matvec(fc2_w, fc2_out, fc2_in, buf, out);
        for (int i = 0; i < fc2_out; ++i) out[i] += fc2_b[i];
        return;
    }

    if (!tmp_buf || tmp_buf_len <= 0) {
        ESP_LOGE(TAG, "ffn_forward: no usable tmp buffer (tmp_buf_len=%d) and internal allocation failed", tmp_buf_len);
        return;
    }

    for (int k = 0; k < fc2_out; ++k) out[k] = 0.0f;

    int chunk_size = tmp_buf_len;
    for (int start = 0; start < fc1_out; start += chunk_size) {
        int this_chunk = fc1_out - start;
        if (this_chunk > chunk_size) this_chunk = chunk_size;

        for (int jj = 0; jj < this_chunk; ++jj) {
            int idx = start + jj;
            const float* wrow = fc1_w + (size_t)idx * fc1_in;
            float s = 0.0f;
            for (int t = 0; t < fc1_in; ++t) s += wrow[t] * x_in[t];
            tmp_buf[jj] = fast_tanh(s + fc1_b[idx]);
        }

        for (int k = 0; k < fc2_out; ++k) {
            const float* wrow2 = fc2_w + (size_t)k * fc2_in;
            float acc = out[k];
            for (int jj = 0; jj < this_chunk; ++jj) {
                acc += wrow2[start + jj] * tmp_buf[jj];
            }
            out[k] = acc;
        }
    }

    for (int k = 0; k < fc2_out; ++k) out[k] += fc2_b[k];
}

// ------------------------------------------------------------------
// Gate compute & streaming forward
// ------------------------------------------------------------------
void compute_gate_step(const float* x_norm, float* g_out, int d_model) {
    for (int i = 0; i < d_model; ++i) {
        float v = x_norm[i] * 0.1f;
        float s = 0.5f * (1.0f + tanhf(v));
        g_out[i] = 1e-6f + (1.0f - 1e-6f) * s;
    }
}

void forward_step_stream(const float* x_t, float* x_next_out) {
    // Uses scratch buffers
    layernorm_forward(x_t, D_MODEL, pre_ln_gamma, pre_ln_beta, scratch_x_norm);

    for (int i = 0; i < D_MODEL; ++i) scratch_phi[i] = fast_tanh(scratch_x_norm[i]);

    matvec(U_weight_f, U_shape0, U_shape1, scratch_phi, scratch_Pt);
    matvec(V_weight_f, V_shape0, V_shape1, scratch_phi, scratch_Qt);

    for (int r = 0; r < R_DIM; ++r) S_running[r] += scratch_Pt[r];

    matvec(A_weight_f, A_shape0, A_shape1, S_running, scratch_tmpA);
    matvec(B_weight_f, B_shape0, B_shape1, scratch_Qt, scratch_tmpB);

    for (int k = 0; k < K_DIM; ++k) {
        float sum = scratch_tmpA[k] + scratch_tmpB[k];
        if (layer_c) sum += layer_c[k];
        scratch_e[k] = fast_tanh(sum);
    }

    float gate_const = const_w * (const_s * const_s);
    for (int k = 0; k < K_DIM; ++k) scratch_G[k] = gate_const * scratch_e[k];

    for (int r = 0; r < R_DIM; ++r) {
        float sM = 0.0f, sN = 0.0f;
        for (int k = 0; k < K_DIM; ++k) {
            sM += scratch_G[k] * A_weight_f[k * R_DIM + r];
            sN += scratch_G[k] * B_weight_f[k * R_DIM + r];
        }
        scratch_Mt[r] = sM;
        scratch_Nt[r] = sN;
    }

    for (int r = 0; r < R_DIM; ++r) {
        M_pref[r] += scratch_Mt[r];
        N_pref[r] += scratch_Nt[r];
    }

    for (int i = 0; i < D_MODEL; ++i) scratch_phi_prime[i] = 1.0f - scratch_phi[i] * scratch_phi[i];

    for (int j = 0; j < D_MODEL; ++j) {
        float s = 0.0f;
        for (int r = 0; r < R_DIM; ++r) s += M_pref[r] * U_weight_f[r * D_MODEL + j];
        scratch_term1[j] = s * scratch_phi_prime[j];
    }
    for (int j = 0; j < D_MODEL; ++j) {
        float s = 0.0f;
        for (int r = 0; r < R_DIM; ++r) s += N_pref[r] * V_weight_f[r * D_MODEL + j];
        scratch_term2[j] = s * scratch_phi_prime[j];
    }

    compute_gate_step(scratch_x_norm, scratch_g, D_MODEL);

    for (int j = 0; j < D_MODEL; ++j)
        scratch_residual[j] = res_scale_val * scratch_g[j] * (scratch_term1[j] + scratch_term2[j]);

    for (int j = 0; j < D_MODEL; ++j) scratch_x_next[j] = x_t[j] + scratch_residual[j];

    ffn_forward(ffn_fc1_weight, ffn_fc1_shape0, ffn_fc1_shape1, ffn_fc1_bias,
                ffn_fc2_weight, ffn_fc2_shape0, ffn_fc2_shape1, ffn_fc2_bias,
                scratch_x_next, scratch_ffn_out, scratch_tmpB, K_DIM);

    for (int j = 0; j < D_MODEL; ++j) x_next_out[j] = scratch_x_next[j] + scratch_ffn_out[j];
}

// ------------------------------------------------------------------
// logits & argmax
// ------------------------------------------------------------------
int compute_logits_and_best(const float* feat, float* out_logits, int n_classes, int d_model, int* out_best_idx) {
    float best_val = -1e30f;
    int best_idx = -1;
    for (int c = 0; c < n_classes; ++c) {
        float s = 0.0f;
        const float* wrow = head_weight_f + (size_t)c * d_model;
        for (int j = 0; j < d_model; ++j) s += wrow[j] * feat[j];
        s += head_bias_f[c];
        out_logits[c] = s;
        if (s > best_val) { best_val = s; best_idx = c; }
    }
    if (out_best_idx) *out_best_idx = best_idx;
    return best_idx;
}

// ------------------------------------------------------------------
// head SGD
// ------------------------------------------------------------------
void head_sgd_update(const float* feat, int d_model, int n_classes, int target_label, float lr) {
    float maxz = -1e30f;
    for (int c = 0; c < n_classes; ++c) {
        float s = 0.0f;
        const float* wrow = head_weight_f + (size_t)c * d_model;
        for (int j = 0; j < d_model; ++j) s += wrow[j] * feat[j];
        s += head_bias_f[c];
        scratch_logits[c] = s;
        if (s > maxz) maxz = s;
    }
    float sumexp = 0.0f;
    for (int c = 0; c < n_classes; ++c) { scratch_logits[c] = expf(scratch_logits[c] - maxz); sumexp += scratch_logits[c]; }
    for (int c = 0; c < n_classes; ++c) scratch_logits[c] /= sumexp;

    for (int c = 0; c < n_classes; ++c) {
        float gz = scratch_logits[c];
        if (c == target_label) gz -= 1.0f;
        float g = lr * gz;
        float* wrow = head_weight_f + (size_t)c * d_model;
        for (int j = 0; j < d_model; ++j) wrow[j] -= g * feat[j];
        head_bias_f[c] -= g;
    }
}

// ------------------------------------------------------------------
// save/load head to NVS (validate minimal)
 // These functions use head_weight_f and head_bias_f passed from callers
// ------------------------------------------------------------------
esp_err_t save_head_to_nvs(const char* ns, const float* head_w, int rows, int cols, const float* head_b) {
    // Basic validation: ensure head arrays are finite
    for (size_t i = 0; i < (size_t)rows * (size_t)cols; ++i) if (!std::isfinite(head_w[i])) return ESP_ERR_INVALID_ARG;
    for (int i = 0; i < rows; ++i) if (!std::isfinite(head_b[i])) return ESP_ERR_INVALID_ARG;

    nvs_handle_t h;
    esp_err_t err = nvs_open(ns, NVS_READWRITE, &h);
    if (err != ESP_OK) return err;
    size_t wbytes = sizeof(float) * rows * cols;
    err = nvs_set_blob(h, "head_w", head_w, wbytes); if (err != ESP_OK) { nvs_close(h); return err; }
    err = nvs_set_blob(h, "head_b", head_b, sizeof(float) * rows); if (err != ESP_OK) { nvs_close(h); return err; }
    err = nvs_commit(h);
    nvs_close(h);
    return err;
}

esp_err_t load_head_from_nvs(const char* ns, float* head_w, int rows, int cols, float* head_b) {
    nvs_handle_t h;
    esp_err_t err = nvs_open(ns, NVS_READONLY, &h);
    if (err != ESP_OK) return err;

    size_t wbytes = sizeof(float) * (size_t)rows * (size_t)cols;
    float* tmp_w = (float*)ps_malloc(wbytes);
    if (!tmp_w) { nvs_close(h); return ESP_ERR_NO_MEM; }
    esp_err_t read_err = nvs_get_blob(h, "head_w", tmp_w, &wbytes);
    if (read_err != ESP_OK) { ps_free(tmp_w); nvs_close(h); return read_err; }

    size_t bbytes = sizeof(float) * (size_t)rows;
    float* tmp_b = (float*)ps_malloc(bbytes);
    if (!tmp_b) { ps_free(tmp_w); nvs_close(h); return ESP_ERR_NO_MEM; }
    read_err = nvs_get_blob(h, "head_b", tmp_b, &bbytes);
    if (read_err != ESP_OK) { ps_free(tmp_w); ps_free(tmp_b); nvs_close(h); return read_err; }

    nvs_close(h);

    // Validate
    size_t hn = (size_t)rows * (size_t)cols;
    for (size_t i = 0; i < hn; ++i) if (!std::isfinite(tmp_w[i])) { ps_free(tmp_w); ps_free(tmp_b); return ESP_ERR_INVALID_STATE; }
    for (int i = 0; i < rows; ++i) if (!std::isfinite(tmp_b[i])) { ps_free(tmp_w); ps_free(tmp_b); return ESP_ERR_INVALID_STATE; }

    memcpy(head_w, tmp_w, sizeof(float)*hn);
    memcpy(head_b, tmp_b, sizeof(float)*rows);
    ps_free(tmp_w); ps_free(tmp_b);

    return ESP_OK;
}

// ------------------------------------------------------------------
// I2S capture (safe memset + explicit fields to avoid aggregate-init issues)
// ------------------------------------------------------------------
static const int I2S_NUM = 0;
static const int I2S_SAMPLE_RATE = 16000;
static const int CAPTURE_SAMPLES = 512; // hop/frame size (adjust to match training)

bool i2s_capture_init() {
    i2s_config_t i2s_config;
    memset(&i2s_config, 0, sizeof(i2s_config));
    // explicit assignments
    i2s_config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
    i2s_config.sample_rate = I2S_SAMPLE_RATE;
    i2s_config.bits_per_sample = (i2s_bits_per_sample_t)16;
    // communication_format enum is deprecated in legacy header but still works
    i2s_config.communication_format = (i2s_comm_format_t)I2S_COMM_FORMAT_I2S_MSB;
    i2s_config.intr_alloc_flags = 0;
    // legacy aliases — zeroed by memset first
    i2s_config.dma_buf_count = 4;
    i2s_config.dma_buf_len = 256;
    i2s_config.use_apll = false;

    i2s_pin_config_t pin_config;
    memset(&pin_config, 0, sizeof(pin_config));
    // TODO: set actual pins for your board
    pin_config.bck_io_num = 45; // placeholder
    pin_config.ws_io_num = 46;  // placeholder
    pin_config.data_out_num = I2S_PIN_NO_CHANGE;
    pin_config.data_in_num = 47; // placeholder
    pin_config.mck_io_num = I2S_PIN_NO_CHANGE;

    esp_err_t res = i2s_driver_install((i2s_port_t)I2S_NUM, &i2s_config, 0, NULL);
    if (res != ESP_OK) { ESP_LOGE(TAG, "i2s_driver_install failed: %d", res); return false; }
    res = i2s_set_pin((i2s_port_t)I2S_NUM, &pin_config);
    if (res != ESP_OK) { ESP_LOGE(TAG, "i2s_set_pin failed: %d", res); return false; }
    ESP_LOGI(TAG, "I2S capture initialized (verify pins & PDM/PCM mode)");
    return true;
}

int i2s_capture_read(int16_t* buf, int max_samples) {
    size_t bytes_read = 0;
    size_t to_read_bytes = (size_t)max_samples * sizeof(int16_t);
    esp_err_t res = i2s_read((i2s_port_t)I2S_NUM, (void*)buf, to_read_bytes, &bytes_read, pdMS_TO_TICKS(200));
    if (res != ESP_OK) {
        ESP_LOGW(TAG, "i2s_read error %d", res);
        return 0;
    }
    return (int)(bytes_read / sizeof(int16_t));
}

// ------------------------------------------------------------------
// MFCC compute (placeholder/fallback)
// ------------------------------------------------------------------
void compute_mfcc_from_pcm_kissfft(const int16_t* pcm_samples, int pcm_len, float* mfcc_out, int mfcc_dim) {
    double energy = 1e-9;
    for (int i = 0; i < pcm_len; ++i) energy += double(pcm_samples[i]) * double(pcm_samples[i]);
    double l = 10.0 * log10(energy / pcm_len + 1e-12);
    float v = (float)((l + 100.0) / 100.0);
    for (int i = 0; i < mfcc_dim; ++i) mfcc_out[i] = v;
}

bool capture_and_compute_mfcc(float* out_mfcc) {
    static int16_t pcm_buf[CAPTURE_SAMPLES];
    int n = i2s_capture_read(pcm_buf, CAPTURE_SAMPLES);
    if (n <= 0) return false;
    if (mfcc_compute_from_pcm(pcm_buf, n, out_mfcc, INPUT_DIM, I2S_SAMPLE_RATE) == false) {
        compute_mfcc_from_pcm_kissfft(pcm_buf, n, out_mfcc, INPUT_DIM);
    }
    return true;
}

// ------------------------------------------------------------------
// Main app (uses functions defined above)
// ------------------------------------------------------------------
extern "C" void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_LOGI(TAG, "TSHM device booting...");

    load_and_dequant_all();

    if (D_MODEL <= 0 || R_DIM <= 0 || K_DIM <= 0 || INPUT_DIM <= 0) {
        ESP_LOGE(TAG, "Invalid model dims (D=%d R=%d K=%d INPUT_DIM=%d)", D_MODEL, R_DIM, K_DIM, INPUT_DIM);
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    // Check asset presence early and log
    bool use_asset_mfcc = false;
    int asset_frames = 0;
    int asset_dim = 0;
    const float* asset_data = nullptr;
    asset_frames = yes_01_mfcc_frames;
    asset_dim = yes_01_mfcc_dim;
    asset_data = yes_01_mfcc;
    if (asset_frames > 0 && asset_dim == INPUT_DIM && asset_data != nullptr) {
        use_asset_mfcc = true;
        ESP_LOGI(TAG, "ASSET DETECTED: yes_01_mfcc frames=%d dim=%d -> will use asset for demo/prediction", asset_frames, asset_dim);
    } else {
        if (asset_frames > 0 && asset_dim != INPUT_DIM) {
            ESP_LOGW(TAG, "Asset MFCC dim (%d) does not match expected INPUT_DIM (%d) -> ignoring asset", asset_dim, INPUT_DIM);
        } else {
            ESP_LOGI(TAG, "No usable MFCC asset found at startup -> demo/prediction may use all-zero MFCC");
        }
        use_asset_mfcc = false;
    }

    int n_classes = head_rows;
    int best_idx = -1;

    // streaming state alloc
    S_running = (float*)ps_malloc(sizeof(float)*R_DIM);
    M_pref = (float*)ps_malloc(sizeof(float)*R_DIM);
    N_pref = (float*)ps_malloc(sizeof(float)*R_DIM);
    if (!S_running || !M_pref || !N_pref) {
        ESP_LOGE(TAG, "Failed to allocate streaming state");
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }
    memset(S_running, 0, sizeof(float)*R_DIM);
    memset(M_pref, 0, sizeof(float)*R_DIM);
    memset(N_pref, 0, sizeof(float)*R_DIM);

    if (!alloc_scratch_once()) {
        ESP_LOGE(TAG, "Scratch allocation failed");
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    // load head from NVS if present
    if (load_head_from_nvs("head", head_weight_f, head_rows, head_cols, head_bias_f) == ESP_OK) {
        ESP_LOGI(TAG, "Loaded head from NVS");
    } else {
        ESP_LOGI(TAG, "No saved head in NVS (using exported head)");
    }

    if (!i2s_capture_init()) {
        ESP_LOGW(TAG, "I2S init failed/unsupported; capture will not work until configured correctly");
    }

    // allocate frame buffers
    float* mfcc_frame = (float*)ps_malloc(sizeof(float)*INPUT_DIM);
    float* embedded_frame = (float*)ps_malloc(sizeof(float)*D_MODEL);
    float* out_frame = (float*)ps_malloc(sizeof(float)*D_MODEL);
    float* out_frame_demo = (float*)ps_malloc(sizeof(float)*D_MODEL);
    if (!mfcc_frame || !embedded_frame || !out_frame || !out_frame_demo) {
        ESP_LOGE(TAG, "Failed to allocate frame buffers"); while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    // warm-up: pick asset first frame or zeros
    if (use_asset_mfcc) {
        const float* src0 = asset_data + 0 * asset_dim;
        for (int j = 0; j < INPUT_DIM; ++j) mfcc_frame[j] = src0[j];
        ESP_LOGI(TAG, "WARMUP: using assets_yes_01 first frame for warm-up / initial prediction");
    } else {
        for (int i = 0; i < INPUT_DIM; ++i) mfcc_frame[i] = 0.0f;
        ESP_LOGI(TAG, "WARMUP: using all-zero MFCC for warm-up / initial prediction");
    }

    // initial forward (warm-up)
    embed_apply(mfcc_frame, embedded_frame);
    int64_t t0 = esp_timer_get_time();
    forward_step_stream(embedded_frame, out_frame);
    int64_t t1 = esp_timer_get_time();
    ESP_LOGI(TAG, "Frame step latency (demo warm-up): %lld us", (long long)(t1 - t0));

    // DIAGNOSTICS (keeps but WILL NOT clobber demo output because demo computed later)
    {
        ESP_LOGI(TAG, "DIAG: Starting diagnostics checks");
        auto print_few = [&](const char* name, const float* p, int n, int show) {
            int s = (show < n) ? show : n;
            ESP_LOGI(TAG, "DIAG: %s (len=%d) first %d elems:", name, n, s);
            for (int i = 0; i < s; ++i) ESP_LOGI(TAG, "  %s[%d] = %f", name, i, p[i]);
        };
        if (embed_weight_f) { print_few("embed_weight_f", embed_weight_f, embed_shape0*embed_shape1, 6); }
        if (head_weight_f)  { print_few("head_weight_f", head_weight_f, head_rows*head_cols, 6); }
        if (head_bias_f)    { print_few("head_bias_f", head_bias_f, head_rows, 6); }

        // small synthetic diag that may clobber scratch buffers (but we'll recompute demo output later)
        float* test_mfcc = (float*)ps_malloc(sizeof(float) * INPUT_DIM);
        if (test_mfcc) {
            for (int i = 0; i < INPUT_DIM; ++i) test_mfcc[i] = 0.1f * (float)(i+1);
            float* embed_out = (float*)ps_malloc(sizeof(float) * D_MODEL);
            if (embed_out) {
                embed_apply(test_mfcc, embed_out);
                float* out_frame_test = (float*)ps_malloc(sizeof(float) * D_MODEL);
                if (out_frame_test) {
                    forward_step_stream(embed_out, out_frame_test);
                    int tmp_best=-2;
                    compute_logits_and_best(out_frame_test, scratch_logits, head_rows, D_MODEL, &tmp_best);
                    ESP_LOGI(TAG, "DIAG: tmp_best=%d", tmp_best);
                    ps_free(out_frame_test);
                }
                ps_free(embed_out);
            }
            ps_free(test_mfcc);
        }
        ESP_LOGI(TAG, "DIAG: Diagnostics checks complete");
    }

    // Now compute dedicated demo output from mfcc_frame (guaranteed source)
    memset(S_running, 0, sizeof(float)*R_DIM);
    memset(M_pref, 0, sizeof(float)*R_DIM);
    memset(N_pref, 0, sizeof(float)*R_DIM);

    ESP_LOGI(TAG, "DEMO INPUT INFO: Feature used for initial prediction:");
    if (use_asset_mfcc) {
        ESP_LOGI(TAG, "  -> assets_yes_01 MFCC (frames=%d dim=%d)", asset_frames, asset_dim);
    } else {
        ESP_LOGI(TAG, "  -> ALL-ZERO MFCC (no usable asset)");
    }
    int show_n = (INPUT_DIM < 8) ? INPUT_DIM : 8;
    for (int i = 0; i < show_n; ++i) ESP_LOGI(TAG, "  mfcc[%d] = %f", i, mfcc_frame[i]);
    if (INPUT_DIM > show_n) ESP_LOGI(TAG, "  ... (total dim=%d)", INPUT_DIM);

    embed_apply(mfcc_frame, embedded_frame);
    forward_step_stream(embedded_frame, out_frame_demo);
    compute_logits_and_best(out_frame_demo, scratch_logits, n_classes, D_MODEL, &best_idx);
    ESP_LOGI(TAG, "Predicted class BEFORE_SGD (from declared demo input): %d (logit=%f)", best_idx, scratch_logits[(best_idx >= 0 && best_idx < head_rows) ? best_idx : 0]);

    // optional demo training (adjust flag/params as you wish)
    const bool run_demo_training_once = true;
    if (run_demo_training_once) {
        const int demo_target_label = 0;
        const int demo_steps = 10;
        const float demo_lr = 0.05f;
        ESP_LOGI(TAG, "Applying %d SGD updates to head (target=%d, lr=%f) using the same demo feature...", demo_steps, demo_target_label, demo_lr);
        int64_t t_up_s = esp_timer_get_time();
        for (int it = 0; it < demo_steps; ++it) head_sgd_update(out_frame_demo, D_MODEL, n_classes, demo_target_label, demo_lr);
        int64_t t_up_e = esp_timer_get_time();
        ESP_LOGI(TAG, "Head SGD total time: %lld us", (long long)(t_up_e - t_up_s));
        if (save_head_to_nvs("head", head_weight_f, head_rows, head_cols, head_bias_f) == ESP_OK) ESP_LOGI(TAG, "Saved head to NVS");
    }

    compute_logits_and_best(out_frame_demo, scratch_logits, n_classes, D_MODEL, &best_idx);
    ESP_LOGI(TAG, "Predicted class AFTER_SGD (on same demo feature): %d (logit=%f)", best_idx, scratch_logits[(best_idx >= 0 && best_idx < head_rows) ? best_idx : 0]);

    // If asset present, play frame-by-frame
    if (use_asset_mfcc) {
        memset(S_running, 0, sizeof(float)*R_DIM);
        memset(M_pref, 0, sizeof(float)*R_DIM);
        memset(N_pref, 0, sizeof(float)*R_DIM);

        bool warned_zero_asset = false;
        for (int f = 0; f < asset_frames; ++f) {
            const float* src = asset_data + (size_t)f * asset_dim;
            for (int j = 0; j < INPUT_DIM; ++j) mfcc_frame[j] = src[j];

            if (!warned_zero_asset && is_all_zero(mfcc_frame, INPUT_DIM)) {
                ESP_LOGW(TAG, "WARNING: Asset MFCC frame %d appears to be all zeros. Predictions may be meaningless.", f + 1);
                warned_zero_asset = true;
            }

            embed_apply(mfcc_frame, embedded_frame);
            int64_t f0 = esp_timer_get_time();
            forward_step_stream(embedded_frame, out_frame);
            int64_t f1 = esp_timer_get_time();

            int tmp_best = -1;
            compute_logits_and_best(out_frame, scratch_logits, n_classes, D_MODEL, &tmp_best);
            float bestv = scratch_logits[(tmp_best >= 0 && tmp_best < head_rows) ? tmp_best : 0];

            ESP_LOGI(TAG, "Asset Frame %d/%d time: %lld us | Predicted id=%d logit=%f", f + 1, asset_frames, (long long)(f1 - f0), tmp_best, bestv);
            vTaskDelay(pdMS_TO_TICKS(20));
        }

        ESP_LOGI(TAG, "Finished asset playback. Entering idle loop.");
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    // Live capture loop (fallback)
    while (1) {
        bool ok = capture_and_compute_mfcc(mfcc_frame);
        if (!ok) { vTaskDelay(pdMS_TO_TICKS(50)); continue; }

        static bool warned_live_zero = false;
        if (!warned_live_zero && is_all_zero(mfcc_frame, INPUT_DIM)) {
            ESP_LOGW(TAG, "WARNING: Captured MFCC frame is all zeros (or near zero). This means either capture/MFCC is failing or input is silent.");
            warned_live_zero = true;
        }

        embed_apply(mfcc_frame, embedded_frame);
        int64_t f0 = esp_timer_get_time();
        forward_step_stream(embedded_frame, out_frame);
        int64_t f1 = esp_timer_get_time();

        int tmp_best = -1;
        compute_logits_and_best(out_frame, scratch_logits, n_classes, D_MODEL, &tmp_best);
        float bestv = scratch_logits[(tmp_best >= 0 && tmp_best < head_rows) ? tmp_best : 0];

        ESP_LOGI(TAG, "Frame time: %lld us | Predicted id=%d logit=%f", (long long)(f1 - f0), tmp_best, bestv);

        vTaskDelay(pdMS_TO_TICKS(20));
    }

    // never reached cleanup
    ps_free(mfcc_frame); ps_free(embedded_frame); ps_free(out_frame); ps_free(out_frame_demo);
}
