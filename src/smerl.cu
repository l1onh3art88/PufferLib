// Minimal multi-mode conditioning. One policy, K persistent behavior modes.
//
// A mode z is a property of a *rollout row*, fixed for the run — not resampled
// per episode. Train-time mode of minibatch row j is mode_ids[idx[j]].
//
// Forward: h = encoder(obs) + embed[z], then RNN / policy heads.
// Embeds train via PPO backprop + Adam (embed_lr). No discriminator, no
// diversity reward, no action-distance loss.
//
// Params live outside the primary weight buffer (`.smerl` sidecar). Disc weight
// slots remain in the sidecar for layout compatibility but are not trained.

#ifndef PUFFERLIB_SMERL_CU
#define PUFFERLIB_SMERL_CU

#include "models.cu"

#define SMERL_MAGIC   0x4C524D53u  // 'SMRL'
#define SMERL_VERSION 1

struct SmerlConfig {
    bool enabled;
    int num_modes;
    float bonus_coef;
    int disc_hidden;   // reserved / sidecar versioning; v1 disc is linear h->K
    float disc_lr;
    float embed_lr;
    float beta1, beta2, eps;
};

struct SmerlState {
    SmerlConfig cfg;
    SmerlCond cond;
    Allocator params;       // optimizable params — dumped verbatim to the sidecar
    Allocator opt;          // grads, Adam moments, step counter, mode/gate buffers, scratch
    IntTensor mode_ids;     // [total_agents] per-row z; -1 = unconditioned
    IntTensor mb_mode;      // [minibatch_segments] gathered per minibatch
    IntTensor gates;        // [num_modes] 1 = mode earns the diversity bonus
    FloatTensor adam_m, adam_v, adam_t;
    // Unused legacy buffers (kept so alloc layout / older call sites stay safe).
    PrecisionTensor div_rewards_TB;
    PrecisionTensor div_rewards_BT;
    PrecisionTensor div_advantages;
    PrecisionTensor zero_values;
    int total_agents;
    int minibatch_segments;
    int horizon;
    int embed_n;            // num_modes * hidden
    int disc_n;             // num_modes * hidden + num_modes
};

// Sidecar header. Arch fields are checked on load so a mismatched sidecar fails
// loudly instead of silently reinterpreting bytes.
struct SmerlHeader {
    uint32_t magic;
    uint32_t version;
    int32_t num_modes;
    int32_t hidden;
    int32_t disc_hidden;
    int32_t reserved;
    int64_t param_bytes;
};

__global__ void smerl_adam_kernel(float* __restrict__ w, float* __restrict__ g,
        float* __restrict__ m, float* __restrict__ v, const float* __restrict__ t_ptr,
        float lr, float b1, float b2, float eps, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float t = *t_ptr;
    float grad = g[idx];
    float mi = b1 * m[idx] + (1.0f - b1) * grad;
    float vi = b2 * v[idx] + (1.0f - b2) * grad * grad;
    m[idx] = mi;
    v[idx] = vi;
    // Bias correction reads t from device memory so this stays cudagraph-safe.
    float mhat = mi / (1.0f - powf(b1, t));
    float vhat = vi / (1.0f - powf(b2, t));
    w[idx] -= lr * mhat / (sqrtf(vhat) + eps);
}

__global__ void smerl_tick_kernel(float* t_ptr) { *t_ptr += 1.0f; }

// Gather this minibatch's per-row modes. idx[j] is a segment == an agent row,
// which is exactly what mode_ids is indexed by.
__global__ void smerl_gather_modes(int* __restrict__ out, const int* __restrict__ mode_ids,
        const int* __restrict__ idx, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) out[j] = mode_ids[idx[j]];
}

// Float GEMM matching puf_mm's layout convention (row-major logical, cublasEx).
// out[M, N] = op(A)[M, K] @ op(B)[K, N] with the same op wiring as cublasGemmExDense.
static inline void smerl_f32_gemm(
        cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, const float* A, const float* B, float* C,
        cudaStream_t stream, float alpha = 1.0f, float beta = 0.0f) {
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    cublasHandle_t handle = cublas_get_handle();
    cublasSetStream(handle, stream);
    cublasGemmEx(handle, op_b, op_a, N, M, K, &alpha,
        B, CUDA_R_32F, ldb, A, CUDA_R_32F, lda, &beta,
        C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

__global__ void smerl_cast_h_to_f32(float* __restrict__ dst,
        const precision_t* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = to_float(src[idx]);
}

__global__ void smerl_add_bias(float* __restrict__ logits,
        const float* __restrict__ b, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * K) logits[idx] += b[idx % K];
}

// Softmax CE: write g = p - one_hot(z) and per-row metrics. No atomics —
// metrics are reduced deterministically in smerl_reduce_metrics.
__global__ void smerl_disc_ce_grads(float* __restrict__ g,
        const float* __restrict__ logits, const int* __restrict__ modes,
        float* __restrict__ row_loss, float* __restrict__ row_hit,
        float* __restrict__ row_valid, int N, int TT, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int z = modes[i / TT];
    float* gi = g + (long)i * K;
    if (z < 0 || z >= K) {
        for (int k = 0; k < K; k++) gi[k] = 0.0f;
        row_loss[i] = 0.0f;
        row_hit[i] = 0.0f;
        row_valid[i] = 0.0f;
        return;
    }
    const float* li = logits + (long)i * K;
    float max_l = li[0];
    for (int k = 1; k < K; k++) max_l = fmaxf(max_l, li[k]);
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += expf(li[k] - max_l);
    float inv = 1.0f / fmaxf(sum, 1e-20f);
    float log_qz = (li[z] - max_l) - logf(fmaxf(sum, 1e-20f));

    int pred = 0;
    float best = li[0];
    for (int k = 1; k < K; k++) {
        if (li[k] > best) { best = li[k]; pred = k; }
    }
    row_loss[i] = -log_qz;
    row_hit[i] = (pred == z) ? 1.0f : 0.0f;
    row_valid[i] = 1.0f;

    for (int k = 0; k < K; k++) {
        float p = expf(li[k] - max_l) * inv;
        gi[k] = p - ((k == z) ? 1.0f : 0.0f);
    }
}

// Sequential reduce in fixed order i=0..N-1 — bit-stable metrics.
__global__ void smerl_reduce_metrics(
        const float* __restrict__ row_loss, const float* __restrict__ row_hit,
        const float* __restrict__ row_valid, float* __restrict__ loss_acc,
        float* __restrict__ acc_acc, float* __restrict__ n_acc, int N) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float L = 0.0f, A = 0.0f, V = 0.0f;
    for (int i = 0; i < N; i++) {
        L += row_loss[i];
        A += row_hit[i];
        V += row_valid[i];
    }
    *loss_acc += L;
    *acc_acc += A;
    *n_acc += V;
}

// db[k] += sum_i g[i, k]  — one thread owns k, fixed loop order, no atomics.
__global__ void smerl_disc_bias_grad(float* __restrict__ db,
        const float* __restrict__ g, int N, int K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    float acc = 0.0f;
    for (int i = 0; i < N; i++) acc += g[(long)i * K + k];
    db[k] += acc;
}

// Write pure diversity reward into div_out (NOT env rewards):
//   r_div = bonus_coef * (log q(z|s) + log K)  if gate[z] else 0
// Soft-clamp r_div alone to [-1,1] for numerical safety; env clamp stays separate.
__global__ void smerl_write_div_from_logits(precision_t* __restrict__ div_out,
        const float* __restrict__ logits, const int* __restrict__ modes,
        const int* __restrict__ gates, float bonus_coef, float log_k,
        int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int z = modes[i];
    if (z < 0 || z >= K || !gates[z] || bonus_coef == 0.0f) {
        div_out[i] = from_float(0.0f);
        return;
    }
    const float* li = logits + (long)i * K;
    float max_l = li[0];
    for (int k = 1; k < K; k++) max_l = fmaxf(max_l, li[k]);
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += expf(li[k] - max_l);
    float log_qz = (li[z] - max_l) - logf(fmaxf(sum, 1e-20f));
    float r = bonus_coef * (log_qz + log_k);
    r = fminf(1.0f, fmaxf(-1.0f, r));
    div_out[i] = from_float(r);
}

// After select_copy built mb_adv = A_task (and returns = V + A_task), add A_div
// into mb advantages only — value targets stay task-pure.
__global__ void smerl_add_div_to_mb_adv(precision_t* __restrict__ mb_adv,
        const precision_t* __restrict__ div_adv, const int* __restrict__ idx,
        int mb_segs, int horizon) {
    int j = blockIdx.x;
    if (j >= mb_segs) return;
    int src = idx[j];
    precision_t* dst = mb_adv + (long)j * horizon;
    const precision_t* src_row = div_adv + (long)src * horizon;
    for (int t = threadIdx.x; t < horizon; t += blockDim.x) {
        dst[t] = from_float(to_float(dst[t]) + to_float(src_row[t]));
    }
}

// logits = h @ W^T + b   (h cast to f32 first)
static void smerl_disc_logits(SmerlCond* c, PrecisionTensor h, int N,
        cudaStream_t stream) {
    int H = c->hidden;
    int K = c->num_modes;
    long n = (long)N * H;
    smerl_cast_h_to_f32<<<grid_size((int)n), BLOCK_SIZE, 0, stream>>>(
        c->h_fp32.data, h.data, (int)n);
    // logits[N, K] = h_fp32[N, H] @ W[K, H]^T
    smerl_f32_gemm(CUBLAS_OP_N, CUBLAS_OP_T, N, K, H,
        c->h_fp32.data, c->disc_w.data, c->disc_logits.data, stream);
    smerl_add_bias<<<grid_size(N * K), BLOCK_SIZE, 0, stream>>>(
        c->disc_logits.data, c->disc_b.data, N, K);
}

static void smerl_create(SmerlState* s, const SmerlConfig& cfg, int hidden,
        int total_agents, int minibatch_segments, int horizon, int agents_per_buffer,
        ulong seed) {
    s->cfg = cfg;
    s->total_agents = total_agents;
    s->minibatch_segments = minibatch_segments;
    s->horizon = horizon;
    int K = cfg.num_modes;
    s->embed_n = K * hidden;
    s->disc_n = K * hidden + K;

    s->cond.num_modes = K;
    s->cond.hidden = hidden;
    s->cond.mode_ids = nullptr;
    s->cond.div_out = nullptr;
    s->cond.gates = nullptr;
    s->cond.bonus_coef = cfg.bonus_coef;
    s->cond.any_gate_on = 0;
    s->cond.max_rows = agents_per_buffer > minibatch_segments * horizon
        ? agents_per_buffer : minibatch_segments * horizon;
    s->cond.disc_train_fn = nullptr;
    s->cond.write_div_fn = nullptr;

    s->cond.embed = {.shape = {K, hidden}};
    s->cond.disc_w = {.shape = {K, hidden}};
    s->cond.disc_b = {.shape = {K}};
    alloc_register(&s->params, &s->cond.embed);
    alloc_register(&s->params, &s->cond.disc_w);
    alloc_register(&s->params, &s->cond.disc_b);

    s->cond.embed_grad = {.shape = {K, hidden}};
    s->cond.disc_w_grad = {.shape = {K, hidden}};
    s->cond.disc_b_grad = {.shape = {K}};
    s->adam_m = {.shape = {s->embed_n + s->disc_n}};
    s->adam_v = {.shape = {s->embed_n + s->disc_n}};
    s->adam_t = {.shape = {1}};
    s->mode_ids = {.shape = {total_agents}};
    s->mb_mode = {.shape = {minibatch_segments}};
    s->gates = {.shape = {K}};
    s->cond.disc_logits = {.shape = {s->cond.max_rows, K}};
    s->cond.disc_g = {.shape = {s->cond.max_rows, K}};
    s->cond.h_fp32 = {.shape = {s->cond.max_rows, hidden}};
    s->cond.row_loss = {.shape = {s->cond.max_rows}};
    s->cond.row_hit = {.shape = {s->cond.max_rows}};
    s->cond.row_valid = {.shape = {s->cond.max_rows}};
    s->cond.disc_loss_acc = {.shape = {1}};
    s->cond.disc_acc_acc = {.shape = {1}};
    s->cond.disc_n_acc = {.shape = {1}};
    s->div_rewards_TB = {.shape = {horizon, total_agents}};
    s->div_rewards_BT = {.shape = {total_agents, horizon}};
    s->div_advantages = {.shape = {total_agents, horizon}};
    s->zero_values = {.shape = {total_agents, horizon}};
    alloc_register(&s->opt, &s->cond.embed_grad);
    alloc_register(&s->opt, &s->cond.disc_w_grad);
    alloc_register(&s->opt, &s->cond.disc_b_grad);
    alloc_register(&s->opt, &s->adam_m);
    alloc_register(&s->opt, &s->adam_v);
    alloc_register(&s->opt, &s->adam_t);
    alloc_register(&s->opt, &s->mode_ids);
    alloc_register(&s->opt, &s->mb_mode);
    alloc_register(&s->opt, &s->gates);
    alloc_register(&s->opt, &s->cond.disc_logits);
    alloc_register(&s->opt, &s->cond.disc_g);
    alloc_register(&s->opt, &s->cond.h_fp32);
    alloc_register(&s->opt, &s->cond.row_loss);
    alloc_register(&s->opt, &s->cond.row_hit);
    alloc_register(&s->opt, &s->cond.row_valid);
    alloc_register(&s->opt, &s->cond.disc_loss_acc);
    alloc_register(&s->opt, &s->cond.disc_acc_acc);
    alloc_register(&s->opt, &s->cond.disc_n_acc);
    alloc_register(&s->opt, &s->div_rewards_TB);
    alloc_register(&s->opt, &s->div_rewards_BT);
    alloc_register(&s->opt, &s->div_advantages);
    alloc_register(&s->opt, &s->zero_values);

    alloc_create(&s->params);
    alloc_create(&s->opt);

    // Same host-curand pattern as puf_normal_init (kernels.cu): temp buffer,
    // GenerateNormal, copy. Seed is caller-supplied (hypers.seed + 0xSMRL + rank);
    // independent of Philox rollout RNG and of the policy kaiming counter.
    //
    // GenerateNormal is async on the default stream — never write straight into
    // the live param slab and memset disc_b without a barrier (that raced).
    long n_params = s->params.total_bytes / sizeof(float);
    long n_gen = (n_params % 2 == 0) ? n_params : n_params + 1;  // GenerateNormal needs even n
    float* tmp = nullptr;
    cudaMalloc(&tmp, n_gen * sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandSetGeneratorOffset(gen, 0);
    curandGenerateNormal(gen, tmp, n_gen, 0.0f, 0.02f);
    curandDestroyGenerator(gen);
    cudaMemcpy(s->params.mem, tmp, n_params * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(tmp);
    cudaDeviceSynchronize();
    // Bias starts at 0 so modes are equiprobable under a cold disc.
    cudaMemset(s->cond.disc_b.data, 0, K * sizeof(float));

    // Default to unconditioned everywhere; Python assigns real modes at setup.
    // Until then an enabled-but-unassigned run behaves exactly like a plain one.
    cudaMemset(s->mode_ids.data, 0xFF, total_agents * sizeof(int));  // = -1
    cudaMemset(s->mb_mode.data, 0xFF, minibatch_segments * sizeof(int));
    cudaMemset(s->gates.data, 0, K * sizeof(int));
    s->cond.gates = s->gates.data;
    s->cond.any_gate_on = 0;
    // zero_values stays zero for div GAE (never written after create).
    cudaMemset(s->zero_values.data, 0, numel(s->zero_values.shape) * sizeof(precision_t));
    cudaMemset(s->div_rewards_TB.data, 0, numel(s->div_rewards_TB.shape) * sizeof(precision_t));
    cudaDeviceSynchronize();
}

static void smerl_destroy(SmerlState* s) {
    alloc_free(&s->params);
    alloc_free(&s->opt);
}

// One Adam step on SMERL params, then zero the grads for the next accumulation.
// Called inside the captured train region — all pointers fixed, step counter on
// device, so it graphs cleanly. Embed and disc use independent learning rates
// but share one Adam step counter (same t for bias correction is fine).
static void smerl_optim_step(SmerlState* s, cudaStream_t stream) {
    smerl_tick_kernel<<<1, 1, 0, stream>>>(s->adam_t.data);

    // Layout in adam_m/v matches params layout: [embed | disc_w | disc_b].
    // grads live in separate tensors; pack into contiguous optim regions via
    // direct kernels on each slice.
    int K = s->cfg.num_modes;
    int H = s->cond.hidden;
    int embed_n = s->embed_n;
    int disc_w_n = K * H;
    int disc_b_n = K;

    // v2: only mode embeds are optimized. Disc weights stay in the sidecar for
    // layout compatibility but are not trained (action-distance replaces CE).
    smerl_adam_kernel<<<grid_size(embed_n), BLOCK_SIZE, 0, stream>>>(
        s->cond.embed.data, s->cond.embed_grad.data,
        s->adam_m.data, s->adam_v.data, s->adam_t.data,
        s->cfg.embed_lr, s->cfg.beta1, s->cfg.beta2, s->cfg.eps, embed_n);

    (void)disc_w_n;
    (void)disc_b_n;
    cudaMemsetAsync(s->cond.embed_grad.data, 0, embed_n * sizeof(float), stream);
    cudaMemsetAsync(s->cond.disc_w_grad.data, 0, disc_w_n * sizeof(float), stream);
    cudaMemsetAsync(s->cond.disc_b_grad.data, 0, disc_b_n * sizeof(float), stream);
}

static void smerl_set_modes(SmerlState* s, const int* host_modes) {
    cudaMemcpy(s->mode_ids.data, host_modes,
        s->total_agents * sizeof(int), cudaMemcpyHostToDevice);
}

static void smerl_set_gates(SmerlState* s, const int* host_gates) {
    cudaMemcpy(s->gates.data, host_gates, s->cfg.num_modes * sizeof(int),
        cudaMemcpyHostToDevice);
    int any = 0;
    for (int i = 0; i < s->cfg.num_modes; i++) {
        if (host_gates[i]) { any = 1; break; }
    }
    s->cond.any_gate_on = any;
}

// Force every row to one mode. Used by heldout eval to score a single mode.
static void smerl_force_mode(SmerlState* s, int mode) {
    int* host = (int*)malloc(s->total_agents * sizeof(int));
    for (int i = 0; i < s->total_agents; i++) host[i] = mode;
    smerl_set_modes(s, host);
    free(host);
    // The train-side gather is bypassed during eval, so pin mb_mode too.
    int* mb = (int*)malloc(s->minibatch_segments * sizeof(int));
    for (int i = 0; i < s->minibatch_segments; i++) mb[i] = mode;
    cudaMemcpy(s->mb_mode.data, mb, s->minibatch_segments * sizeof(int),
        cudaMemcpyHostToDevice);
    free(mb);
}

// Minimal: no disc / r_div hooks.
static void smerl_bind_hooks(SmerlState* s) {
    s->cond.disc_train_fn = nullptr;
    s->cond.write_div_fn = nullptr;
}

// Zero the full (T,B) diversity reward buffer (call at the start of each rollout).
static void smerl_zero_div_rewards(SmerlState* s, cudaStream_t stream) {
    if (s == nullptr) return;
    puf_zero(&s->div_rewards_TB, stream);
}

static void smerl_save(SmerlState* s, const char* path) {
    SmerlHeader h = {
        .magic = SMERL_MAGIC,
        .version = SMERL_VERSION,
        .num_modes = s->cfg.num_modes,
        .hidden = s->cond.hidden,
        .disc_hidden = s->cfg.disc_hidden,
        .reserved = 0,
        .param_bytes = s->params.total_bytes,
    };
    std::vector<char> buf(s->params.total_bytes);
    cudaMemcpy(buf.data(), s->params.mem, s->params.total_bytes, cudaMemcpyDeviceToHost);
    std::vector<int> gates(s->cfg.num_modes);
    cudaMemcpy(gates.data(), s->gates.data, s->cfg.num_modes * sizeof(int),
        cudaMemcpyDeviceToHost);
    // Persist as uint8 for a compact stable sidecar layout.
    std::vector<unsigned char> gates_u8(s->cfg.num_modes);
    for (int i = 0; i < s->cfg.num_modes; i++) gates_u8[i] = gates[i] ? 1 : 0;

    FILE* f = fopen(path, "wb");
    if (!f) throw std::runtime_error(std::string("Failed to open ") + path + " for writing");
    fwrite(&h, sizeof(h), 1, f);
    fwrite(buf.data(), 1, buf.size(), f);
    fwrite(gates_u8.data(), 1, gates_u8.size(), f);
    fclose(f);
}

static void smerl_load(SmerlState* s, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("Failed to open ") + path + " for reading");
    SmerlHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1) {
        fclose(f);
        throw std::runtime_error(std::string("Truncated smerl sidecar: ") + path);
    }
    auto fail = [&](const std::string& msg) {
        fclose(f);
        throw std::runtime_error("smerl sidecar " + std::string(path) + ": " + msg);
    };
    if (h.magic != SMERL_MAGIC) fail("bad magic");
    if (h.version != SMERL_VERSION) fail("version " + std::to_string(h.version) +
        ", expected " + std::to_string(SMERL_VERSION));
    if (h.num_modes != s->cfg.num_modes) fail("num_modes " + std::to_string(h.num_modes) +
        ", expected " + std::to_string(s->cfg.num_modes));
    if (h.hidden != s->cond.hidden) fail("hidden " + std::to_string(h.hidden) +
        ", expected " + std::to_string(s->cond.hidden));
    if (h.disc_hidden != s->cfg.disc_hidden) fail("disc_hidden " + std::to_string(h.disc_hidden) +
        ", expected " + std::to_string(s->cfg.disc_hidden));
    if (h.param_bytes != s->params.total_bytes) fail("param_bytes " +
        std::to_string(h.param_bytes) + ", expected " + std::to_string(s->params.total_bytes));

    std::vector<char> buf(h.param_bytes);
    if ((int64_t)fread(buf.data(), 1, buf.size(), f) != h.param_bytes) fail("truncated params");
    std::vector<unsigned char> gates_u8(s->cfg.num_modes);
    if ((int)fread(gates_u8.data(), 1, gates_u8.size(), f) != s->cfg.num_modes) fail("truncated gates");
    fclose(f);

    std::vector<int> gates(s->cfg.num_modes);
    for (int i = 0; i < s->cfg.num_modes; i++) gates[i] = gates_u8[i] ? 1 : 0;
    cudaMemcpy(s->params.mem, buf.data(), buf.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(s->gates.data, gates.data(), gates.size() * sizeof(int),
        cudaMemcpyHostToDevice);
}

// Legacy metric hook (always zeros — disc / action-div removed).
static void smerl_pop_disc_metrics(SmerlState* s, float* loss_out, float* acc_out) {
    (void)s;
    *loss_out = 0.0f;
    *acc_out = 0.0f;
}

#endif  // PUFFERLIB_SMERL_CU
