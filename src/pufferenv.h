#pragma once

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ini.h"
#include "raylib.h"
#ifdef PLATFORM_WEB
void puf_web_vsync(void);
#else
static void puf_web_vsync(void) {
}
#endif

typedef struct Env Env;
typedef struct Log Log;

typedef struct Agent {
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int policy;
} Agent;

// Env backend identity. Trainer: if (PUF_BACKEND == PUF_GPU) — not #if.
// GPU env sources #define PUF_BACKEND PUF_GPU before including this header.
#define PUF_CPU 0
#define PUF_GPU 1
#ifndef PUF_BACKEND
#define PUF_BACKEND PUF_CPU
#endif

// Shared env API. CPU: per-env Env*. GPU: Env* is device batch base; step/reset
// run the full vector inside the env. puf_bind_stream / puf_vec_create are GPU
// create-path hooks (CPU builds get stubs in pufferl).
void puf_init(Env* env, Dict* kwargs);
void puf_reset(Env* env);
void puf_step(Env* env);
// CPU: host Env*. GPU: device batch base; implementation D2Hs what it needs and draws.
void puf_render(Env* env);
void puf_close(Env* env);
void puf_log(Log* log, Dict* out);
// Bot ladder writes this between rungs so one PuffeRL can eval a whole ladder.
// Default no-op; envs with scripted opponents #define PUF_HAS_BOT_POLICY and
// assign env->bot_policy.
#ifndef PUF_HAS_BOT_POLICY
static inline void puf_set_bot_policy(Env* env, int bot_policy) {
}
#endif

typedef uint16_t bf16;

static inline bf16 f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

static inline float bf16_to_f32(bf16 b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}
