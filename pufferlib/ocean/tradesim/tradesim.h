#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "raylib.h"

#define HOLD 0
#define BUY 1
#define SELL 2
#define CLOSE 3

#define MAX_OBSERVATIONS 414
#define N_STACK 1

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct Client {
    float width;
    float height;
} Client;

typedef struct TradeSim {
    Client* client;
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    int tick;
    int max_steps_per_episode;
    int warmup_steps;
    float initial_capital;
    float capital;
    float transaction_fee_pct;
    float position_size_fixed_dollar;
    int max_position;
    int min_position;
    int mode;
    float* features;
    float* prices;
    float* atrs;
    float* timestamps;
    float* regimes
    int position;
    int _step;
    float pt_atr_mult;
    float sl_atr_mult;
    float unrealized_pnl;
    float realized_pnl;
    float* short_trade_wins;
    float* long_trade_wins;
    float short_trades;
    float long_trades;
} TradeSim;

void init(TradeSim* env) {
    env->tick = 0;
   
}

void allocate(TradeSim* env) {
    init(env);
    env->observations = (float*)calloc(MAX_OBSERVATIONS, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void c_close(TradeSim* env) {
    free(env->prices);
    free(env->atrs);
    free(env->timestamps);
    free(env->regimes);
    free(env->features);
    free(env->short_trade_wins);
    free(env->long_trade_wins);
    free(env);
}

void free_allocated(TradeSim* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

void add_log(TradeSim* env) {
    env->log.episode_length += env->tick;
    env->log.episode_return += env->score;
    env->log.score += env->score;
    env->log.perf += env->score / (float)env->max_score;
    env->log.n += 1;
}

void compute_observations(TradeSim* env) {
    memcpy(env->observations, env->features + (env->tick * MAX_OBSERVATIONS), MAX_OBSERVATIONS * sizeof(float));
}

void c_reset(TradeSim* env) {
    env->score = 0;
    env->tick = 0;
    memcpy(env->observations, env->features, MAX_OBSERVATIONS * sizeof(float));
    compute_observations(env);
}

void step_trade(TradeSim* env, float action) {
    if(action != HOLD) {

    }

}

void c_step(TradeSim* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;

    float action = env->actions[0];
    for (int i = 0; i < env->frameskip; i++) {
        env->tick += 1;
        step_trade(env, action);
    }

    compute_observations(env);
}

Color BRICK_COLORS[6] = {RED, ORANGE, YELLOW, GREEN, SKYBLUE, BLUE};

static inline bool file_exists(const char* path) {
    return access(path, F_OK) != -1;
}

Client* make_client(TradeSim* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;

    InitWindow(env->width, env->height, "PufferLib Tradesim");
    SetTargetFPS(60);

    if (!found) {
        TraceLog(LOG_ERROR, "Failed to find puffers_128.png from current directory.");
        CloseWindow();
        free(client);
        exit(EXIT_FAILURE);
    }

    client->ball = LoadTexture(texturePath);
    TraceLog(LOG_INFO, "Resource path resolution: %s", texturePath);

    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(Breakout* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    Client* client = env->client;

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    EndDrawing();

    //PlaySound(client->sound);
}
