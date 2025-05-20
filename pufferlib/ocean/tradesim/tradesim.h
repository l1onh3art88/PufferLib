#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
#include <time.h>


#define HOLD 0
#define BUY 1
#define SELL 2
#define CLOSE 3

#define MAX_OBSERVATIONS 414
#define N_STACK 1

#define REWARD_TYPE_SIMPLE_PnL 0
#define REWARD_TYPE_SORTINO_RATIO 1

#define timestamp_char_len 19

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
    double* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    int tick;
    int max_steps_per_episode;
    int warmup_steps;
    int num_features;
    double initial_capital;
    double capital;
    double transaction_fee_pct;
    double position_size_fixed_dollar;
    int max_position;
    int min_position;
    int mode;
    double* features;
    double* prices;
    double* atrs;
    unsigned char* timestamps;
    int* regimes;
    double position;
    int _step;
    double profit_target;
    double stop_loss;
    double pt_atr_mult;
    double sl_atr_mult;
    double unrealized_pnl;
    double realized_pnl;
    int short_trade_wins;
    int long_trade_wins;
    int short_trades;
    int long_trades;
    char* data_path;
    double* means;
    double* stds;
    double step_return;
    double* returns;
    float width;
    float height;
    double slippage_factor;
    double entry_price;
    int reward_type;
} TradeSim;

void read_data(TradeSim* env) {
    FILE* file = fopen(env->data_path, "rb");
    if (file == NULL) {
        TraceLog(LOG_ERROR, "Failed to open data file: %s", env->data_path);
        exit(EXIT_FAILURE);
    }
    // read rows and cols
    fread(&env->max_steps_per_episode, sizeof(int), 1, file);
    fread(&env->num_features, sizeof(int), 1, file);
    // read means
    env->means = (double*)calloc(env->num_features, sizeof(double));
    fread(env->means, sizeof(double), env->num_features, file);
    // read stds
    env->stds = (double*)calloc(env->num_features, sizeof(double));
    fread(env->stds, sizeof(double), env->num_features, file);
    // read prices
    env->prices = (double*)calloc(env->max_steps_per_episode, sizeof(double));
    fread(env->prices, sizeof(double), env->max_steps_per_episode, file);
    // read atrs
    env->atrs = (double*)calloc(env->max_steps_per_episode, sizeof(double));
    fread(env->atrs, sizeof(double), env->max_steps_per_episode, file);
    // read timestamps
    env->timestamps = (unsigned char*)calloc(env->max_steps_per_episode*timestamp_char_len, sizeof(unsigned char));
    for (int i = 0; i < env->max_steps_per_episode; i++) {
        fread(env->timestamps + (i*timestamp_char_len), sizeof(unsigned char), timestamp_char_len, file);
    }
    // read regimes
    env->regimes = (int*)calloc(env->max_steps_per_episode, sizeof(int));
    fread(env->regimes, sizeof(int), env->max_steps_per_episode, file);
    // read data
    env->features = (double*)calloc(env->max_steps_per_episode * env->num_features, sizeof(double));
    fread(env->features, sizeof(double), env->max_steps_per_episode * env->num_features, file);
    fclose(file);
}

void init(TradeSim* env) {
    env->tick = 0;
    env->_step = 1;
    env->short_trade_wins = 0;
    env->long_trade_wins = 0;
    env->entry_price = 0;
    read_data(env);
    env->returns = (double*)calloc(env->max_steps_per_episode, sizeof(double));
    env->returns[0] = env->initial_capital;
    env->reward_type = REWARD_TYPE_SIMPLE_PnL;
}

void allocate(TradeSim* env) {
    init(env);
    env->observations = (double*)calloc(MAX_OBSERVATIONS, sizeof(double));
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
    free(env->means);
    free(env->stds);
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
    env->log.episode_return += env->step_return;
    env->log.score += env->step_return;
    env->log.perf += env->step_return;
    env->log.n += 1;
}

void compute_observations(TradeSim* env) {
    memcpy(env->observations, env->features + (env->tick * (MAX_OBSERVATIONS-1)), (MAX_OBSERVATIONS-1) * sizeof(float));
    env->observations[MAX_OBSERVATIONS-1] = env->position;
}

void c_reset(TradeSim* env) {
    env->tick = 0;
    env->_step = 1;
    env->position = 0;
    env->capital = env->initial_capital;
    env->unrealized_pnl = 0;
    env->realized_pnl = 0;
    env->short_trades = 0;
    env->long_trades = 0;
    env->short_trade_wins = 0;
    env->long_trade_wins = 0;
    env->entry_price = 0;
    compute_observations(env);
}

int legal_action(TradeSim* env, float action) {
    if(action == BUY && env->position!=0) {
        return 0;
    }
    if(action == SELL && env->position!=0) {
        return 0;
    }
    if(action == CLOSE && env->position==0) {
        return 0;
    }
    return 1;
}

void step_trade(TradeSim* env, float action) {
    int current_regime = env->regimes[env->_step - 1];
    double current_price = env->prices[env->_step];
    double current_atr = env->atrs[env->_step - 1];
    if(!legal_action(env,action)){
        env->log.episode_return -= 0.1;
        env->rewards[0] = -0.1;
        return;
    }
    int reset_internals = 0;
    if(env->entry_price == 0) {
        env->entry_price = current_price;
    }
    double base_commision = env->entry_price * fabs(env->position) * env->transaction_fee_pct;
    double base_slippage = env->entry_price * fabs(env->position) * env->slippage_factor;

    double final_commision  = base_commision*2;
    double final_slippage = base_slippage*2;
    if(action == BUY) {
        env->entry_price = current_price;
        env->long_trades += 1;
        env->position = fminf(env->initial_capital + env->unrealized_pnl + env->realized_pnl, env->position_size_fixed_dollar) / current_price;
        env->profit_target = current_price + env->pt_atr_mult * current_atr;
        env->stop_loss = current_price - env->sl_atr_mult * current_atr;
        env->unrealized_pnl = 0 + final_commision + final_slippage;
    }
    if(action == SELL) {
        env->entry_price = current_price;
        env->short_trades += 1;
        env->position = -fminf(env->initial_capital + env->unrealized_pnl + env->realized_pnl, env->position_size_fixed_dollar) / current_price;
        env->profit_target = current_price - env->pt_atr_mult * current_atr;
        env->stop_loss = current_price + env->sl_atr_mult * current_atr;
        env->unrealized_pnl = 0 - final_commision - final_slippage;
    }
    
    if(env->position > 0 ){
        if(current_price >= env->profit_target) {
            env->long_trade_wins += 1;
            double trade_pnl = (env->profit_target - env->entry_price) * env->position;
            env->realized_pnl += trade_pnl - final_commision - final_slippage;
            env->unrealized_pnl = 0;
        } else if(current_price <= env->stop_loss) {
            double trade_pnl = (env->stop_loss - env->entry_price) * env->position;
            env->realized_pnl += trade_pnl - final_commision - final_slippage;
            env->unrealized_pnl = 0;
        } else if(action == CLOSE){
            double trade_pnl = (current_price - env->entry_price) * env->position;
            env->realized_pnl += trade_pnl - final_commision - final_slippage;
            env->unrealized_pnl = 0;
        } else {
            env->unrealized_pnl = ((current_price - env->entry_price) * env->position) - base_commision - base_slippage;
        }
    } else if (env->position < 0) {
        if(current_price <= env->profit_target) {
            env->short_trade_wins += 1;
            double trade_pnl = (env->entry_price - env->profit_target) * env->position;
            env->realized_pnl += trade_pnl - final_commision - final_slippage;
            env->unrealized_pnl = 0;
        } else if(current_price >= env->stop_loss) {
            double trade_pnl = (env->entry_price - env->stop_loss) * env->position;
            env->realized_pnl += trade_pnl - final_commision - final_slippage;
            env->unrealized_pnl = 0;
        } else if(action == CLOSE){
            double trade_pnl = (env->entry_price - current_price) * env->position;
            env->realized_pnl += trade_pnl - final_commision - final_slippage;
            env->unrealized_pnl = 0;
        } else {
            env->unrealized_pnl = ((env->entry_price - current_price) * env->position) - base_commision - base_slippage;
        }
    }
    env->capital  = env->initial_capital + env->unrealized_pnl + env->realized_pnl;
    env->returns[env->_step] = env->capital;
    env->step_return = env->returns[-1] - env->returns[-2];
    if(reset_internals) {
        env->position = 0;
        env->profit_target = 0;
        env->stop_loss = 0;
        env->entry_price = 0;
    }
}

void c_step(TradeSim* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;
    if(env->_step >= env->max_steps_per_episode) {
        c_reset(env);
    }

    float action = env->actions[0];
    env->tick += 1;
    step_trade(env, action);
    env->_step += 1;
    if(env->reward_type == REWARD_TYPE_SIMPLE_PnL) {
        env->rewards[0] = env->step_return;
        env->log.episode_return += env->step_return;
    } 
    compute_observations(env);
}

Client* make_client(TradeSim* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;

    InitWindow(env->width, env->height, "PufferLib Tradesim");
    SetTargetFPS(60);
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(TradeSim* env) {
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
