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
#define REWARD_TYPE_SHARPE_RATIO 2

#define timestamp_char_len 19

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
    float long_win_pct;
    float short_win_pct;
    float overall_win_pct;
    float realized_pnl;
    float capital;
    float illegal_move_pct;
} Log;

typedef struct Client {
    float width;
    float height;
} Client;

typedef struct TradeSim {
    Client* client;
    Log log;
    double* observations;
    int* actions;
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
    int entry_step;
    int reward_type;
    float long_win_pct;
    float short_win_pct;
    float overall_win_pct;
    float reward_pnl_scale;
    float reward_illegal_move;
    float illegal_move_count;
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
    // read config settings
    fread(&env->initial_capital, sizeof(double), 1, file);
    fread(&env->position_size_fixed_dollar, sizeof(double), 1, file);
    fread(&env->pt_atr_mult, sizeof(double), 1, file);
    fread(&env->sl_atr_mult, sizeof(double), 1, file);
    fread(&env->warmup_steps, sizeof(int), 1, file);
    fread(&env->slippage_factor, sizeof(double), 1, file);
    fread(&env->transaction_fee_pct, sizeof(double), 1, file);
    fread(&env->max_steps_per_episode, sizeof(int), 1, file);
    fread(&env->reward_type, sizeof(int), 1, file);
    fclose(file);
}

void init(TradeSim* env) {
    env->tick = 0;
    env->_step = 1;
    env->short_trade_wins = 0;
    env->long_trade_wins = 0;
    env->entry_price = 0;
    env->entry_step = 0;
    read_data(env);
    env->returns = (double*)calloc(env->max_steps_per_episode, sizeof(double));
    env->returns[0] = env->initial_capital;
    env->long_win_pct = 0;
    env->short_win_pct = 0;
    env->overall_win_pct = 0;
    env->illegal_move_count = 0;
}

void allocate(TradeSim* env) {
    init(env);
    env->observations = (double*)calloc(MAX_OBSERVATIONS, sizeof(double));
    env->actions = (int*)calloc(1, sizeof(int));
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
    env->log.episode_return += env->rewards[0];
    env->log.score += env->realized_pnl;
    env->log.perf += env->rewards[0];
    env->log.long_win_pct = env->long_win_pct;
    env->log.short_win_pct = env->short_win_pct;
    env->log.overall_win_pct = env->overall_win_pct;
    env->log.n += 1;
    env->log.capital = env->capital;
    env->log.realized_pnl = env->realized_pnl;
    env->log.illegal_move_pct = env->illegal_move_count / (env->max_steps_per_episode);
}

void compute_observations(TradeSim* env) {
    memcpy(env->observations, env->features + (env->tick * (MAX_OBSERVATIONS-1)), (MAX_OBSERVATIONS-1) * sizeof(double));
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
    env->entry_step = 0;
    env->long_win_pct = 0;
    env->short_win_pct = 0;
    env->overall_win_pct = 0;
    env->illegal_move_count = 0;
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

double step_trade(TradeSim* env, float action) {
    int current_regime = env->regimes[env->_step - 1];
    double current_price = env->prices[env->_step];
    double current_atr = env->atrs[env->_step - 1];
    if(!legal_action(env,action)){
        env->illegal_move_count += 1;
        return -2;
    }
    int reset_internals = 0;
    if(env->entry_price == 0) {
        env->entry_price = current_price;
        env->entry_step = env->_step;
    }
    double base_commision = env->entry_price * fabs(env->position) * env->transaction_fee_pct;
    double base_slippage = env->entry_price * fabs(env->position) * env->slippage_factor;
    double final_commision  = base_commision*2;
    double final_slippage = base_slippage*2;
    if(action == BUY) {
        env->entry_price = current_price;
        env->entry_step = env->_step;
        env->long_trades += 1;
        env->position = fminf(env->initial_capital + env->unrealized_pnl + env->realized_pnl, env->position_size_fixed_dollar) / current_price;
        env->profit_target = current_price + env->pt_atr_mult * current_atr;
        env->stop_loss = current_price - env->sl_atr_mult * current_atr;
        env->unrealized_pnl = 0 + final_commision + final_slippage;
    }
    if(action == SELL) {
        env->entry_price = current_price;
        env->entry_step = env->_step;
        env->short_trades += 1;
        env->position = -fminf(env->initial_capital + env->unrealized_pnl + env->realized_pnl, env->position_size_fixed_dollar) / current_price;
        env->profit_target = current_price - env->pt_atr_mult * current_atr;
        env->stop_loss = current_price + env->sl_atr_mult * current_atr;
        env->unrealized_pnl = 0 - final_commision - final_slippage;
    }
    double trade_pnl = 0;
    double trade_pnl_pct = 0;
    if(env->position > 0 ){
        if(current_price >= env->profit_target) {
            reset_internals = 1;
            env->long_trade_wins += 1;
            trade_pnl = (env->profit_target - env->entry_price) * env->position;
            trade_pnl_pct = trade_pnl / (env->position * env->entry_price);
            env->realized_pnl += trade_pnl - final_commision - final_slippage;
            env->unrealized_pnl = 0;
        } else if(current_price <= env->stop_loss) {
            reset_internals = 1;
            trade_pnl = (env->stop_loss - env->entry_price) * env->position;
            trade_pnl_pct = trade_pnl / (env->position * env->entry_price);
            env->realized_pnl += trade_pnl - final_commision - final_slippage;
            env->unrealized_pnl = 0;
        } else if(action == CLOSE){
            reset_internals = 1;
            trade_pnl = (current_price - env->entry_price) * env->position;
            trade_pnl_pct = trade_pnl / (env->position * env->entry_price);
            if (trade_pnl > 0) {
                env->long_trade_wins += 1;
            }
            env->realized_pnl += trade_pnl - final_commision - final_slippage;
            env->unrealized_pnl = 0;
        } else {
            env->unrealized_pnl = ((current_price - env->entry_price) * env->position) - base_commision - base_slippage;
        }
    } else if (env->position < 0) {
        if(current_price <= env->profit_target) {
            reset_internals = 1;
            env->short_trade_wins += 1;
            trade_pnl = (env->profit_target - env->entry_price) * env->position;
            trade_pnl_pct = trade_pnl / (-env->position * env->entry_price);
            env->realized_pnl += trade_pnl - final_commision - final_slippage;
            env->unrealized_pnl = 0;
        } else if(current_price >= env->stop_loss) {
            reset_internals = 1;
            trade_pnl = (env->stop_loss - env->entry_price) * env->position;
            trade_pnl_pct = trade_pnl / (-env->position * env->entry_price);
            env->realized_pnl += trade_pnl - final_commision - final_slippage;
            env->unrealized_pnl = 0;
        } else if(action == CLOSE){
            reset_internals = 1;
            trade_pnl = (current_price - env->entry_price) * env->position;
            trade_pnl_pct = trade_pnl / (-env->position * env->entry_price);
            if (trade_pnl > 0) {
                env->short_trade_wins += 1;
            }
            env->realized_pnl += trade_pnl - final_commision - final_slippage;
            env->unrealized_pnl = 0;
        } else {
            env->unrealized_pnl = ((current_price - env->entry_price) * env->position) - base_commision - base_slippage;
        }
    }
    env->capital  = env->initial_capital + env->unrealized_pnl + env->realized_pnl;
    env->returns[env->_step] = env->capital;
    env->step_return = env->returns[env->_step] - env->returns[env->_step - 1];
    if(reset_internals) {
        env->position = 0;
        env->profit_target = 0;
        env->stop_loss = 0;
        env->entry_price = 0;
        env->entry_step = 0;
    }
    env->overall_win_pct = (env->short_trade_wins + env->long_trade_wins) / (env->short_trades + env->long_trades + 1e-5);
    env->short_win_pct = env->short_trade_wins / (env->short_trades + 1e-5);
    env->long_win_pct = env->long_trade_wins / (env->long_trades + 1e-5);

    return trade_pnl_pct;
}

void c_step(TradeSim* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;
    if(env->_step >= env->max_steps_per_episode) {
        add_log(env);
        c_reset(env);
    }

    int action = env->actions[0];
    env->tick += 1;
    double trade_pnl_pct = step_trade(env, action);
    env->_step += 1;
    if(env->reward_type == REWARD_TYPE_SIMPLE_PnL) {
        /*if(trade_pnl_pct > 0.0){
            env->rewards[0] = 1.0f;
        } else if(trade_pnl_pct < 0.0){
            env->rewards[0] = -1.0f;   
        }*/
        if(trade_pnl_pct == -2){
            env->rewards[0] = env->reward_illegal_move;
        } else {
            env->rewards[0] = trade_pnl_pct*env->reward_pnl_scale;
        }
        env->log.episode_return += env->rewards[0];
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
    DrawText(TextFormat("Step: %d", env->_step), 10, 10, 20, WHITE);
    DrawText(TextFormat("Capital: %f", env->capital), 10, 50, 20, WHITE);
    DrawText(TextFormat("Unrealized PnL: %f", env->unrealized_pnl), 10, 70, 20, WHITE);
    DrawText(TextFormat("Realized PnL: %f", env->realized_pnl), 10, 90, 20, WHITE);
    DrawText(TextFormat("Profit Target: %f", env->profit_target), 10, 110, 20, WHITE);
    DrawText(TextFormat("Stop Loss: %f", env->stop_loss), 10, 130, 20, WHITE);
    DrawText(TextFormat("Long Win Pct: %.2f", env->long_win_pct), 10, 150, 20, WHITE);
    DrawText(TextFormat("Short Win Pct: %.2f", env->short_win_pct), 10, 170, 20, WHITE);
    DrawText(TextFormat("Overall Win Pct: %.2f", env->overall_win_pct), 10, 190, 20, WHITE);
    // Draw a graph of the current price based on the current step and previous steps
    int graph_width = client->width * 0.75;
    int graph_height = client->height * 0.75;
    float x_margin = 55;
    float y_margin = 55;
    int start_x = (client->width - graph_width) / 2;
    int start_y = (client->height - graph_height) / 2;
    DrawRectangle(start_x, start_y, graph_width, graph_height, WHITE);

    // Find min and max prices for scaling
    double min_price = env->prices[0];
    double max_price = env->prices[0];
    for(int i = 0; i < env->_step; i++) {
        if(env->prices[i] < min_price) min_price = env->prices[i];
        if(env->prices[i] > max_price) max_price = env->prices[i];
    }
    
    // Add some padding to min/max
    double price_range = max_price - min_price;
    min_price -= price_range * 0.1;
    max_price += price_range * 0.1;
    price_range = max_price - min_price;

    // Draw axes
    // Y-axis
    DrawLine(start_x + x_margin, start_y, start_x + x_margin, start_y + graph_height, BLACK);
    // X-axis
    DrawLine(start_x + x_margin, start_y + graph_height - y_margin, start_x + graph_width, start_y + graph_height - y_margin, BLACK);

    // Draw y-axis ticks and labels
    int num_y_ticks = 5;
    for(int i = 0; i <= num_y_ticks; i++) {
        float y_pos = start_y + graph_height - (i * graph_height / num_y_ticks);
        float price = min_price + (i * price_range / num_y_ticks);
        // Draw tick mark
        DrawLine(start_x + x_margin - 5, y_pos, start_x + x_margin, y_pos, BLACK);
        // Draw price label
        DrawText(TextFormat("%.2f", price), start_x + x_margin - 45, y_pos - 10 - y_margin, 15, BLACK);
    }

    // Draw x-axis ticks and labels
    int num_x_ticks = 5;
    for(int i = 0; i <= num_x_ticks; i++) {
        float x_pos = start_x + x_margin + 25 + (i * (graph_width - x_margin)) / num_x_ticks;
        int step = (i * env->max_steps_per_episode) / num_x_ticks;
        // Draw tick mark
        DrawLine(x_pos, start_y + graph_height - y_margin, x_pos, start_y + graph_height - y_margin + 5, BLACK);
        // Draw step label
        DrawText(TextFormat("%d", step), x_pos - 10, start_y + graph_height - y_margin + 10, 15, BLACK);
    }
    
    // Draw price points
    for(int i = 0; i < env->_step; i++) {
        // Scale x position to fit graph width
        float x_pos = start_x + x_margin + 25 + (i * (graph_width - x_margin - 25)) / (env->max_steps_per_episode - 1);
        // Scale y position to fit graph height
        float y_pos = start_y + graph_height - y_margin - ((env->prices[i] - min_price) / price_range * graph_height);
        if(i == env->entry_step && env->position != 0) {
            // draw a rd or green arrow pointing at the cicle below
            if(env->position > 0) {
                DrawCircle(x_pos, y_pos, 3, GREEN);
            } else {
                DrawCircle(x_pos, y_pos, 3, RED);
            }
        } else {
            DrawCircle(x_pos, y_pos, 3, BLACK);
        }
    }
    EndDrawing();

    //PlaySound(client->sound);
}
