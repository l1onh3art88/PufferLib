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

#define MIN_CANDLE_WIDTH 4
#define MAX_CANDLE_WIDTH 20
#define DEFAULT_CANDLE_WIDTH 8
#define DEFAULT_VISIBLE_CANDLES 150
#define ZOOM_FACTOR 1.2f

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
    
    // Viewport control
    float candle_width;
    int visible_candles;
    double view_start_time;  // Index of first visible candle
    double view_end_time;    // Index of last visible candle
    double view_min_price;   // Minimum visible price
    double view_max_price;   // Maximum visible price
    bool is_dragging;
    Vector2 drag_start;
    Vector2 last_mouse_pos;
    Font font;
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
    float episode_return;
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
    env->episode_return = 0;
    env->realized_pnl = 0;
    env->unrealized_pnl = 0;
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
    env->log.episode_return += env->episode_return;
    env->log.score += env->realized_pnl;
    env->log.perf += env->episode_return;
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
    env->episode_return = 0;
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
        if(env->position != 0){
            double commision = current_price * fabs(env->position) * env->transaction_fee_pct;
            double slippage = current_price * fabs(env->position) * env->slippage_factor;
            env->unrealized_pnl = ((current_price - env->entry_price) * env->position) - commision - slippage;
            env->capital  = env->initial_capital + env->unrealized_pnl + env->realized_pnl;
            env->returns[env->_step] = env->capital;
            env->step_return = env->returns[env->_step] - env->returns[env->_step - 1];
        }
        return -2;
    }
    int reset_internals = 0;
    if(env->entry_price == 0) {
        env->entry_price = current_price;
        env->entry_step = env->_step;
    }
    double commision; 
    double slippage; 
    if(action == BUY) {
        env->entry_price = current_price;
        env->entry_step = env->_step;
        env->long_trades += 1;
        env->position = fminf(env->initial_capital + env->unrealized_pnl + env->realized_pnl, env->position_size_fixed_dollar) / current_price;
        env->profit_target = current_price + env->pt_atr_mult * current_atr;
        env->stop_loss = current_price - env->sl_atr_mult * current_atr;
        commision = current_price * fabs(env->position) * env->transaction_fee_pct;
        slippage = current_price * fabs(env->position) * env->slippage_factor;
        env->unrealized_pnl = 0 - commision - slippage;
    }
    if(action == SELL) {
        env->entry_price = current_price;
        env->entry_step = env->_step;
        env->short_trades += 1;
        env->position = -fminf(env->initial_capital + env->unrealized_pnl + env->realized_pnl, env->position_size_fixed_dollar) / current_price;
        env->profit_target = current_price - env->pt_atr_mult * current_atr;
        env->stop_loss = current_price + env->sl_atr_mult * current_atr;
        commision = current_price * fabs(env->position) * env->transaction_fee_pct;
        slippage = current_price * fabs(env->position) * env->slippage_factor;
        env->unrealized_pnl = 0 - commision - slippage;
 
    }
    double trade_pnl = 0;
    double trade_pnl_pct = 0;
    if(env->position > 0 ){
        if(current_price >= env->profit_target) {
            reset_internals = 1;
            env->long_trade_wins += 1;
            trade_pnl = (env->profit_target - env->entry_price) * env->position;
            trade_pnl_pct = trade_pnl / (env->position * env->entry_price);
            commision = env->entry_price * fabs(env->position) * env->transaction_fee_pct *2.0;
            slippage = env->entry_price * fabs(env->position) * env->slippage_factor *2.0;
            env->realized_pnl += trade_pnl - commision - slippage;
            env->unrealized_pnl = 0;
        } else if(current_price <= env->stop_loss) {
            reset_internals = 1;
            trade_pnl = (env->stop_loss - env->entry_price) * env->position;
            trade_pnl_pct = trade_pnl / (env->position * env->entry_price);
            commision = env->entry_price * fabs(env->position) * env->transaction_fee_pct *2.0;
            slippage = env->entry_price * fabs(env->position) * env->slippage_factor *2.0;
            env->realized_pnl += trade_pnl - commision - slippage;
            env->unrealized_pnl = 0;
        } else if(action == CLOSE){
            reset_internals = 1;
            trade_pnl = (current_price - env->entry_price) * env->position;
            trade_pnl_pct = trade_pnl / (env->position * env->entry_price);
            if (trade_pnl > 0) {
                env->long_trade_wins += 1;
            }
            commision = env->entry_price * fabs(env->position) * env->transaction_fee_pct *2.0;
            slippage = env->entry_price * fabs(env->position) * env->slippage_factor *2.0;
            env->realized_pnl += trade_pnl - commision - slippage;
            env->unrealized_pnl = 0;
        } else {
            commision = env->entry_price * fabs(env->position) * env->transaction_fee_pct;
            slippage = env->entry_price * fabs(env->position) * env->slippage_factor;
            env->unrealized_pnl = ((current_price - env->entry_price) * env->position) - commision - slippage;
        }
    } else if (env->position < 0) {
        if(current_price <= env->profit_target) {
            reset_internals = 1;
            env->short_trade_wins += 1;
            trade_pnl = (env->profit_target - env->entry_price) * env->position;
            trade_pnl_pct = trade_pnl / (-env->position * env->entry_price);
            commision = env->entry_price * fabs(env->position) * env->transaction_fee_pct *2.0;
            slippage = env->entry_price * fabs(env->position) * env->slippage_factor *2.0;
            env->realized_pnl += trade_pnl - commision - slippage;
            env->unrealized_pnl = 0;
        } else if(current_price >= env->stop_loss) {
            reset_internals = 1;
            trade_pnl = (env->stop_loss - env->entry_price) * env->position;
            trade_pnl_pct = trade_pnl / (-env->position * env->entry_price);
            commision = env->entry_price * fabs(env->position) * env->transaction_fee_pct *2.0;
            slippage = env->entry_price * fabs(env->position) * env->slippage_factor *2.0;
            env->realized_pnl += trade_pnl - commision - slippage;
            env->unrealized_pnl = 0;
        } else if(action == CLOSE){
            reset_internals = 1;
            trade_pnl = (current_price - env->entry_price) * env->position;
            trade_pnl_pct = trade_pnl / (-env->position * env->entry_price);
            if (trade_pnl > 0) {
                env->short_trade_wins += 1;
            }
            commision = env->entry_price * fabs(env->position) * env->transaction_fee_pct *2.0;
            slippage = env->entry_price * fabs(env->position) * env->slippage_factor *2.0;
            env->realized_pnl += trade_pnl - commision - slippage;
            env->unrealized_pnl = 0;
        } else {
            commision = env->entry_price * fabs(env->position) * env->transaction_fee_pct;
            slippage = env->entry_price * fabs(env->position) * env->slippage_factor;
            env->unrealized_pnl = ((current_price - env->entry_price) * env->position) - commision - slippage;
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
        env->episode_return += env->rewards[0];
    } 
    compute_observations(env);
}

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_GREY = (Color){128, 128, 128, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};


Client* make_client(TradeSim* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;

    
    // Calculate Graph dimensions 
    int graph_width = client->width * 0.75;
    float x_margin = 55;
    float graph_content_width = graph_width - x_margin - 25;
    // Initialize viewport
    client->candle_width = DEFAULT_CANDLE_WIDTH;
    client->visible_candles = (int)(graph_content_width / client->candle_width);
    client->view_start_time = 0;
    client->view_end_time = client->visible_candles;
    client->is_dragging = false;
    
    // Set initial price range
    client->view_min_price = env->prices[0];
    client->view_max_price = env->prices[0];
    for(int i = 0; i < env->max_steps_per_episode; i++) {
        client->view_min_price = fmin(client->view_min_price, env->prices[i]);
        client->view_max_price = fmax(client->view_max_price, env->prices[i]);
    }
    // Add some padding
    double price_range = client->view_max_price - client->view_min_price;
    client->view_min_price -= price_range * 0.1;
    client->view_max_price += price_range * 0.1;
    printf("Initial viewport: start=%f, end=%f, min_price=%f, max_price=%f\n", 
    client->view_start_time, client->view_end_time, 
    client->view_min_price, client->view_max_price);
    InitWindow(env->width, env->height, "PufferLib Tradesim");
    client->font = LoadFontEx("resources/tradesim/OpenSans-Regular.ttf", 32, 0, 94);
    if (client->font.texture.id == 0) {  // Check if font failed to load
        TraceLog(LOG_WARNING, "Failed to load custom font, falling back to default font");
        client->font = GetFontDefault();
    } else {
        TraceLog(LOG_INFO, "Custom font loaded successfully");
    }
    SetTargetFPS(20);
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void update_viewport(Client* client, TradeSim* env) {
    // If we're near the end of the visible range, update the viewport to follow the current step
    if (env->_step > client->view_end_time - 20) {  // Start scrolling when we're within 20 candles of the end
        client->view_start_time = fmax(0, env->_step - client->visible_candles);
        client->view_end_time = fmin(client->view_start_time + client->visible_candles, env->_step);
    }
    if( env->_step >= env->max_steps_per_episode) {
        client->view_start_time = 0;
        client->view_end_time = client->visible_candles;
    }
}

void c_render(TradeSim* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    Client* client = env->client;

    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    if (IsKeyPressed(KEY_TAB)) ToggleFullscreen();

    // Update viewport
    update_viewport(client, env);

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    // Setup graph area
    int graph_width = client->width;
    int graph_height = client->height;
    float x_margin = 100;
    float y_margin =80;
    int start_x = 0;
    int start_y = 20;
    float stat_margin = 120;
    // Draw graph background
    DrawRectangle(start_x, start_y, graph_width, graph_height, PUFF_BACKGROUND);
    DrawTextEx(client->font, TextFormat("Step: %d", env->_step), (Vector2){stat_margin+10, 20}, 20, 1, PUFF_WHITE);
    DrawTextEx(client->font, TextFormat("Capital: %.2f", env->capital), (Vector2){stat_margin+10, 50}, 20, 1, PUFF_WHITE);
    DrawTextEx(client->font, TextFormat("Unrealized PnL: %.2f", env->unrealized_pnl), (Vector2){stat_margin+10, 70}, 20, 1, PUFF_WHITE);
    DrawTextEx(client->font, TextFormat("Realized PnL: %.2f", env->realized_pnl), (Vector2){stat_margin+10, 90}, 20, 1, PUFF_WHITE);
    DrawTextEx(client->font, TextFormat("Profit Target: %.4f", env->profit_target), (Vector2){stat_margin+10, 110}, 20, 1, PUFF_WHITE);
    DrawTextEx(client->font, TextFormat("Stop Loss: %.4f", env->stop_loss), (Vector2){stat_margin+10, 130}, 20, 1, PUFF_WHITE);
    DrawTextEx(client->font, TextFormat("Long Win Pct: %.2f", env->long_win_pct), (Vector2){stat_margin+10, 150}, 20, 1, PUFF_WHITE);
    DrawTextEx(client->font, TextFormat("Short Win Pct: %.2f", env->short_win_pct), (Vector2){stat_margin+10, 170}, 20, 1, PUFF_WHITE);
    DrawTextEx(client->font, TextFormat("Overall Win Pct: %.2f", env->overall_win_pct), (Vector2){stat_margin+10, 190}, 20, 1, PUFF_WHITE);
    if(env->position != 0){
        DrawTextEx(client->font, TextFormat("Entry Price: %.2f", env->entry_price), (Vector2){stat_margin+10, 210}, 20, 1, PUFF_WHITE);
    }
    if( env->position > 0 ){
        DrawTextEx(client->font, TextFormat("Long Position: %.2f", env->position), (Vector2){stat_margin+10, 230}, 20, 1, PUFF_WHITE);
    } else if( env->position < 0 ){
        DrawTextEx(client->font, TextFormat("Short Position: %.2f", env->position), (Vector2){stat_margin+10, 230}, 20, 1, PUFF_WHITE);
    }
    

    // Calculate the actual width available for candles
    float graph_content_width = graph_width - x_margin - 25;
    float total_candles = client->view_end_time - client->view_start_time;
    float candle_spacing = graph_content_width / total_candles;  // This will be our new spacing
    // Draw main axes
    DrawLine(start_x + x_margin, start_y, start_x + x_margin, start_y + graph_height - y_margin, PUFF_CYAN);
    DrawLine(start_x + x_margin, start_y + graph_height - y_margin, start_x + x_margin + graph_content_width, start_y + graph_height - y_margin, PUFF_CYAN);
    // Draw y-axis ticks and labels
    int num_y_ticks = 10;
    for(int i = 0; i <= num_y_ticks; i++) {
        float y_pos = start_y + graph_height - y_margin - (i * (graph_height - y_margin) / num_y_ticks);
        float price = client->view_min_price + (i * (client->view_max_price - client->view_min_price) / num_y_ticks);
        DrawLine(start_x + x_margin, y_pos, start_x + x_margin + graph_content_width, y_pos, PUFF_CYAN);
        DrawTextEx(client->font, TextFormat("%.2f", price), (Vector2){start_x + x_margin - 75, y_pos - 10}, 15, 1, WHITE);
    }

    

    // Draw x-axis ticks and labels
    int num_x_ticks = 5;
    for(int i = 0; i <= num_x_ticks; i++) {
        float relative_pos = (float)i / num_x_ticks;  // 0 to 1
        float x_pos = start_x + x_margin + (relative_pos * graph_content_width);
        int step = client->view_start_time + (relative_pos * total_candles);
        if (step >= 0 && step < env->max_steps_per_episode) {
            DrawLine(x_pos, start_y + graph_height - y_margin, x_pos, start_y, PUFF_CYAN);
            unsigned char timestamp[20];
            for(int j = 0; j < timestamp_char_len; j++) {
                timestamp[j] = env->timestamps[step*timestamp_char_len + j];
            }
            timestamp[timestamp_char_len] = '\0';
            char date[11] = {0};  // YYYY-MM-DD\0
        char time[9] = {0};   // HH:MM:SS\0
        strncpy(date, (char*)timestamp, 10);  // Copy first 10 chars (date)
        strncpy(time, (char*)timestamp + 11, 8);  // Copy 8 chars starting from position 11 (time)
        
        // Draw date and time on separate lines
        DrawTextEx(client->font, date, (Vector2){x_pos - 30, start_y + graph_height - y_margin + 10}, 15, 1, WHITE);
        DrawTextEx(client->font, time, (Vector2){x_pos - 20, start_y + graph_height - y_margin + 30}, 15, 1, WHITE);
        }
    }

    // Draw candles using the same spacing calculation
    for(int i = client->view_start_time; i < client->view_end_time && i < env->_step; i++) {
        float relative_pos = (float)(i - client->view_start_time) / total_candles;  // 0 to 1
        float x_pos = start_x + x_margin + (relative_pos * graph_content_width);
        
        // Get open and close prices
        double open_price = env->prices[i];
        double close_price = (i < env->_step - 1) ? env->prices[i + 1] : env->prices[i];
        // Calculate y positions
        float open_y = start_y + graph_height - y_margin - 
            ((open_price - client->view_min_price) / (client->view_max_price - client->view_min_price) * (graph_height - y_margin));
        float close_y = start_y + graph_height - y_margin - 
            ((close_price - client->view_min_price) / (client->view_max_price - client->view_min_price) * (graph_height - y_margin));
        
        // Draw candle
        Color candle_color = (close_price >= open_price) ? GREEN : RED;
        float body_top = fmin(open_y, close_y);
        float body_height = fabs(open_y - close_y);
        if (body_height < 2.0f) body_height = 2.0f;
        
        DrawRectangle(x_pos, body_top, client->candle_width, body_height, candle_color);
    }

    // Draw entry price with purple horizontal line at price
    if (env->position != 0){
        Color position_color = PUFF_CYAN;
        
        float line_thickness = 2.0f;  // Adjust this value to make the line thicker or thinner
    
        DrawLineEx(
            (Vector2){start_x + x_margin, start_y + graph_height - y_margin - 
                ((env->entry_price - client->view_min_price) / (client->view_max_price - client->view_min_price) * (graph_height - y_margin))},
            (Vector2){start_x + graph_width, start_y + graph_height - y_margin - 
                ((env->entry_price - client->view_min_price) / (client->view_max_price - client->view_min_price) * (graph_height - y_margin))},
            line_thickness,
            position_color
        );
    }
    

    EndDrawing();
}
