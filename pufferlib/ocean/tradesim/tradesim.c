#include <time.h>
#include "tradesim.h"
// #include "puffernet.h"

void demo() {
    TradeSim env = {
        .data_path = "resources/tradesim/data.bin",
        .width = 1000,
        .height = 1000,
    };
    allocate(&env);

    // env.client = make_client(&env);

    // c_reset(&env);
    // while (!WindowShouldClose()) {
    //     // User can take control of the paddle
    //     if (IsKeyDown(KEY_LEFT_SHIFT)) {
    //         if(env.continuous) {
    //             float move = GetMouseWheelMove();
    //             float clamped_wheel = fmaxf(-1.0f, fminf(1.0f, move));
    //             env.actions[0] = clamped_wheel;
    //         } else {
    //             env.actions[0] = 0.0;
    //             if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 1;
    //             if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 2;
    //         }
    //     } else {
    //         int* actions = (int*)env.actions;
    //         forward_linearlstm(net, env.observations, actions);
    //         env.actions[0] = actions[0];
    //     }

    //     c_step(&env);
    //     c_render(&env);
    // }
    // free_linearlstm(net);
    // free(weights);
    free_allocated(&env);
    // close_client(env.client);
}

void test_performance(int timeout) {
    TradeSim env = {
        .data_path = "resources/tradesim/data.bin",
        .slippage_factor = 0.0001,
        .transaction_fee_pct = 0.0001,
        .position_size_fixed_dollar = 10000,
        .pt_atr_mult = 1.0,
        .sl_atr_mult = 1.0,
    };
    allocate(&env);
    c_reset(&env);

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        env.actions[0] = rand() % 4;
        c_step(&env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = num_steps / (end - start);
    printf("Test Environment SPS: %f\n", sps);
    free_allocated(&env);
}

int main() {
    // demo();
    test_performance(10);
}
