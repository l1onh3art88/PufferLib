#include <time.h>
#include "tradesim.h"
// #include "puffernet.h"

void demo() {
    TradeSim env = {
        .data_path = "resources/tradesim/data.bin",
        .width = 1920,
        .height = 1080,
    };
    allocate(&env);

    env.client = make_client(&env);

    c_reset(&env);
    while (!WindowShouldClose()) {
        // User can take control of the paddle
        // if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if(IsKeyPressed(KEY_Q)) {
                env.actions[0] = 1;
            }
            else if(IsKeyPressed(KEY_E)) {
                env.actions[0] = 2;
            }
            else if(IsKeyPressed(KEY_SPACE)){
                env.actions[0] = 3;
            } else {
                env.actions[0] = 0;
            }
        // }

        c_step(&env);
        c_render(&env);
    }
    free_allocated(&env);
    // close_client(env.client);
}

void test_performance(int timeout) {
    TradeSim env = {
        .data_path = "resources/tradesim/data.bin",
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
    return;
}

int main() {
    // demo();
    test_performance(10);
}
