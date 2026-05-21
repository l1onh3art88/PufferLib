#include "robocode.h"

int main() {
    Robocode env = {0};
    env.num_agents = 1;
    env.num_bots = 1;
    env.reward_damage = 0.01;
    env.width = 800;
    env.height = 600;
    allocate_env(&env);
    c_reset(&env);

    Client* client = make_client(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = 2;
        env.actions[1] = 4;
        env.actions[2] = 5;
        env.actions[3] = 5;
        env.actions[4] = 0;


        //if (IsKeyPressed(KEY_ESCAPE)) break;
        if (IsKeyDown(KEY_W)) env.actions[0] = 3.0f;
        if (IsKeyDown(KEY_S)) env.actions[0] = 1.0f;
        if (IsKeyDown(KEY_A)) env.actions[1] = 3.0f;
        if (IsKeyDown(KEY_D)) env.actions[1] = 5.0f;
        if (IsKeyDown(KEY_Q)) env.actions[2] = 4.0f;
        if (IsKeyDown(KEY_E)) env.actions[2] = 6.0f;
        if (IsKeyDown(KEY_LEFT)) env.actions[3] = 0.0f;
        if (IsKeyDown(KEY_RIGHT)) env.actions[3] = 8.0f;
        if (IsKeyDown(KEY_SPACE)) env.actions[4] = 1.0f;

        c_step(&env);
        c_render(&env);
    }
    c_close(&env);
    CloseWindow();
    return 0;
}
