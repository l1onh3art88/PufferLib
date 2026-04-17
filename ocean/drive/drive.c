#include <time.h>
#include <unistd.h>
#include "drive.h"
#include "puffernet.h"

void demo() {
    Drive env = {
        .dynamics_model = CLASSIC,
        .human_agent_idx = 0,
        .reward_vehicle_collision = -0.1f,
        .reward_offroad_collision = -0.1f,
    };
    env.map_name = strdup("resources/drive/map_010.bin");
    allocate(&env);
    c_reset(&env);
    c_render(&env);
    Weights* weights = load_weights("resources/drive/drive_weights.bin");
    int logit_sizes[2] = {7, 13};
    PufferNet* net = make_puffernet(weights, env.active_agent_count, OBS_SIZE, 256, 4, logit_sizes, 2);
    int accel_delta = 2;
    int steer_delta = 4;
    while (!WindowShouldClose()) {
        float (*actions)[2] = (float(*)[2])env.actions;
        float obs_f32[env.active_agent_count * OBS_SIZE];
        for (int j = 0; j < env.active_agent_count * OBS_SIZE; j++)
            obs_f32[j] = bf16_to_f32(env.observations[j]);
        forward_puffernet(net, obs_f32, env.actions);
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            actions[env.human_agent_idx][0] = 3;
            actions[env.human_agent_idx][1] = 6;
            if(IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)){
                actions[env.human_agent_idx][0] += accel_delta;
                if(actions[env.human_agent_idx][0] > 6) actions[env.human_agent_idx][0] = 6;
            }
            if(IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)){
                actions[env.human_agent_idx][0] -= accel_delta;
                if(actions[env.human_agent_idx][0] < 0) actions[env.human_agent_idx][0] = 0;
            }
            if(IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)){
                actions[env.human_agent_idx][1] += steer_delta;
                if(actions[env.human_agent_idx][1] < 0) actions[env.human_agent_idx][1] = 0;
            }
            if(IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)){
                actions[env.human_agent_idx][1] -= steer_delta;
                if(actions[env.human_agent_idx][1] > 12) actions[env.human_agent_idx][1] = 12;
            }   
            if(IsKeyPressed(KEY_TAB)){
                env.human_agent_idx = (env.human_agent_idx + 1) % env.active_agent_count;
            }
        }
        c_step(&env);
        c_render(&env);
    }

    close_client(env.client);
    free_allocated(&env);
    free_puffernet(net);
    free(weights);
}

void performance_test() {
    long test_time = 10;
    Drive env = {
        .dynamics_model = CLASSIC,
        .human_agent_idx = 0,
    };
    env.map_name = strdup("resources/drive/map_942.bin");
    allocate(&env);
    c_reset(&env);


    long start = time(NULL);
    int i = 0;

    float (*actions)[2] = (float(*)[2])env.actions;
    while (time(NULL) - start < test_time) {
        for(int i =0; i < env.active_agent_count; i++){
            actions[i][0] = rand() % 7;
            actions[i][1] = rand() % 13;
        }
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (long)(i*env.active_agent_count) / (end - start));
    free_allocated(&env);
}

int main() {
    //demo();
    performance_test();
    return 0;
}
