#include <time.h>
#include "go.h"
#include "puffernet.h"

typedef struct GoNet GoNet;
struct GoNet {
    int num_agents;
    float* obs_2d;
    float* obs_1d;
    Conv2D* conv1;
    ReLU* relu1;
    Conv2D* conv2;
    Linear* flat;
    CatDim1* cat;
    ReLU* relu2;
    Linear* proj;
    ReLU* relu3;
    Linear* actor;
    Linear* value_fn;
};
GoNet* init_gonet(Weights* weights, int num_agents, int grid_size) {
    GoNet* net = calloc(1, sizeof(GoNet));
    int hidden_size = 128;
    int cnn_channels = 64;
    int output_size = grid_size - 4;
    int cnn_flat_size = cnn_channels * output_size * output_size;

    net->num_agents = num_agents;
    net->obs_2d = calloc(num_agents * grid_size * grid_size * 2, sizeof(float)); // 2 channels for player and opponent
    net->obs_1d = calloc(num_agents * 2, sizeof(float)); // 2 additional features

    net->conv1 = make_conv2d(weights, num_agents, grid_size, grid_size, 2, cnn_channels, 3, 1);
    net->relu1 = make_relu(num_agents, cnn_channels * output_size * output_size);
    net->conv2 = make_conv2d(weights, num_agents, output_size, output_size, cnn_channels, cnn_channels, 3, 1);
    net->flat = make_linear(weights, num_agents, 2, 32);
    net->cat = make_cat_dim1(num_agents, cnn_flat_size, 32);
    net->relu2 = make_relu(num_agents, cnn_flat_size + 32);
    net->proj = make_linear(weights, num_agents, cnn_flat_size + 32, hidden_size);
    net->relu3 = make_relu(num_agents, hidden_size);
    net->actor = make_linear(weights, num_agents, hidden_size, grid_size*grid_size + 1); // Replace with actual action space size
    net->value_fn = make_linear(weights, num_agents, hidden_size, 1);

    return net;
}

void free_gonet(GoNet* net) {
    free(net->obs_2d);
    free(net->obs_1d);
    free(net->conv1);
    free(net->relu1);
    free(net->conv2);
    free(net->flat);
    free(net->cat);
    free(net->relu2);
    free(net->proj);
    free(net->relu3);
    free(net->actor);
    free(net->value_fn);
    free(net);
}

void forward(GoNet* net, float* observations, int* actions, int grid_size) {
    int full_board = grid_size * grid_size;
    
    // Clear previous observations
    memset(net->obs_2d, 0, net->num_agents * grid_size * grid_size * 2 * sizeof(float));
    
    // Reshape observations into 2D boards and additional features
    float (*obs_2d)[2][grid_size][grid_size] = (float (*)[2][grid_size][grid_size])net->obs_2d;
    float (*obs_1d)[2] = (float (*)[2])net->obs_1d;
    
    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b * (full_board * 2 + 2);  // offset for each batch
        
        // Process black stones board
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                obs_2d[b][0][i][j] = observations[b_offset + i*grid_size + j];
            }
        }
        
        // Process white stones board
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                obs_2d[b][1][i][j] = observations[b_offset + full_board + i*grid_size + j];
            }
        }
        
        // Process additional features
        obs_1d[b][0] = observations[b_offset + full_board * 2];
        obs_1d[b][1] = observations[b_offset + full_board * 2 + 1];
    }

    // Forward pass through the network
    conv2d(net->conv1, net->obs_2d);
    relu(net->relu1, net->conv1->output);
    conv2d(net->conv2, net->relu1->output);

    linear(net->flat, net->obs_1d);

    cat_dim1(net->cat, net->conv2->output, net->flat->output);
    relu(net->relu2, net->cat->output);
    linear(net->proj, net->relu2->output);
    relu(net->relu3, net->proj->output);
    
    linear(net->actor, net->relu3->output);
    linear(net->value_fn, net->relu3->output);

    // Get action by taking argmax of actor output
    for (int i = 0; i < net->num_agents; i++) {
        float max_val = net->actor->output[i * (grid_size * grid_size + 1)];
        int max_idx = 0;
        for (int j = 1; j < grid_size * grid_size + 1; j++) {
            float val = net->actor->output[i * (grid_size * grid_size + 1) + j];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        actions[i] = max_idx;
        printf("action[%d]: %d\n", i, actions[i]);
    }
}

void demo(int grid_size) {

    CGo env = {
        .width = 950,
        .height = 64*(grid_size+1),
        .grid_size = grid_size,
        .board_width = 64*(grid_size+1) + 400,
        .board_height = 64*(grid_size+1),
        .grid_square_size = 64,
        .moves_made = 0,
        .komi = 7.5,
        .reward_move_pass = -0.25,
        .reward_move_invalid = -0.1,
        .reward_move_valid = 0.1
    };

    Weights* weights = load_weights("resources/go_weights.bin", 254867);
    GoNet* net = init_gonet(weights, 1, grid_size);
    allocate(&env);
    reset(&env);
 
    Client* client = make_client(env.width, env.height);

    while (!WindowShouldClose()) {
        // User can take control of the paddle
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = -1;

            if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                Vector2 mousePos = GetMousePosition();
        
                // Calculate the offset for the board
                int boardOffsetX = env.grid_square_size;
                int boardOffsetY = env.grid_square_size;
                
                // Adjust mouse position relative to the board
                int relativeX = mousePos.x - boardOffsetX;
                int relativeY = mousePos.y - boardOffsetY;
                
                // Calculate cell indices for the corners
                int cellX = (relativeX + env.grid_square_size / 2) / env.grid_square_size;
                int cellY = (relativeY + env.grid_square_size / 2) / env.grid_square_size;
                
                // Ensure the click is within the game board
                if (cellX >= 0 && cellX <= env.grid_size && cellY >= 0 && cellY <= env.grid_size) {
                    // Calculate the point index (1-19) based on the click position
                    int pointIndex = cellY * (env.grid_size) + cellX + 1; 
                    env.actions[0] = (unsigned short)pointIndex;
                }
            // Check if pass button is clicked
                int passButtonX = env.width - 300;
                int passButtonY = 200;
                int passButtonWidth = 100;
                int passButtonHeight = 50;

                if (mousePos.x >= passButtonX && mousePos.x <= passButtonX + passButtonWidth &&
                    mousePos.y >= passButtonY && mousePos.y <= passButtonY + passButtonHeight) {
                    env.actions[0] = 0; // Send action 0 for pass
                }
            }
        }
        else {
            forward(net, env.observations, env.actions, grid_size);
        }

        if (env.actions[0] != -1) {
            step(&env);
        }

        render(client, &env);

    }
    close_client(client);
    free_allocated(&env);
}

void performance_test() {
    long test_time = 10;
    CGo env = {
        .width = 1000,
        .height = 800,
        .grid_size = 9,
        .board_width = 600,
        .board_height = 600,
        .grid_square_size = 600/9,
        .moves_made = 0,
        .komi = 7.5,
	.reward_move_pass = -0.25,
	.reward_move_invalid = -0.1,
	.reward_move_valid = 0.1
    };
    allocate(&env);
    reset(&env);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand() % (env.grid_size)*(env.grid_size);
        step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated(&env);
}

int main() {
    demo(7);
    // performance_test();
    return 0;
}
