#include "drive.h"
#define NUM_ATNS 2
#define ACT_SIZES {7, 13}
#define OBS_TENSOR_T FloatTensor

#define MAP_BINARY_DIR "drive_data/binaries"

#define MY_VEC_INIT
#define MY_VEC_RESET
#define Env Drive
#include "vecenv.h"

// Returns the map id and writes the capped agent count to *agent_count_out.
static int sample_valid_map(unsigned int* rng, int num_maps, int* agent_count_out) {
    while (1) {
        int map_id = rand_r(rng) % num_maps;
        char map_file[512];
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", MAP_BINARY_DIR, map_id);
        Env temp_env = {0};
        temp_env.map_name = strdup(map_file);
        init(&temp_env);
        int count = temp_env.active_agent_count < MAX_AGENTS
                  ? temp_env.active_agent_count : MAX_AGENTS;
        c_close(&temp_env);
        if (count > 0) {
            *agent_count_out = count;
            return map_id;
        }
    }
}

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts, Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int num_maps = (int)dict_get(env_kwargs, "num_maps")->value;
    int agents_per_buffer = total_agents / num_buffers;

    float reward_vehicle_collision = dict_get(env_kwargs, "reward_vehicle_collision")->value;
    float reward_offroad_collision = dict_get(env_kwargs, "reward_offroad_collision")->value;
    float reward_goal_post_respawn = dict_get(env_kwargs, "reward_goal_post_respawn")->value;
    float reward_vehicle_collision_post_respawn = dict_get(env_kwargs, "reward_vehicle_collision_post_respawn")->value;
    int human_agent_idx = (int)dict_get(env_kwargs, "human_agent_idx")->value;

    // Verify that the path has valid binaries
    char first_map[512];
    snprintf(first_map, sizeof(first_map), "%s/map_%03d.bin", MAP_BINARY_DIR, 0);
    FILE* test_fp = fopen(first_map, "rb");
    if (!test_fp) {
        printf("ERROR: Cannot find map files at %s/\n", MAP_BINARY_DIR);
        *num_envs_out = 0;
        return NULL;
    }
    fclose(test_fp);

    // Build per-env layout with lazy map sampling (no upfront scan).
    // For each slot, randomly pick a map, probe it for agents, retry if invalid.
    int max_envs = agents_per_buffer * num_buffers;
    int* env_map_ids = (int*)malloc(max_envs * sizeof(int));
    int* env_max_agents = (int*)malloc(max_envs * sizeof(int));
    int total_envs = 0;
    unsigned int map_set_rng = 42;

    for (int b = 0; b < num_buffers; b++) {
        buffer_env_starts[b] = total_envs;
        int buffer_agents = 0;
        while (buffer_agents < agents_per_buffer) {
            int cap;
            int map_id = sample_valid_map(&map_set_rng, num_maps, &cap);
            int remaining = agents_per_buffer - buffer_agents;
            if (cap <= remaining) {
                env_map_ids[total_envs] = map_id;
                env_max_agents[total_envs] = cap;
                buffer_agents += cap;
                total_envs++;
            } else {
                // Map has more agents than slots remaining; fill the rest with 1-agent envs
                while (buffer_agents < agents_per_buffer) {
                    int fcap;
                    int fmap_id = sample_valid_map(&map_set_rng, num_maps, &fcap);
                    env_map_ids[total_envs] = fmap_id;
                    env_max_agents[total_envs] = 1;
                    buffer_agents++;
                    total_envs++;
                }
            }
        }
        buffer_env_counts[b] = total_envs - buffer_env_starts[b];
    }

    printf("total envs: %d\n", total_envs);

    // Initialize all envs
    Env* envs = (Env*)calloc(total_envs, sizeof(Env));
    for (int i = 0; i < total_envs; i++) {
        srand(i);
        char map_file[512];
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", MAP_BINARY_DIR, env_map_ids[i]);
        Env* env = &envs[i];
        memset(env, 0, sizeof(Env));
        env->rng = i;
        env->num_maps = num_maps;
        env->map_name = strdup(map_file);
        env->human_agent_idx = human_agent_idx;
        env->reward_vehicle_collision = reward_vehicle_collision;
        env->reward_offroad_collision = reward_offroad_collision;
        env->reward_goal_post_respawn = reward_goal_post_respawn;
        env->reward_vehicle_collision_post_respawn = reward_vehicle_collision_post_respawn;
        env->max_agents = env_max_agents[i];
        env->map_set_rng = map_set_rng;
        init(env);
        env->num_agents = env->active_agent_count;
    }

    free(env_map_ids);
    free(env_max_agents);

    printf("Created %d envs, %d total agents (target %d)\n",
           total_envs, total_agents, total_agents);

    *num_envs_out = total_envs;
    return envs;
}

void my_vec_reset(StaticVec* vec) {
    Env* envs = (Env*)vec->envs;

    // Grab shared state before closing envs
    int num_maps = envs[0].num_maps;
    int human_agent_idx = envs[0].human_agent_idx;
    float reward_vehicle_collision = envs[0].reward_vehicle_collision;
    float reward_offroad_collision = envs[0].reward_offroad_collision;
    float reward_goal_post_respawn = envs[0].reward_goal_post_respawn;
    float reward_vehicle_collision_post_respawn = envs[0].reward_vehicle_collision_post_respawn;
    unsigned int map_set_rng = envs[0].map_set_rng + 1;
    int num_buffers = vec->buffers;
    int agents_per_buffer = vec->agents_per_buffer;

    // Close and free each env's internals
    for (int i = 0; i < vec->size; i++) {
        c_close(&envs[i]);
    }
    free(vec->envs);
    vec->envs = NULL;

    // Build new env layout with fresh lazy map sampling
    int max_envs = agents_per_buffer * num_buffers;
    int* env_map_ids = (int*)malloc(max_envs * sizeof(int));
    int* env_max_agents = (int*)malloc(max_envs * sizeof(int));
    int total_envs = 0;

    for (int b = 0; b < num_buffers; b++) {
        vec->buffer_env_starts[b] = total_envs;
        int buffer_agents = 0;
        while (buffer_agents < agents_per_buffer) {
            int cap;
            int map_id = sample_valid_map(&map_set_rng, num_maps, &cap);
            int remaining = agents_per_buffer - buffer_agents;
            if (cap <= remaining) {
                env_map_ids[total_envs] = map_id;
                env_max_agents[total_envs] = cap;
                buffer_agents += cap;
                total_envs++;
            } else {
                while (buffer_agents < agents_per_buffer) {
                    int fcap;
                    int fmap_id = sample_valid_map(&map_set_rng, num_maps, &fcap);
                    env_map_ids[total_envs] = fmap_id;
                    env_max_agents[total_envs] = 1;
                    buffer_agents++;
                    total_envs++;
                }
            }
        }
        vec->buffer_env_counts[b] = total_envs - vec->buffer_env_starts[b];
    }

    // Allocate and initialize new envs
    envs = (Env*)calloc(total_envs, sizeof(Env));
    for (int i = 0; i < total_envs; i++) {
        srand(i);
        char map_file[512];
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", MAP_BINARY_DIR, env_map_ids[i]);
        Env* env = &envs[i];
        memset(env, 0, sizeof(Env));
        env->rng = i;
        env->num_maps = num_maps;
        env->map_name = strdup(map_file);
        env->human_agent_idx = human_agent_idx;
        env->reward_vehicle_collision = reward_vehicle_collision;
        env->reward_offroad_collision = reward_offroad_collision;
        env->reward_goal_post_respawn = reward_goal_post_respawn;
        env->reward_vehicle_collision_post_respawn = reward_vehicle_collision_post_respawn;
        env->map_set_rng = map_set_rng;
        env->max_agents = env_max_agents[i];
        init(env);
        env->num_agents = env->active_agent_count;
    }

    free(env_map_ids);
    free(env_max_agents);

    vec->envs = envs;
    vec->size = total_envs;

    // Re-wire obs/actions/rewards/terminals to existing vec buffers
    size_t obs_elem_size = obs_element_size();
    for (int buf = 0; buf < num_buffers; buf++) {
        int buf_start = buf * agents_per_buffer;
        int buf_agent = 0;
        int env_start = vec->buffer_env_starts[buf];
        int env_count = vec->buffer_env_counts[buf];
        for (int e = 0; e < env_count; e++) {
            Env* env = &envs[env_start + e];
            int slot = buf_start + buf_agent;
            env->observations = (void*)((char*)vec->observations + slot * OBS_SIZE * obs_elem_size);
            env->actions = vec->actions + slot * NUM_ATNS;
            env->rewards = vec->rewards + slot;
            env->terminals = vec->terminals + slot;
            buf_agent += env->num_agents;
        }
    }
}

void my_init(Env* env, Dict* kwargs) {
    env->human_agent_idx = dict_get(kwargs, "human_agent_idx")->value;
    env->reward_vehicle_collision = dict_get(kwargs, "reward_vehicle_collision")->value;
    env->reward_offroad_collision = dict_get(kwargs, "reward_offroad_collision")->value;
    env->reward_goal_post_respawn = dict_get(kwargs, "reward_goal_post_respawn")->value;
    env->reward_vehicle_collision_post_respawn = dict_get(kwargs, "reward_vehicle_collision_post_respawn")->value;
    int map_id = dict_get(kwargs, "map_id")->value;
    int max_agents = dict_get(kwargs, "max_agents")->value;

    char map_file[512];
    snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", MAP_BINARY_DIR, map_id);
    env->num_agents = max_agents;
    env->map_name = strdup(map_file);
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "offroad_rate", log->offroad_rate);
    dict_set(out, "collision_rate", log->collision_rate);
    dict_set(out, "dnf_rate", log->dnf_rate);
    dict_set(out, "n", log->n);
    dict_set(out, "completion_rate", log->completion_rate);
    dict_set(out, "clean_collision_rate", log->clean_collision_rate);
}
