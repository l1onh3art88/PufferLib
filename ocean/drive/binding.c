#include <sys/file.h>
#include "drive.h"
#define NUM_ATNS 2
#define ACT_SIZES {7, 13}
#define OBS_TENSOR_T PrecisionTensor

#define MAP_BINARY_DIR "drive_data/binaries"

#define MY_VEC_INIT
#define MY_VEC_RESET
#define MY_VEC_CLOSE
#define Env Drive
#include "vecenv.h"

// Scan all maps in parallel; returns count of valid maps.
// Caller must free *ids_out and *caps_out.
static int prescan_valid_maps(int num_maps, int** ids_out, int** caps_out) {
    char cache_file[512];
    char lock_file[512];
    snprintf(cache_file, sizeof(cache_file), "%s/valid_map_cache.bin", MAP_BINARY_DIR);
    snprintf(lock_file,  sizeof(lock_file),  "%s/valid_map_cache.lock", MAP_BINARY_DIR);

    // Serialize across processes: only one scans; the rest wait and read the cache.
    int lock_fd = open(lock_file, O_CREAT | O_RDWR, 0666);
    flock(lock_fd, LOCK_EX);

    // Try cache (may have been written by whoever held the lock before us).
    FILE* cf = fopen(cache_file, "rb");
    if (cf) {
        int cached_num_maps, valid;
        if (fread(&cached_num_maps, sizeof(int), 1, cf) == 1 &&
            fread(&valid, sizeof(int), 1, cf) == 1 &&
            cached_num_maps == num_maps && valid > 0) {
            int* ids = (int*)malloc(valid * sizeof(int));
            int* caps = (int*)malloc(valid * sizeof(int));
            if (fread(ids, sizeof(int), valid, cf) == (size_t)valid &&
                fread(caps, sizeof(int), valid, cf) == (size_t)valid) {
                fclose(cf);
                flock(lock_fd, LOCK_UN);
                close(lock_fd);
                printf("Loaded %d valid maps from cache\n", valid);
                *ids_out = ids;
                *caps_out = caps;
                return valid;
            }
            free(ids);
            free(caps);
        }
        fclose(cf);
    }

    // We hold the lock and no valid cache exists — do the scan.
    int* map_caps = (int*)calloc(num_maps, sizeof(int));
    #pragma omp parallel for schedule(dynamic, 4)
    for (int m = 0; m < num_maps; m++) {
        char map_file[512];
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", MAP_BINARY_DIR, m);
        Env temp_env = {0};
        temp_env.map_name = strdup(map_file);
        init(&temp_env);
        int count = temp_env.active_agent_count < MAX_AGENTS
                  ? temp_env.active_agent_count : MAX_AGENTS;
        c_close(&temp_env);
        map_caps[m] = count;
    }
    int* ids = (int*)malloc(num_maps * sizeof(int));
    int* caps = (int*)malloc(num_maps * sizeof(int));
    int valid = 0;
    for (int m = 0; m < num_maps; m++) {
        if (map_caps[m] > 0) {
            ids[valid] = m;
            caps[valid] = map_caps[m];
            valid++;
        }
    }
    free(map_caps);

    cf = fopen(cache_file, "wb");
    if (cf) {
        fwrite(&num_maps, sizeof(int), 1, cf);
        fwrite(&valid, sizeof(int), 1, cf);
        fwrite(ids, sizeof(int), valid, cf);
        fwrite(caps, sizeof(int), valid, cf);
        fclose(cf);
        printf("Wrote valid map cache (%d valid / %d total)\n", valid, num_maps);
    }

    flock(lock_fd, LOCK_UN);
    close(lock_fd);

    *ids_out = ids;
    *caps_out = caps;
    return valid;
}

// O(1) sample from pre-built valid map list.
static int sample_from_valid(unsigned int* rng, int* valid_ids, int* valid_caps,
                              int valid_count, int* cap_out) {
    int idx = rand_r(rng) % valid_count;
    *cap_out = valid_caps[idx];
    return valid_ids[idx];
}

void my_vec_close(Env* envs) {
    free(envs[0].valid_map_ids);
    free(envs[0].valid_map_caps);
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

    // Pre-scan all maps in parallel once; pointer lives for the lifetime of the vec.
    int* valid_map_ids;
    int* valid_map_caps;
    printf("Scanning %d maps...\n", num_maps);
    int valid_count = prescan_valid_maps(num_maps, &valid_map_ids, &valid_map_caps);
    if (valid_count == 0) {
        printf("ERROR: No valid maps found in %s/\n", MAP_BINARY_DIR);
        *num_envs_out = 0;
        return NULL;
    }
    printf("Found %d valid maps out of %d total\n", valid_count, num_maps);
    // valid_map_ids / valid_map_caps are NOT freed here; stored on every env so
    // my_vec_reset can reuse them without rescanning.

    // Build per-env layout; each sample is now an O(1) array lookup.
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
            int map_id = sample_from_valid(&map_set_rng, valid_map_ids, valid_map_caps, valid_count, &cap);
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
                    int fmap_id = sample_from_valid(&map_set_rng, valid_map_ids, valid_map_caps, valid_count, &fcap);
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

    // Initialize all envs; each env holds the shared valid-map pointers.
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
        env->valid_map_count = valid_count;
        env->valid_map_ids = valid_map_ids;
        env->valid_map_caps = valid_map_caps;
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

    // Grab shared state before closing envs (valid map list survives; just borrow pointer).
    int num_maps = envs[0].num_maps;
    int human_agent_idx = envs[0].human_agent_idx;
    float reward_vehicle_collision = envs[0].reward_vehicle_collision;
    float reward_offroad_collision = envs[0].reward_offroad_collision;
    float reward_goal_post_respawn = envs[0].reward_goal_post_respawn;
    float reward_vehicle_collision_post_respawn = envs[0].reward_vehicle_collision_post_respawn;
    unsigned int map_set_rng = envs[0].map_set_rng + 1;
    int valid_count = envs[0].valid_map_count;
    int* valid_map_ids = envs[0].valid_map_ids;
    int* valid_map_caps = envs[0].valid_map_caps;
    int num_buffers = vec->buffers;
    int agents_per_buffer = vec->agents_per_buffer;

    // Close and free each env's internals
    for (int i = 0; i < vec->size; i++) {
        c_close(&envs[i]);
    }
    free(vec->envs);
    vec->envs = NULL;

    // Build new env layout with O(1) map sampling (valid map list reused from init).
    int max_envs = agents_per_buffer * num_buffers;
    int* env_map_ids = (int*)malloc(max_envs * sizeof(int));
    int* env_max_agents = (int*)malloc(max_envs * sizeof(int));
    int total_envs = 0;

    for (int b = 0; b < num_buffers; b++) {
        vec->buffer_env_starts[b] = total_envs;
        int buffer_agents = 0;
        while (buffer_agents < agents_per_buffer) {
            int cap;
            int map_id = sample_from_valid(&map_set_rng, valid_map_ids, valid_map_caps, valid_count, &cap);
            int remaining = agents_per_buffer - buffer_agents;
            if (cap <= remaining) {
                env_map_ids[total_envs] = map_id;
                env_max_agents[total_envs] = cap;
                buffer_agents += cap;
                total_envs++;
            } else {
                while (buffer_agents < agents_per_buffer) {
                    int fcap;
                    int fmap_id = sample_from_valid(&map_set_rng, valid_map_ids, valid_map_caps, valid_count, &fcap);
                    env_map_ids[total_envs] = fmap_id;
                    env_max_agents[total_envs] = 1;
                    buffer_agents++;
                    total_envs++;
                }
            }
        }
        vec->buffer_env_counts[b] = total_envs - vec->buffer_env_starts[b];
    }

    // Allocate and initialize new envs; re-attach shared valid map pointers.
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
        env->valid_map_count = valid_count;
        env->valid_map_ids = valid_map_ids;
        env->valid_map_caps = valid_map_caps;
        init(env);
        env->skip_next_log = 1;
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
