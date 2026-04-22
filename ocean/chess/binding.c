#include "chess.h"
#define OBS_SIZE 1082
#define NUM_ATNS 1
#define ACT_SIZES {97}
#define OBS_TENSOR_T ByteTensor
#define MY_ACTION_MASK 97

#define MY_VEC_INIT
#define MY_VEC_CLOSE
#define Env Chess
#include "vecenv.h"

#define DEFAULT_STARTING_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#define FEN_CURRICULUM_PATH "resources/chess/fens.txt"

static char** SHARED_FEN_CURRICULUM = NULL;
static int SHARED_NUM_FENS = 0;

static char** load_fen_file(const char* path, int* num_fens_out) {
    FILE* f = fopen(path, "r");
    if (f == NULL) {
        *num_fens_out = 0;
        return NULL;
    }

    int num_fens = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '#' && line[0] != '\n' && line[0] != '\r') {
            num_fens++;
        }
    }
    if (num_fens == 0) {
        fclose(f);
        *num_fens_out = 0;
        return NULL;
    }

    char** fens = (char**)malloc(num_fens * sizeof(char*));
    rewind(f);
    int idx = 0;
    while (fgets(line, sizeof(line), f) && idx < num_fens) {
        if (line[0] != '#' && line[0] != '\n' && line[0] != '\r') {
            size_t len = strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
                line[--len] = '\0';
            }
            fens[idx++] = strdup(line);
        }
    }
    fclose(f);
    *num_fens_out = num_fens;
    return fens;
}

static void apply_kwargs(Env* env, Dict* kwargs) {
    env->max_moves = (int)dict_get(kwargs, "max_moves")->value;
    env->reward_draw = (float)dict_get(kwargs, "reward_draw")->value;
    env->reward_invalid_piece = (float)dict_get(kwargs, "reward_invalid_piece")->value;
    env->reward_invalid_move = (float)dict_get(kwargs, "reward_invalid_move")->value;
    env->reward_repetition = (float)dict_get(kwargs, "reward_repetition")->value;
    env->render_fps = (int)dict_get(kwargs, "render_fps")->value;
    env->mode = (int)dict_get(kwargs, "mode")->value;
    env->enable_50_move_rule = (int)dict_get(kwargs, "enable_50_move_rule")->value;
    env->enable_threefold_repetition = (int)dict_get(kwargs, "enable_threefold_repetition")->value;
    env->random_fen = (int)dict_get(kwargs, "random_fen")->value;
    env->fen_curric_pct = (float)dict_get(kwargs, "fen_curric_pct")->value;

    env->client = NULL;
    env->legal_dirty = 1;
    env->human_color = -1;
    env->log_pgn = 0;
    env->log_pgn_choice_made = 1;
    env->pgn_filename[0] = '\0';
    env->pgn_game_number = 0;
    strcpy(env->starting_fen, DEFAULT_STARTING_FEN);
    strcpy(env->last_result, "Game starting...");
}

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;

    float curric_pct = (float)dict_get(env_kwargs, "fen_curric_pct")->value;
    if (curric_pct > 0.0f && SHARED_FEN_CURRICULUM == NULL) {
        SHARED_FEN_CURRICULUM = load_fen_file(FEN_CURRICULUM_PATH, &SHARED_NUM_FENS);
        if (SHARED_FEN_CURRICULUM != NULL) {
            printf("Loaded %d FENs from %s\n", SHARED_NUM_FENS, FEN_CURRICULUM_PATH);
        }
    }

    int mode = (int)dict_get(env_kwargs, "mode")->value;
    int agents_per_env = (mode == CHESS_MODE_SELFPLAY) ? 2 : 1;
    int num_envs = total_agents / agents_per_env;
    Env* envs = (Env*)calloc(num_envs, sizeof(Env));
    for (int i = 0; i < num_envs; i++) {
        Env* env = &envs[i];
        apply_kwargs(env, env_kwargs);
        env->num_agents = agents_per_env;
        env->rng = i;
        // In selfplay, slot 0 is always WHITE and slot 1 is always BLACK; learner_color is unused.
        env->learner_color = (agents_per_env == 1) ? (i % 2) : CHESS_WHITE;
        env->fen_curriculum = SHARED_FEN_CURRICULUM;
        env->num_fens = SHARED_NUM_FENS;
        init_bitboards();
    }

    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;
    for (int i = 0; i < num_envs; i++) {
        buf_agents += agents_per_env;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }

    *num_envs_out = num_envs;
    return envs;
}

void my_vec_close(Env* envs) {
    if (SHARED_FEN_CURRICULUM != NULL) {
        for (int i = 0; i < SHARED_NUM_FENS; i++) {
            free(SHARED_FEN_CURRICULUM[i]);
        }
        free(SHARED_FEN_CURRICULUM);
        SHARED_FEN_CURRICULUM = NULL;
        SHARED_NUM_FENS = 0;
    }
}

void my_init(Env* env, Dict* kwargs) {
    apply_kwargs(env, kwargs);
    env->num_agents = (env->mode == CHESS_MODE_SELFPLAY) ? 2 : 1;
    env->learner_color = (env->num_agents == 1) ? CHESS_WHITE : CHESS_WHITE;
    env->fen_curriculum = NULL;
    env->num_fens = 0;
    init_bitboards();
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "draw_rate", log->draw_rate);
    dict_set(out, "timeout_rate", log->timeout_rate);
    dict_set(out, "chess_moves", log->chess_moves);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "invalid_action_rate", log->invalid_action_rate);
}
