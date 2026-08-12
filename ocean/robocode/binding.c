#include "robocode.h"
#define OBS_SIZE (EGO_FEATURES + OTHER_FEATURES)
#define NUM_ATNS 5
#define ACT_SIZES {4, 9, 11, 11, 6}
#define OBS_TENSOR_T FloatTensor

#define MY_USES_PERM
#define MY_USES_TAGS
#define MY_VEC_INIT
#define Env Robocode
#include "vecenv.h"

// Selfplay-pool routing: write per-slot pointers into the global vec buffers,
// respecting agent_perm if set (selfplay re-routes logical slots into specific
// physical rows so banks own contiguous ranges). Identity perm = adjacent
// slot_base + s layout — matches single-agent / bot-mode runs.
void my_setup_perm(StaticVec* vec, Env* env, int slot_base) {
    for (int s = 0; s < env->num_agents; s++) {
        int phys = vec->agent_perm ? vec->agent_perm[slot_base + s] : (slot_base + s);
        env->obs_ptr[s]      = (float*)vec->observations + (size_t)phys * OBS_SIZE;
        env->action_ptr[s]   = vec->actions + (size_t)phys * NUM_ATNS;
        env->reward_ptr[s]   = vec->rewards + phys;
        env->terminal_ptr[s] = vec->terminals + phys;
    }
}


static inline float dict_get_float_default(Dict* kwargs, const char* key, float default_value) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item ? (float)item->value : default_value;
}

void my_init(Env* env, Dict* kwargs);

// Fill buffer_env_starts / buffer_env_counts for a packed primary width.
static void robocode_fill_buffer_info(Env* envs, int num_envs, int num_buffers,
                                      int primary_per_buffer,
                                      int* buffer_env_starts, int* buffer_env_counts) {
    for (int i = 0; i < num_buffers; i++) {
        buffer_env_starts[i] = 0;
        buffer_env_counts[i] = 0;
    }
    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    for (int i = 0; i < num_envs; i++) {
        buf_agents += envs[i].num_agents;
        buffer_env_counts[buf]++;
        if (buf_agents >= primary_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }
}

// Stock-identical packing (copy of vecenv.h default my_vec_init). Used when
// mix_enabled=0 so pure agent-vs-bot is bit-compatible with pre-mix commits.
static Env* robocode_vec_init_stock(int* num_envs_out, int* buffer_env_starts,
                                    int* buffer_env_counts,
                                    Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;

    Env* envs = (Env*)calloc((size_t)total_agents, sizeof(Env));
    int num_envs = 0;
    int agents_created = 0;
    while (agents_created < total_agents) {
        srand(num_envs);
        envs[num_envs].rng = (unsigned int)num_envs;
        my_init(&envs[num_envs], env_kwargs);
        agents_created += envs[num_envs].num_agents;
        num_envs++;
    }
    envs = (Env*)realloc(envs, (size_t)num_envs * sizeof(Env));
    robocode_fill_buffer_info(envs, num_envs, num_buffers, agents_per_buffer,
                              buffer_env_starts, buffer_env_counts);
    *num_envs_out = num_envs;
    return envs;
}

// Custom env packing for opponent mix: only fill primary agent rows so frozen
// bank slices (end of each buffer) stay free for historical opponents.
// When mix hist demand is 0, do not carve frozen seats (matches Python
// selfplay.setup and pure agent-vs-bot packing = full total_agents).
Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int mix_enabled = (int)dict_get_float_default(env_kwargs, "mix_enabled", 0.0f);
    // Critical: mix off must not take a different packing/RNG path than pre-mix.
    if (!mix_enabled) {
        return robocode_vec_init_stock(num_envs_out, buffer_env_starts,
                                       buffer_env_counts, vec_kwargs, env_kwargs);
    }

    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;

    int mix_bot_pct = (int)dict_get_float_default(env_kwargs, "mix_bot_pct", 20.0f);
    int mix_hist_pct = (int)dict_get_float_default(env_kwargs, "mix_hist_pct", 30.0f);
    if (mix_bot_pct < 0) mix_bot_pct = 0;
    if (mix_hist_pct < 0) mix_hist_pct = 0;
    if (mix_bot_pct + mix_hist_pct > 100) mix_hist_pct = 100 - mix_bot_pct;

    int num_frozen_banks = 0;
    float frozen_bank_pct = 0.0f;
    DictItem* nb = dict_get_unsafe(vec_kwargs, "num_frozen_banks");
    if (nb) num_frozen_banks = (int)nb->value;
    DictItem* fp = dict_get_unsafe(vec_kwargs, "frozen_bank_pct");
    if (fp) frozen_bank_pct = (float)fp->value;

    int frozen_per_bank = (int)(agents_per_buffer * frozen_bank_pct);
    if (frozen_per_bank < 0) frozen_per_bank = 0;
    // Reserve frozen rows only when hist envs are requested.
    int frozen_per_buffer = 0;
    if (mix_hist_pct > 0 && num_frozen_banks > 0 && frozen_per_bank > 0) {
        frozen_per_buffer = frozen_per_bank * num_frozen_banks;
    }
    int primary_per_buffer = agents_per_buffer - frozen_per_buffer;
    if (primary_per_buffer < 1) primary_per_buffer = agents_per_buffer;
    int primary_cap = primary_per_buffer * num_buffers;

    // Bot-only mix (100% bot, no hist): same packing as stock — one agent per
    // env filling total_agents. Only mix_id differs for policy a/b assignment.
    if (mix_bot_pct >= 100 && mix_hist_pct <= 0) {
        Env* envs = (Env*)calloc((size_t)total_agents, sizeof(Env));
        int num_envs = 0;
        int agents_created = 0;
        while (agents_created < total_agents) {
            srand(num_envs);
            envs[num_envs].rng = (unsigned int)num_envs;
            envs[num_envs].mix_id = (unsigned int)num_envs;
            my_init(&envs[num_envs], env_kwargs);
            agents_created += envs[num_envs].num_agents;
            num_envs++;
        }
        envs = (Env*)realloc(envs, (size_t)num_envs * sizeof(Env));
        robocode_fill_buffer_info(envs, num_envs, num_buffers, agents_per_buffer,
                                  buffer_env_starts, buffer_env_counts);
        *num_envs_out = num_envs;
        return envs;
    }

    Env* envs = (Env*)calloc((size_t)total_agents, sizeof(Env));
    int num_envs = 0;
    int agents_created = 0;
    while (agents_created < primary_cap) {
        int remaining = primary_cap - agents_created;
        // Composition id only. Episode RNG seed stays env index (stock parity).
        unsigned int mix_id = (unsigned int)num_envs;
        if (remaining == 1) {
            if (mix_bot_pct > 0) {
                mix_id = 0;  // r=0 → bot bucket
            } else {
                break;
            }
        }
        srand(num_envs);
        envs[num_envs].rng = (unsigned int)num_envs;
        envs[num_envs].mix_id = mix_id;
        my_init(&envs[num_envs], env_kwargs);

        int n_ag = envs[num_envs].num_agents;
        if (n_ag < 1) n_ag = 1;
        if (agents_created + n_ag > primary_cap) {
            free(envs[num_envs].robots);
            free(envs[num_envs].bullets);
            free(envs[num_envs].logs);
            bot_mems_free(&envs[num_envs]);
            memset(&envs[num_envs], 0, sizeof(Env));
            break;
        }
        agents_created += n_ag;
        num_envs++;
    }

    envs = (Env*)realloc(envs, (size_t)num_envs * sizeof(Env));
    robocode_fill_buffer_info(envs, num_envs, num_buffers, primary_per_buffer,
                              buffer_env_starts, buffer_env_counts);
    *num_envs_out = num_envs;
    return envs;
}

void my_init(Env* env, Dict* kwargs) {
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->num_agents = dict_get(kwargs, "num_agents")->value;
    env->num_bots = dict_get(kwargs, "num_bots")->value;
    env->max_ticks = (int)dict_get(kwargs, "max_ticks")->value;
    env->reward_damage = dict_get_float_default(kwargs, "reward_damage", 0.0f);
    env->reward_spot = dict_get_float_default(kwargs, "reward_spot", 0.0f);
    env->reward_melee_damage_inflicted = dict_get_float_default(kwargs, "reward_melee_damage_inflicted", 0.0f);
    env->reward_damage_taken = dict_get_float_default(kwargs, "reward_damage_taken", 0.0f);
    env->reward_range_damage_inflicted = dict_get_float_default(kwargs, "reward_range_damage_inflicted", 0.0f);
    env->reward_melee_damage_inflicted_slot_0 = dict_get_float_default(kwargs,
        "reward_melee_damage_inflicted_slot_0", env->reward_melee_damage_inflicted);
    env->reward_damage_taken_slot_0 = dict_get_float_default(kwargs,
        "reward_damage_taken_slot_0", env->reward_damage_taken);
    env->reward_range_damage_inflicted_slot_0 = dict_get_float_default(kwargs,
        "reward_range_damage_inflicted_slot_0", env->reward_range_damage_inflicted);
    env->reward_melee_damage_inflicted_slot_1 = dict_get_float_default(kwargs,
        "reward_melee_damage_inflicted_slot_1", env->reward_melee_damage_inflicted);
    env->reward_damage_taken_slot_1 = dict_get_float_default(kwargs,
        "reward_damage_taken_slot_1", env->reward_damage_taken);
    env->reward_range_damage_inflicted_slot_1 = dict_get_float_default(kwargs,
        "reward_range_damage_inflicted_slot_1", env->reward_range_damage_inflicted);
    DictItem* dr_item = dict_get_unsafe(kwargs, "dr");
    env->dr = dr_item ? (float)dr_item->value : 0.0f;
    env->bot_policy = dict_get(kwargs, "bot_policy")->value;
    {
        DictItem* bp1 = dict_get_unsafe(kwargs, "bot_policy_1");
        env->bot_policy_1 = bp1 ? (int)bp1->value : -1;
    }
    // Curriculum vs bots: fraction of ticks with random bot actions, decays
    // by bot_cl_decay on each agent win until 0 (full bot). Defaults off.
    env->bot_cl_noise = dict_get_float_default(kwargs, "bot_cl_noise", 0.0f);
    if (env->bot_cl_noise < 0.0f) env->bot_cl_noise = 0.0f;
    if (env->bot_cl_noise > 1.0f) env->bot_cl_noise = 1.0f;
    env->bot_cl_decay = dict_get_float_default(kwargs, "bot_cl_decay", 0.0f);
    if (env->bot_cl_decay < 0.0f) env->bot_cl_decay = 0.0f;

    // Opponent mix (option 2). Composition uses env->mix_id only (set by
    // my_vec_init). env->rng is the episode seed and must not be used for mix.
    env->mix_enabled = (int)dict_get_float_default(kwargs, "mix_enabled", 0.0f);
    env->mix_bot_pct = (int)dict_get_float_default(kwargs, "mix_bot_pct", 20.0f);
    env->mix_hist_pct = (int)dict_get_float_default(kwargs, "mix_hist_pct", 30.0f);
    env->mix_bot_policy_a = (int)dict_get_float_default(kwargs, "mix_bot_policy_a", 3.0f);
    env->mix_bot_policy_b = (int)dict_get_float_default(kwargs, "mix_bot_policy_b", 6.0f);
    env->mix_bot_a_pct = (int)dict_get_float_default(kwargs, "mix_bot_a_pct", 20.0f);
    if (env->mix_bot_pct < 0) env->mix_bot_pct = 0;
    if (env->mix_hist_pct < 0) env->mix_hist_pct = 0;
    if (env->mix_bot_pct + env->mix_hist_pct > 100) {
        env->mix_hist_pct = 100 - env->mix_bot_pct;
    }
    if (env->mix_bot_a_pct < 0) env->mix_bot_a_pct = 0;
    if (env->mix_bot_a_pct > 100) env->mix_bot_a_pct = 100;
    env->mix_mode = ROBOCODE_MIX_OFF;

    if (env->mix_enabled) {
        // Deterministic mix_id hash into BOT / HIST / SP (does not touch rng).
        unsigned int id = env->mix_id;
        int r = (int)(id % 100u);
        if (r < env->mix_bot_pct) {
            env->mix_mode = ROBOCODE_MIX_BOT;
            env->num_agents = 1;
            env->num_bots = 1;
            // Interleaved policy_a/b — must NOT use (id/100)%100, which clusters
            // ~1000 consecutive envs onto policy_a (max_run≈1000) and diverges
            // hard from the original hardcode:
            //   bot_policy = (env_idx % 10 == 0) ? 3 : 6;   // 10% a, evenly
            // When a_pct divides 100: stride = 100/a_pct, a on (id % stride)==0.
            // a_pct=10 → stride 10 → exact match of that hardcode.
            int a_pct = env->mix_bot_a_pct;
            int use_a = 0;
            if (a_pct >= 100) {
                use_a = 1;
            } else if (a_pct > 0) {
                if (100 % a_pct == 0) {
                    unsigned int stride = (unsigned int)(100 / a_pct);
                    use_a = (id % stride) == 0u;
                } else {
                    // Fallback: first a_pct residues of each 100 (max_run=a_pct).
                    use_a = (int)(id % 100u) < a_pct;
                }
            }
            env->bot_policy = use_a
                ? env->mix_bot_policy_a : env->mix_bot_policy_b;
        } else if (r < env->mix_bot_pct + env->mix_hist_pct) {
            env->mix_mode = ROBOCODE_MIX_HIST;
            env->num_agents = 2;
            env->num_bots = 0;
            // CL noise only applies to scripted bots.
            env->bot_cl_noise = 0.0f;
            env->bot_cl_decay = 0.0f;
        } else {
            env->mix_mode = ROBOCODE_MIX_SP;
            env->num_agents = 2;
            env->num_bots = 0;
            env->bot_cl_noise = 0.0f;
            env->bot_cl_decay = 0.0f;
        }
    }

    env->bot_match_winner = -2;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "damage_received", log->damage_received);
    dict_set(out, "melee_damage_inflicted", log->melee_damage_inflicted);
    dict_set(out, "damage_taken", log->damage_taken);
    dict_set(out, "range_damage_inflicted", log->range_damage_inflicted);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    // Historical-pool stats. selfplay.py reads hist_score_bank_<b> /
    // hist_n_bank_<b> per bank to drive swap decisions. Legacy aggregate
    // hist_score / hist_n sum across all banks for backward-compat dashboards.
    dict_set(out, "hist_score", log->hist_score);
    dict_set(out, "hist_n", log->hist_n);
    dict_set(out, "hist_score_bank_0", log->hist_score_bank[0]);
    dict_set(out, "hist_score_bank_1", log->hist_score_bank[1]);
    dict_set(out, "hist_n_bank_0", log->hist_n_bank[0]);
    dict_set(out, "hist_n_bank_1", log->hist_n_bank[1]);
    // Per-slot scores — match() reads slot_0_score / slot_1_score as A/B win rates.
    dict_set(out, "slot_0_score", log->slot_0_score);
    dict_set(out, "slot_1_score", log->slot_1_score);
    dict_set(out, "draw_rate", log->draw_rate);
    dict_set(out, "bot_cl_noise", log->bot_cl_noise);
    // CL-adjusted winrate: E[win_credit * (1 - noise_faced)]. Max 1 only when
    // always winning with bot_cl_noise annealed to 0. Prefer as Protein metric.
    dict_set(out, "cl_perf", log->cl_perf);
    // Opponent-mix WR: use score/n (ratio invariant under aggregate /N).
    dict_set(out, "mix_bot_score", log->mix_bot_score);
    dict_set(out, "mix_bot_n", log->mix_bot_n);
    dict_set(out, "mix_sp_score", log->mix_sp_score);
    dict_set(out, "mix_sp_n", log->mix_sp_n);
    dict_set(out, "mix_hist_score", log->mix_hist_score);
    dict_set(out, "mix_hist_n", log->mix_hist_n);
    dict_set(out, "n", log->n);
}
