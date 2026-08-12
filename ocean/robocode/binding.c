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

// One packing path for all cases. Stock-equivalent seeding:
//   srand(i); rng = i; my_init; leave rng (mix only *reads* rng as env index).
// Frozen primary carve only when mix_enabled && mix_hist_pct > 0.
Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;

    int mix_on = (int)dict_get_float_default(env_kwargs, "mix_enabled", 0.0f);
    int bot_pct = (int)dict_get_float_default(env_kwargs, "mix_bot_pct", 20.0f);
    int hist_pct = (int)dict_get_float_default(env_kwargs, "mix_hist_pct", 30.0f);
    if (bot_pct < 0) bot_pct = 0;
    if (hist_pct < 0) hist_pct = 0;
    if (bot_pct + hist_pct > 100) hist_pct = 100 - bot_pct;

    int primary_per_buffer = agents_per_buffer;
    if (mix_on && hist_pct > 0) {
        int n_banks = 0;
        float frozen_pct = 0.0f;
        DictItem* nb = dict_get_unsafe(vec_kwargs, "num_frozen_banks");
        if (nb) n_banks = (int)nb->value;
        DictItem* fp = dict_get_unsafe(vec_kwargs, "frozen_bank_pct");
        if (fp) frozen_pct = (float)fp->value;
        int frozen = (int)(agents_per_buffer * frozen_pct);
        if (frozen < 0) frozen = 0;
        if (n_banks > 0 && frozen > 0) {
            primary_per_buffer = agents_per_buffer - frozen * n_banks;
            if (primary_per_buffer < 1) primary_per_buffer = agents_per_buffer;
        }
    }
    int primary_cap = primary_per_buffer * num_buffers;

    Env* envs = (Env*)calloc((size_t)total_agents, sizeof(Env));
    int num_envs = 0;
    int agents_created = 0;
    while (agents_created < primary_cap) {
        int remaining = primary_cap - agents_created;
        unsigned int seed = (unsigned int)num_envs;
        // Last primary slot: force bot composition (id 0) so we take 1 agent
        // instead of a 2-agent SP/hist that would overshoot. Episode seed stays
        // `seed` after my_init (see restore below).
        unsigned int compose = seed;
        if (mix_on && remaining == 1) {
            if (bot_pct <= 0) break;
            compose = 0;
        }

        srand((int)seed);
        envs[num_envs].rng = compose;
        my_init(&envs[num_envs], env_kwargs);
        envs[num_envs].rng = seed;  // stock episode seed (no-op if compose==seed)

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

    for (int i = 0; i < num_buffers; i++) {
        buffer_env_starts[i] = 0;
        buffer_env_counts[i] = 0;
    }
    int buf = 0, buf_agents = 0;
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
    env->bot_cl_noise = dict_get_float_default(kwargs, "bot_cl_noise", 0.0f);
    if (env->bot_cl_noise < 0.0f) env->bot_cl_noise = 0.0f;
    if (env->bot_cl_noise > 1.0f) env->bot_cl_noise = 1.0f;
    env->bot_cl_decay = dict_get_float_default(kwargs, "bot_cl_decay", 0.0f);
    if (env->bot_cl_decay < 0.0f) env->bot_cl_decay = 0.0f;

    // Opponent mix: read-only use of rng as env-index seed (set by my_vec_init).
    // Does not call rand_r — same as prior hardcode:
    //   bot_policy = ((env->rng % 10) == 0) ? 3 : 6;
    if ((int)dict_get_float_default(kwargs, "mix_enabled", 0.0f)) {
        int bot_pct = (int)dict_get_float_default(kwargs, "mix_bot_pct", 20.0f);
        int hist_pct = (int)dict_get_float_default(kwargs, "mix_hist_pct", 30.0f);
        int a_pct = (int)dict_get_float_default(kwargs, "mix_bot_a_pct", 20.0f);
        int pol_a = (int)dict_get_float_default(kwargs, "mix_bot_policy_a", 3.0f);
        int pol_b = (int)dict_get_float_default(kwargs, "mix_bot_policy_b", 6.0f);
        if (bot_pct < 0) bot_pct = 0;
        if (hist_pct < 0) hist_pct = 0;
        if (bot_pct + hist_pct > 100) hist_pct = 100 - bot_pct;
        if (a_pct < 0) a_pct = 0;
        if (a_pct > 100) a_pct = 100;

        unsigned int id = env->rng;
        int r = (int)(id % 100u);
        if (r < bot_pct) {
            env->num_agents = 1;
            env->num_bots = 1;
            // a_pct=10 → (id % 10)==0, matches hardcode above.
            int use_a = 0;
            if (a_pct >= 100) use_a = 1;
            else if (a_pct > 0 && 100 % a_pct == 0)
                use_a = (id % (unsigned int)(100 / a_pct)) == 0u;
            else if (a_pct > 0)
                use_a = (int)(id % 100u) < a_pct;
            env->bot_policy = use_a ? pol_a : pol_b;
        } else if (r < bot_pct + hist_pct) {
            env->num_agents = 2;
            env->num_bots = 0;
            env->bot_cl_noise = 0.0f;
            env->bot_cl_decay = 0.0f;
        } else {
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
