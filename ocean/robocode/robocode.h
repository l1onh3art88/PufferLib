#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "raylib.h"

#define NUM_ACTIONS 5
#define NUM_BULLETS 16
#define EGO_FEATURES 14
#define OTHER_FEATURES 8

static const float ACCEL_VALUES[4] = {
    -2.0f, -1.0f, 0, 1.0f
};

static const float TURN_VALUES[9] = {
    -10.0f, -6.0f, -3.0f, -1.0f, 0, 1.0f, 3.0f, 6.0f, 10.0f
};

static const float GUN_TURN_VALUES[11] = {
    -20.0f, -10.0f, -5.0f, -3.0f, -1.0f, 0, 1.0f, 3.0f, 5.0f, 15.0f, 20.0f
};

static const float RADAR_TURN_VALUES[11] = {
    -45.0f, -25.0f, -10.0f, -5.0f,  -1.0f, 0, 1.0f, 5.0f, 10.0f, 25.0f, 45.0f
};
static const float FIREPOWER_VALUES[6] = {
    0, 0.1f, 0.5f, 1.0f, 2.0f, 3.0f
};
float cos_deg(float deg) {
    return cosf(deg * 3.14159265358979323846f / 180.0f);
}

float sin_deg(float deg) {
    return sinf(deg * 3.14159265358979323846f / 180.0f);
}

typedef struct BotMem BotMem;  // defined in bots.h

#define ROBOCODE_MAX_BANKS 2

typedef struct Log Log;
struct Log {
    float perf;             // bots killed this episode
    float episode_return;
    float episode_length;
    float score;            // damage dealt this episode
    float damage_received;  // starting energy - current energy at episode end
    float melee_damage_inflicted;
    float damage_taken;
    float range_damage_inflicted;
    // Historical pool tracking (selfplay-pool mode). Per-bank score/games for
    // matches against frozen historical opponents. hist_score / hist_n are
    // legacy aggregates summed across all banks.
    float hist_score;
    float hist_n;
    float hist_score_bank[ROBOCODE_MAX_BANKS];
    float hist_n_bank[ROBOCODE_MAX_BANKS];
    // Per-slot scores for match() scoring + selfplay sanity-check. In selfplay
    // both should average to ~0.5; in match A=slot 0, B=slot 1, slot_0_score is
    // the win rate of policy A. Each game contributes 1.0 worth of credit total
    // (win=1.0 to winner, draw=0.5 each). We scale by num_agents on accumulation
    // so the eval_log mean (sum / n where n increments by num_agents per ep)
    // equals win_rate directly.
    float slot_0_score;
    float slot_1_score;
    float draw_rate;
    // Curriculum: bot random-action probability faced this episode (pre-decay).
    float bot_cl_noise;
    // Curriculum: hist/frozen opponent random-action probability (pre-decay).
    float hist_cl_noise;
    // CL-adjusted win credit: episode_score * (1 - noise_faced).
    // noise = bot_cl on bot games, hist_cl on hist games, 0 on live SP.
    // After /n this is mean winrate discounted by noise. Max=1 only when
    // always winning against a fully annealed opponent (noise=0). Protein target.
    float cl_perf;
    // Opponent-mix win rates (ratio score/n is invariant under aggregate /N).
    // BOT = agent vs scripted; SP = live selfplay; HIST = vs frozen bank.
    float mix_bot_score;     // raw bot win credit (no noise discount)
    float mix_bot_cl_score;  // bot win * (1 - noise_faced); for mix_bot_wr
    float mix_bot_n;
    float mix_sp_score;
    float mix_sp_n;
    float mix_hist_score;
    float mix_hist_n;
    float n;
};

typedef struct Bullet Bullet;
struct Bullet {
    float x;
    float y;
    float heading;
    float firepower;
    bool live;
};

typedef struct Robot Robot;
struct Robot {
    float x;
    float y;
    float v;
    float heading;
    float gun_heading;
    float radar_heading_prev;
    float radar_heading;
    float speed_mult;
    float handling_mult;
    float power_mult;
    float reward_melee_damage_inflicted;
    float reward_damage_taken;
    float reward_range_damage_inflicted;
    int bullet_idx;
    float gun_heat;
    float energy;
    float start_energy;  // episode start; damage_received uses this (energy_dr)
};

typedef struct Client Client;
typedef struct Robocode Robocode;
struct Robocode {
    Client* client;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    // Per-slot pointers populated by my_setup_perm (MY_USES_PERM). Required for
    // selfplay-pool mode where agent_perm reroutes logical slots into specific
    // physical rows (primary vs frozen-bank). For non-selfplay (num_agents=1)
    // these still point at the env's single slot.
    float* obs_ptr[2];
    float* action_ptr[2];
    float* reward_ptr[2];
    float* terminal_ptr[2];

    int num_agents;
    int num_bots;
    int tick;
    int max_ticks;   // episode timeout; configured per-run via [env].max_ticks
    int width;
    int height;
    Robot* robots;
    Bullet* bullets;
    Log log;
    Log* logs;
    // reward_damage is kept as a legacy config field. The explicit shaped
    // damage coefficients are fixed per env config / sweep trial and copied
    // into each learning slot on reset so policies can condition on them.
    float reward_damage;
    float reward_spot;
    float reward_melee_damage_inflicted;
    float reward_damage_taken;
    float reward_range_damage_inflicted;
    float reward_melee_damage_inflicted_slot_0;
    float reward_damage_taken_slot_0;
    float reward_range_damage_inflicted_slot_0;
    float reward_melee_damage_inflicted_slot_1;
    float reward_damage_taken_slot_1;
    float reward_range_damage_inflicted_slot_1;
    float dr;
    // Episode-level domain randomization (physics/arena), sampled in c_reset.
    // base_width/height are the configured defaults; width/height may change
    // each episode when arena_dr > 0. Obs already normalize by width/height.
    int base_width;
    int base_height;
    float arena_dr;    // multiplicative size jitter around base (0 = fixed)
    float spawn_dr;    // P(structured spawn) vs uniform; headings always random
    float energy_dr;   // start energy jitter around 100; independent per robot
    int bot_policy;          // policy for bot index 0 (and sole bot when num_bots==1)
    int bot_policy_1;        // policy for bot index 1 (bot-vs-bot tournaments)
    int bot_match_winner;    // last pure-bot episode: 0/1 winner, -1 draw, -2 none
    // Curriculum learning vs scripted bots: each bot tick, with probability
    // bot_cl_noise, replace the bot policy with random discrete actions
    // (same action tables as agents). On agent win (outcome +1) noise decays
    // by bot_cl_decay until 0 = full-strength bot. Per-env so vectorized
    // workers diversify difficulty as they win.
    float bot_cl_noise;      // current random-action probability in [0, 1]
    float bot_cl_decay;      // subtract from bot_cl_noise on each agent win
    // Same idea for frozen historical opponents (tag > 0): with probability
    // hist_cl_noise, overwrite slot-1 (frozen) discrete actions with uniform
    // random before they execute. Anneal on primary (slot-0) win.
    float hist_cl_noise;
    float hist_cl_decay;
    BotMem* bot_mems;        // per-bot scratch (allocated by bots.h)

    // Selfplay-pool tagging. tag = 0 means pure selfplay (both slots = primary
    // policy). tag = 1..ROBOCODE_MAX_BANKS means historical: slot 0 = primary,
    // slot 1 = frozen historical opponent from bank (tag - 1). boundary_reached
    // is set on game-end so Python can detect when historical envs have all
    // completed at least one game since the last swap arm.
    // Bot-mix envs keep tag=0 (no frozen opponent).
    // Opponent mix (if enabled) is applied only in my_init from env index seed
    // (rng set to env index before my_init; not advanced by mix).
    int tag;
    int boundary_reached;

    unsigned int rng;
};

static inline void bot_mems_alloc(Robocode* env);
static inline void bot_mems_free(Robocode* env);
static inline void bot_mems_episode_reset(Robocode* env);

void init(Robocode* env){
    int total_robots = env->num_agents + env->num_bots;
    env->robots = (Robot*)calloc(total_robots, sizeof(Robot));
    env->bullets = (Bullet*)calloc(NUM_BULLETS*total_robots, sizeof(Bullet));
    env->logs = (Log*)calloc(env->num_agents, sizeof(Log));
    bot_mems_alloc(env);
}

void allocate_env(Robocode* env) {
    int obs_size = EGO_FEATURES + OTHER_FEATURES;
    init(env);
    env->observations =(float*)calloc(obs_size*env->num_agents, sizeof(float));
    env->actions = (float*)calloc(NUM_ACTIONS*env->num_agents, sizeof(float));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (float*)calloc(env->num_agents, sizeof(float));
    // Standalone (non-vecenv) path: wire per-slot pointers to adjacent rows of
    // the env-owned buffers. vecenv path overrides these via my_setup_perm.
    for (int s = 0; s < env->num_agents; s++) {
        env->obs_ptr[s]      = env->observations + s * obs_size;
        env->action_ptr[s]   = env->actions + s * NUM_ACTIONS;
        env->reward_ptr[s]   = env->rewards + s;
        env->terminal_ptr[s] = env->terminals + s;
    }
}


void c_close(Robocode* env) {
    free(env->robots);
    free(env->bullets);
    free(env->logs);
    bot_mems_free(env);
}

void add_log(Robocode* env) {
    // Called at episode end. Finalize damage_received from current energy,
    // then fold per-agent running logs into the aggregate env->log.
    for (int i = 0; i < env->num_agents; i++) {
        env->logs[i].damage_received =
            env->robots[i].start_energy - (float)env->robots[i].energy;
        env->log.perf            += env->logs[i].perf;
        env->log.episode_return  += env->logs[i].episode_return;
        env->log.episode_length  += env->logs[i].episode_length;
        env->log.score                   += env->logs[i].score;
        env->log.damage_received         += env->logs[i].damage_received;
        env->log.melee_damage_inflicted  += env->logs[i].melee_damage_inflicted;
        env->log.damage_taken            += env->logs[i].damage_taken;
        env->log.range_damage_inflicted  += env->logs[i].range_damage_inflicted;
        env->log.bot_cl_noise            += env->logs[i].bot_cl_noise;
        env->log.hist_cl_noise           += env->logs[i].hist_cl_noise;
        env->log.n                       += 1.0f;
    }
}

bool segment_intersects_aabb(
    float x0, float y0,
    float x1, float y1,
    float left, float right,
    float bottom, float top
) {
    float tmin = 0.0f;
    float tmax = 1.0f;

    float dx = x1 - x0;
    float dy = y1 - y0;

    if (fabsf(dx) < 1e-8f) {
        if (x0 < left || x0 > right) return false;
    } else {
        float tx1 = (left  - x0) / dx;
        float tx2 = (right - x0) / dx;
        if (tx1 > tx2) { float tmp = tx1; tx1 = tx2; tx2 = tmp; }
        if (tx1 > tmin) tmin = tx1;
        if (tx2 < tmax) tmax = tx2;
        if (tmin > tmax) return false;
    }

    if (fabsf(dy) < 1e-8f) {
        if (y0 < bottom || y0 > top) return false;
    } else {
        float ty1 = (bottom - y0) / dy;
        float ty2 = (top    - y0) / dy;
        if (ty1 > ty2) { float tmp = ty1; ty1 = ty2; ty2 = tmp; }
        if (ty1 > tmin) tmin = ty1;
        if (ty2 < tmax) tmax = ty2;
        if (tmin > tmax) return false;
    }

    return true;
}

// Closest-approach distance between two moving point-bullets during t∈[0,1].
// Returns true if the squared min distance falls under r2.
static bool bullets_collide(
    float ax0, float ay0, float ax1, float ay1,
    float bx0, float by0, float bx1, float by1,
    float r2
) {
    float p0x = ax0 - bx0, p0y = ay0 - by0;
    float p1x = ax1 - bx1, p1y = ay1 - by1;
    float dx = p1x - p0x, dy = p1y - p0y;
    float aq = dx*dx + dy*dy;
    float bq = p0x*dx + p0y*dy;
    float t = (aq > 1e-8f) ? -bq/aq : 0.0f;
    if (t < 0.0f) t = 0.0f;
    else if (t > 1.0f) t = 1.0f;
    float mx = p0x + t*dx, my = p0y + t*dy;
    return (mx*mx + my*my) < r2;
}

static inline void add_agent_reward(Robocode* env, int agent_idx, float reward) {
    *env->reward_ptr[agent_idx] += reward;
    env->logs[agent_idx].episode_return += reward;
}

static inline void record_melee_damage_inflicted(Robocode* env, int agent_idx, float damage) {
    if (damage <= 0.0f || agent_idx < 0 || agent_idx >= env->num_agents) return;
    env->logs[agent_idx].melee_damage_inflicted += damage;
    add_agent_reward(env, agent_idx,
        damage * env->robots[agent_idx].reward_melee_damage_inflicted);
}

static inline void record_damage_taken(Robocode* env, int agent_idx, float damage) {
    if (damage <= 0.0f || agent_idx < 0 || agent_idx >= env->num_agents) return;
    env->logs[agent_idx].damage_taken += damage;
    add_agent_reward(env, agent_idx,
        damage * env->robots[agent_idx].reward_damage_taken);
}

static inline void record_range_damage_inflicted(Robocode* env, int agent_idx, float damage) {
    if (damage <= 0.0f || agent_idx < 0 || agent_idx >= env->num_agents) return;
    env->logs[agent_idx].range_damage_inflicted += damage;
    add_agent_reward(env, agent_idx,
        damage * env->robots[agent_idx].reward_range_damage_inflicted);
}

static inline void record_melee_collision(Robocode* env, int a_idx, int b_idx, float damage) {
    record_melee_damage_inflicted(env, a_idx, damage);
    record_melee_damage_inflicted(env, b_idx, damage);
    record_damage_taken(env, a_idx, damage);
    record_damage_taken(env, b_idx, damage);
}

void move(Robocode* env, Robot* robot, float distance) {
    int robot_idx = (int)(robot - env->robots);
    float dx = cos_deg(robot->heading);
    float dy = sin_deg(robot->heading);
    //float accel = 1.0;//2.0*distance / (robot->v * robot->v);
    float accel = distance;
    float handling = fmaxf(robot->handling_mult, 0.0f);
    float max_speed = 8.0f * fmaxf(robot->speed_mult, 0.0f);

    if (accel > handling) {
        accel = handling;
    } else if (accel < -2.0f * handling) {
        accel = -2.0f * handling;
    }

    robot->v += accel;
    if (robot->v > max_speed) {
        robot->v = max_speed;
    } else if (robot->v < -max_speed) {
        robot->v = -max_speed;
    }

    float new_x = robot->x + dx * robot->v;
    float new_y = robot->y + dy * robot->v;

    // Collision check
    int total_robots = env->num_agents + env->num_bots;
    for (int j = 0; j < total_robots; j++) {
        Robot* target = &env->robots[j];
        if (target == robot) {
            continue;
        }
        if (target->energy < 0) {
            continue;
        }
        float abs_x = fabsf(target->x - new_x);
        float abs_y = fabsf(target->y - new_y);
        if(abs_x > 32.0f || abs_y > 32.0f){
            continue;
        }

        float melee_damage = 0.6f;
        record_melee_collision(env, robot_idx, j, melee_damage);
        target->energy -= melee_damage;
        robot->energy -= melee_damage;
        robot->v = 0;
        target->v = 0;   // both robots stop on ramming collision (classic rule)
        return;
    }
    
    robot->x = new_x;
    robot->y = new_y;

}

float turn(float* heading, float degrees, float d_angle, float turn_offset) {
    if (degrees > d_angle) {
        degrees = d_angle;
    } else if (degrees < -d_angle) {
        degrees = -d_angle;
    }

    *heading += (degrees + turn_offset);
    if (*heading >= 360.0f) {
        *heading -= 360.0f;
    } else if (*heading < 0.0f) {
        *heading += 360.0f;
    }
    return degrees;
}

void fire(Robocode* env, Robot* robot, int robot_idx, float firepower) {
    if (robot->gun_heat > 0) {
        return;
    }
    if (robot->energy < firepower) {
        return;
    }
    robot->energy -= firepower;

    Bullet* bullet = &env->bullets[robot_idx*NUM_BULLETS + robot->bullet_idx];
    robot->bullet_idx = (robot->bullet_idx + 1) % NUM_BULLETS;
    robot->gun_heat += 1.0f + firepower/5.0f;

    bullet->x = robot->x + 16*cos_deg(robot->gun_heading);
    bullet->y = robot->y + 16*sin_deg(robot->gun_heading);
    bullet->heading = robot->gun_heading;
    bullet->firepower = firepower;
    bullet->live = true;
}

static inline float rand_unit(Robocode* env) {
    return (float)rand_r(&env->rng) / ((float)RAND_MAX + 1.0f);
}

static inline void sample_dr_triplet(Robocode* env, float* a, float* b, float* c) {
    *a = 1.0f;
    *b = 1.0f;
    *c = 1.0f;

    if (env->dr <= 0.0f) return;
    float upper = 1.0f + env->dr;
    if (upper <= 0.0f) return;
    float lower = 1.0f / upper;
    float width = upper - lower;
    if (width <= 0.0f) return;

    for (int tries = 0; tries < 64; tries++) {
        float first = lower + width * rand_unit(env);
        float second = lower + width * rand_unit(env);
        float third = 3.0f - first - second;
        if (third >= lower && third <= upper) {
            *a = first;
            *b = second;
            *c = third;
            return;
        }
    }
}

static inline void sample_agent_multipliers(Robocode* env, Robot* robot) {
    sample_dr_triplet(env, &robot->speed_mult, &robot->handling_mult, &robot->power_mult);
}

static inline void assign_agent_reward_coefficients(Robocode* env, Robot* robot, int agent_idx) {
    if (agent_idx == 0) {
        robot->reward_melee_damage_inflicted = env->reward_melee_damage_inflicted_slot_0;
        robot->reward_damage_taken = env->reward_damage_taken_slot_0;
        robot->reward_range_damage_inflicted = env->reward_range_damage_inflicted_slot_0;
    } else if (agent_idx == 1) {
        robot->reward_melee_damage_inflicted = env->reward_melee_damage_inflicted_slot_1;
        robot->reward_damage_taken = env->reward_damage_taken_slot_1;
        robot->reward_range_damage_inflicted = env->reward_range_damage_inflicted_slot_1;
    } else {
        robot->reward_melee_damage_inflicted = env->reward_melee_damage_inflicted;
        robot->reward_damage_taken = env->reward_damage_taken;
        robot->reward_range_damage_inflicted = env->reward_range_damage_inflicted;
    }
}

int scan_area(Robocode* env, Robot* robot){
    // Sweep is the signed angle traversed from radar_heading_prev to
    // radar_heading, normalized to (-180, 180]. A robot is scanned if its
    // bearing from us lies inside this wedge and within 1200 units.
    float start = robot->radar_heading_prev;
    float sweep = robot->radar_heading - start;
    if (sweep > 180.0f) sweep -= 360.0f;
    if (sweep < -180.0f) sweep += 360.0f;

    int total_robots = env->num_agents + env->num_bots;
    for (int j = 0; j < total_robots; j++) {
        Robot* other = &env->robots[j];
        if (other == robot) continue;
        if (other->energy < 0) continue;

        float dx = other->x - robot->x;
        float dy = other->y - robot->y;
        if (dx*dx + dy*dy > 1200.0f*1200.0f) continue;

        float bearing = atan2f(dy, dx) * 180.0f / 3.14159265358979323846f;
        if (bearing < 0.0f) bearing += 360.0f;

        float diff = bearing - start;
        if (diff > 180.0f) diff -= 360.0f;
        if (diff < -180.0f) diff += 360.0f;

        // diff is inside the wedge when it shares sign with sweep and is shorter.
        if (diff * sweep >= 0.0f && fabsf(diff) <= fabsf(sweep)) return j;
    }
    return -1;
}

void compute_observations(Robocode* env){
    for(int i = 0; i < env->num_agents; i++){
        Robot* robot = &env->robots[i];
        float* obs = env->obs_ptr[i];
        obs[0] = robot->x / env->width;
        obs[1] = robot->y / env->height;
        // Absolute headings stored as degrees in [0, 360); convert to radians [0, 2π).
        obs[2] = robot->heading * DEG2RAD;
        obs[3] = robot->gun_heading * DEG2RAD;
        obs[4] = robot->radar_heading * DEG2RAD;
        obs[5] = robot->radar_heading_prev * DEG2RAD;
        obs[6] = robot->v / 8.0f;
        obs[7] = robot->energy / 100.0f;
        obs[8] = robot->speed_mult;
        obs[9] = robot->power_mult;
        obs[10] = robot->handling_mult;
        obs[11] = robot->reward_melee_damage_inflicted;
        obs[12] = robot->reward_damage_taken;
        obs[13] = robot->reward_range_damage_inflicted;

        int scanned = scan_area(env, robot);
        if (scanned < 0) {
            memset(&obs[EGO_FEATURES], 0, OTHER_FEATURES * sizeof(float));
            continue;
        }
        *env->reward_ptr[i] += env->reward_spot;
        env->logs[i].episode_return += env->reward_spot;
        // Zero-sum: penalize the scanned agent (being seen = bad).
        // Guarded so bots (j >= num_agents) don't trigger an OOB write.
        if (scanned < env->num_agents) {
            *env->reward_ptr[scanned] -= env->reward_spot;
            env->logs[scanned].episode_return -= env->reward_spot;
        }
        Robot* other = &env->robots[scanned];
        // Relative position rotated into ego (body) frame. Engine convention:
        // cos_deg -> x, sin_deg -> y, so forward = (cos h, sin h), right = (sin h, -cos h).
        float dx_w = other->x - robot->x;
        float dy_w = other->y - robot->y;
        float c = cos_deg(robot->heading);
        float s = sin_deg(robot->heading);
        float dx_ego =  c*dx_w + s*dy_w;   // forward
        float dy_ego = -s*dx_w + c*dy_w;   // perpendicular (matches drive's R(-h))
        // Relative headings: wrap raw delta (in (-360, 360)) to (-180, 180]
        // then scale to [-1, 1].
        float dh_body  = other->heading - robot->heading;
        float dh_gun   = other->heading - robot->gun_heading;
        float dh_radar = other->heading - robot->radar_heading;
        if (dh_body  >  180.0f) dh_body  -= 360.0f; else if (dh_body  < -180.0f) dh_body  += 360.0f;
        if (dh_gun   >  180.0f) dh_gun   -= 360.0f; else if (dh_gun   < -180.0f) dh_gun   += 360.0f;
        if (dh_radar >  180.0f) dh_radar -= 360.0f; else if (dh_radar < -180.0f) dh_radar += 360.0f;
        // Aim error: bearing to target (world) minus my gun heading, wrapped to (-180, 180].
        // ~0 means gun is pointed at the target.
        float bearing = atan2f(dy_w, dx_w) * 180.0f / 3.14159265358979323846f;
        float aim_err = bearing - robot->gun_heading;
        if (aim_err >  180.0f) aim_err -= 360.0f;
        else if (aim_err < -180.0f) aim_err += 360.0f;
        int off = EGO_FEATURES;
        obs[off + 0] = dx_ego / 1200.0f;
        obs[off + 1] = dy_ego / 1200.0f;
        obs[off + 2] = dh_body  * DEG2RAD;
        obs[off + 3] = dh_gun   * DEG2RAD;
        obs[off + 4] = dh_radar * DEG2RAD;
        obs[off + 5] = other->energy / 100.0f;
        obs[off + 6] = aim_err * DEG2RAD;
        obs[off + 7] = 1.0f;
    }
}
static inline float sample_mult_range(Robocode* env, float dr) {
    if (dr <= 0.0f) return 1.0f;
    float upper = 1.0f + dr;
    float lower = 1.0f / upper;
    return lower + (upper - lower) * rand_unit(env);
}

static inline void sample_episode_arena(Robocode* env) {
    int bw = env->base_width > 0 ? env->base_width : env->width;
    int bh = env->base_height > 0 ? env->base_height : env->height;
    if (bw < 64) bw = 64;
    if (bh < 64) bh = 64;
    if (env->arena_dr <= 0.0f) {
        env->width = bw;
        env->height = bh;
        return;
    }
    // Independent multiplicative jitter → aspect ratio varies too.
    float sw = sample_mult_range(env, env->arena_dr);
    float sh = sample_mult_range(env, env->arena_dr);
    int w = (int)lroundf((float)bw * sw);
    int h = (int)lroundf((float)bh * sh);
    if (w < 400) w = 400;
    if (h < 400) h = 400;
    if (w > 1600) w = 1600;
    if (h > 1600) h = 1600;
    env->width = w;
    env->height = h;
}

// Fill xs[0..n) ys[0..n) with spawn positions. Returns 1 on success.
static inline int sample_spawn_positions(Robocode* env, int n, float* xs, float* ys) {
    const float margin = 48.0f;
    float w = (float)env->width;
    float h = (float)env->height;
    if (w < 2.0f * margin + 64.0f || h < 2.0f * margin + 64.0f) return 0;

    int structured = (n == 2 && env->spawn_dr > 0.0f && rand_unit(env) < env->spawn_dr);
    if (!structured) {
        for (int i = 0; i < n; i++) {
            int tries = 0;
            for (;;) {
                float x = margin + rand_unit(env) * (w - 2.0f * margin);
                float y = margin + rand_unit(env) * (h - 2.0f * margin);
                int ok = 1;
                for (int j = 0; j < i; j++) {
                    if (fabsf(x - xs[j]) <= 32.0f && fabsf(y - ys[j]) <= 32.0f) {
                        ok = 0;
                        break;
                    }
                }
                if (ok || ++tries > 64) {
                    xs[i] = x;
                    ys[i] = y;
                    break;
                }
            }
        }
        return 1;
    }

    // Structured 1v1 layouts — forces distance / angle variety beyond uniform.
    int mode = (int)(rand_r(&env->rng) % 4u);
    float flip = rand_unit(env) < 0.5f ? 1.0f : -1.0f;
    if (mode == 0) {
        // Opposite corners.
        xs[0] = margin;           ys[0] = margin;
        xs[1] = w - margin;       ys[1] = h - margin;
        if (flip < 0.0f) { xs[0] = w - margin; ys[0] = margin; xs[1] = margin; ys[1] = h - margin; }
    } else if (mode == 1) {
        // Opposite mid-walls (east-west duel).
        xs[0] = margin;           ys[0] = h * 0.5f;
        xs[1] = w - margin;       ys[1] = h * 0.5f;
        if (flip < 0.0f) { float t = xs[0]; xs[0] = xs[1]; xs[1] = t; }
    } else if (mode == 2) {
        // Opposite mid-walls (north-south duel).
        xs[0] = w * 0.5f;         ys[0] = margin;
        xs[1] = w * 0.5f;         ys[1] = h - margin;
        if (flip < 0.0f) { float t = ys[0]; ys[0] = ys[1]; ys[1] = t; }
    } else {
        // Close-range knife fight near a random point.
        float cx = margin + 0.25f * (w - 2.0f * margin) + 0.5f * (w - 2.0f * margin) * rand_unit(env);
        float cy = margin + 0.25f * (h - 2.0f * margin) + 0.5f * (h - 2.0f * margin) * rand_unit(env);
        float ang = rand_unit(env) * 6.2831853f;
        float sep = 40.0f + 80.0f * rand_unit(env);
        xs[0] = cx + sep * cosf(ang);
        ys[0] = cy + sep * sinf(ang);
        xs[1] = cx - sep * cosf(ang);
        ys[1] = cy - sep * sinf(ang);
        for (int i = 0; i < 2; i++) {
            if (xs[i] < margin) xs[i] = margin;
            if (xs[i] > w - margin) xs[i] = w - margin;
            if (ys[i] < margin) ys[i] = margin;
            if (ys[i] > h - margin) ys[i] = h - margin;
        }
    }
    return 1;
}

void c_reset(Robocode* env) {
    env->tick = 0;
    // boundary_reached is owned by selfplay.py alignment; do not clear it here.
    int total_robots = env->num_agents + env->num_bots;
    memset(env->bullets, 0, NUM_BULLETS * total_robots * sizeof(Bullet));

    sample_episode_arena(env);

    float xs[16];
    float ys[16];
    if (total_robots > 16) total_robots = 16;
    if (!sample_spawn_positions(env, total_robots, xs, ys)) {
        // Fallback: center line.
        for (int i = 0; i < total_robots; i++) {
            xs[i] = (float)env->width * (0.25f + 0.5f * (float)i / (float)total_robots);
            ys[i] = (float)env->height * 0.5f;
        }
    }

    for (int idx = 0; idx < total_robots; idx++) {
        Robot* robot = &env->robots[idx];
        robot->x = xs[idx];
        robot->y = ys[idx];
        robot->v = 0;
        // Random pose when spawn_dr>0 (train). Eval forces spawn_dr=0 → heading 0
        // for a fixed testbed. Previously headings were always 0.
        float hdg = (env->spawn_dr > 0.0f) ? (rand_unit(env) * 360.0f) : 0.0f;
        robot->heading = hdg;
        robot->gun_heading = hdg;
        robot->radar_heading = hdg;
        robot->radar_heading_prev = hdg;
        float em = sample_mult_range(env, env->energy_dr);
        float energy = 100.0f * em;
        if (energy < 30.0f) energy = 30.0f;
        if (energy > 200.0f) energy = 200.0f;
        robot->energy = energy;
        robot->start_energy = energy;
        // Occasional warm gun so first-shot timing isn't identical every ep.
        robot->gun_heat = (env->energy_dr > 0.0f && rand_unit(env) < 0.35f)
            ? (3.0f * rand_unit(env)) : 3.0f;
        robot->bullet_idx = 0;
        if (idx < env->num_agents) {
            sample_agent_multipliers(env, robot);
            assign_agent_reward_coefficients(env, robot, idx);
            env->logs[idx] = (Log){0};
        } else {
            robot->speed_mult = 1.0f;
            robot->handling_mult = 1.0f;
            robot->power_mult = 1.0f;
            robot->reward_melee_damage_inflicted = 0.0f;
            robot->reward_damage_taken = 0.0f;
            robot->reward_range_damage_inflicted = 0.0f;
        }
    }
    bot_mems_episode_reset(env);
    compute_observations(env);
}

#include "bots.h"

// Agent matches end as soon as a slot reaches zero energy. If all slots are
// disabled in the same tick, score the episode as a draw. Return 2 = no end.
static inline int agent_terminal_outcome(Robocode* env) {
    if (env->num_agents <= 0) return 2;
    bool slot0_dead = env->robots[0].energy <= 0.0f;
    if (env->num_agents == 1) return slot0_dead ? -1 : 2;

    bool any_nonzero_dead = false;
    bool any_nonzero_alive = false;
    for (int a = 1; a < env->num_agents; a++) {
        if (env->robots[a].energy <= 0.0f) any_nonzero_dead = true;
        else any_nonzero_alive = true;
    }

    if (slot0_dead && !any_nonzero_alive) return 0;
    if (slot0_dead) return -1;
    if (any_nonzero_dead) return +1;
    return 2;
}

// Helper for every episode-end path. outcome: +1 slot-0 won, -1 slot-0 lost,
// 0 draw. Historical accounting only applies when env->tag > 0.
static inline void end_episode(Robocode* env, int outcome) {
    float s0_score = (outcome > 0) ? 1.0f : (outcome < 0) ? 0.0f : 0.5f;
    // Noise faced this episode (pre-decay). Bot vs scripted, hist vs frozen, else 0.
    float noise = 0.0f;
    if (env->num_bots > 0 && env->num_agents == 1)
        noise = env->bot_cl_noise;
    else if (env->tag > 0 && env->tag <= ROBOCODE_MAX_BANKS)
        noise = env->hist_cl_noise;
    if (noise < 0.0f) noise = 0.0f;
    if (noise > 1.0f) noise = 1.0f;
    // CL-adjusted win credit: full credit only for wins at noise=0.
    float cl_credit = s0_score * (1.0f - noise);

    // Scale by num_agents so that (slot_0_score / n) where n increments by
    // num_agents per episode in add_log gives the win rate directly. match()
    // reads this from env/slot_0_score after eval_log divides by n.
    env->log.slot_0_score += s0_score * env->num_agents;
    env->log.slot_1_score += (1.0f - s0_score) * env->num_agents;
    env->log.cl_perf += cl_credit * env->num_agents;
    if (outcome == 0) env->log.draw_rate += env->num_agents;
    // Per-opponent-mode WR (layout is fixed at my_init; no mix_mode field).
    if (env->num_bots > 0 && env->num_agents == 1) {
        env->log.mix_bot_score += s0_score;
        env->log.mix_bot_cl_score += cl_credit;
        env->log.mix_bot_n += 1.0f;
    } else if (env->tag > 0 && env->tag <= ROBOCODE_MAX_BANKS) {
        env->log.mix_hist_score += s0_score;
        env->log.mix_hist_n += 1.0f;
    } else if (env->num_agents >= 2 && env->num_bots == 0) {
        env->log.mix_sp_score += s0_score;
        env->log.mix_sp_n += 1.0f;
    }
    if (env->tag > 0 && env->tag <= ROBOCODE_MAX_BANKS) {
        int bank_idx = env->tag - 1;
        env->log.hist_score_bank[bank_idx] += s0_score;
        env->log.hist_n_bank[bank_idx]     += 1.0f;
        env->log.hist_score                += s0_score;
        env->log.hist_n                    += 1.0f;
        env->boundary_reached = 1;
    }
    // Snapshot pre-decay noise for metrics (what the agent actually faced).
    for (int a = 0; a < env->num_agents; a++) {
        env->logs[a].bot_cl_noise = (env->num_bots > 0 && env->num_agents == 1)
            ? noise : 0.0f;
        env->logs[a].hist_cl_noise = (env->tag > 0 && env->tag <= ROBOCODE_MAX_BANKS)
            ? noise : 0.0f;
    }
    // Curriculum: primary win → harden bot / frozen opp (lower random rate).
    if (outcome > 0 && env->num_bots > 0 && env->bot_cl_decay > 0.0f
            && env->bot_cl_noise > 0.0f) {
        env->bot_cl_noise -= env->bot_cl_decay;
        if (env->bot_cl_noise < 0.0f) env->bot_cl_noise = 0.0f;
    }
    if (outcome > 0 && env->tag > 0 && env->tag <= ROBOCODE_MAX_BANKS
            && env->hist_cl_decay > 0.0f && env->hist_cl_noise > 0.0f) {
        env->hist_cl_noise -= env->hist_cl_decay;
        if (env->hist_cl_noise < 0.0f) env->hist_cl_noise = 0.0f;
    }
    for (int a = 0; a < env->num_agents; a++) *env->terminal_ptr[a] = 1.0f;
    add_log(env);
    c_reset(env);
}

void c_step(Robocode* env) {
    // Timeout: all agents step in lockstep, so logs[0].episode_length is shared.
    env->tick += 1;
    if (env->tick > env->max_ticks) {
        if (env->num_agents == 0) {
            env->bot_match_winner = -1;  // pure bot-vs-bot draw
            c_reset(env);
        } else {
            end_episode(env, 0);  // draw
        }
        return;
    }

    int total_robots = env->num_agents + env->num_bots;
    int total_bullets = total_robots * NUM_BULLETS;

    // Reset per-agent reward/terminal and short-circuit reset on agent death.
    for (int a = 0; a < env->num_agents; a++) {
        *env->reward_ptr[a]   = 0.0f;
        *env->terminal_ptr[a] = 0.0f;
    }
    int agent_outcome = agent_terminal_outcome(env);
    if (agent_outcome != 2) {
        end_episode(env, agent_outcome);
        return;
    }
    // move all bullets
    float prev_x[total_bullets], prev_y[total_bullets];
    for (int i = 0; i < total_bullets; i++) {
        Bullet* b = &env->bullets[i];
        if (!b->live) continue;
        prev_x[i] = b->x; prev_y[i] = b->y;
        float v = 20.0f - 3.0f * b->firepower;
        b->x += v * cos_deg(b->heading);
        b->y += v * sin_deg(b->heading);
        if (b->x < 0 || b->x > env->width || b->y < 0 || b->y > env->height) {
            b->live = false;
        }
    }

    // bullet-bullet collisions. 
    for (int i = 0; i < total_bullets; i++) {
        if (!env->bullets[i].live) continue;
        for (int j = i + 1; j < total_bullets; j++) {
            if (!env->bullets[j].live) continue;
            if (bullets_collide(prev_x[i], prev_y[i], env->bullets[i].x, env->bullets[i].y,
                                prev_x[j], prev_y[j], env->bullets[j].x, env->bullets[j].y,
                                64.0f)) {
                env->bullets[i].live = false;
                env->bullets[j].live = false;
                break;
            }
        }
    }

    // bullet-robot collisions + rewards.
    bool any_bot_alive = (env->num_bots == 0);
    for (int shooter = 0; shooter < total_robots; shooter++) {
        Robot* robot = &env->robots[shooter];
        if (shooter >= env->num_agents && robot->energy > 0.0f) any_bot_alive = true;
        for (int blt = 0; blt < NUM_BULLETS; blt++) {
            int bi = shooter * NUM_BULLETS + blt;
            Bullet* bullet = &env->bullets[bi];
            if (!bullet->live) continue;

            for (int j = 0; j < total_robots; j++) {
                if (!bullet->live) break;  
                if (j == shooter) continue;
                Robot* target = &env->robots[j];
                if (target->energy < 0) continue;
                // Broad-phase: keep if EITHER endpoint of the swept segment is
                // within 32 units of target center. Using only the end position
                // would miss tunneling at firepower 0.1 (speed ~19.7/tick).
                float dx0 = target->x - prev_x[bi],   dy0 = target->y - prev_y[bi];
                float dx1 = target->x - bullet->x,   dy1 = target->y - bullet->y;
                float d2  = fminf(dx0*dx0 + dy0*dy0, dx1*dx1 + dy1*dy1);
                if (d2 > 1024.0f) continue;
                bool hit = segment_intersects_aabb(
                        prev_x[bi], prev_y[bi],
                        bullet->x, bullet->y,
                        target->x - 16, target->x + 16,
                        target->y - 16, target->y + 16
                );
                if (!hit) continue;

                float damage = 4 * bullet->firepower;
                if (bullet->firepower > 1.0f) damage += 2*(bullet->firepower - 1.0f);

                target->energy -= damage;
                robot->energy += 3 * bullet->firepower;
                bullet->live = false;

                bool s_agent = shooter < env->num_agents;
                bool t_agent = j < env->num_agents;
                // onHitByBullet for scripted bot targets (agent→bot or bot→bot).
                if (!t_agent) {
                    bot_on_hit_by_bullet(env, j, bullet->heading, bullet->firepower);
                }
                // DrussGT gun learning when a bot's bullet hits anyone.
                if (!s_agent && env->bot_mems != NULL) {
                    int bmem = shooter - env->num_agents;
                    if (bmem >= 0 && bmem < env->num_bots
                            && bot_policy_for(env, shooter) == BOT_DRUSSGT) {
                        dgt_on_bullet_hit(&env->bot_mems[bmem], target->x, target->y);
                    }
                }
                bool killed = target->energy <= 0.0f;
                if (s_agent) {
                    record_range_damage_inflicted(env, shooter, damage);
                    env->logs[shooter].score += damage;
                    if (killed) add_agent_reward(env, shooter, 1.0f);
                    if (killed && !t_agent) env->logs[shooter].perf += 1.0f;
                }
                if (t_agent) {
                    record_damage_taken(env, j, damage);
                    if (killed) add_agent_reward(env, j, -1.0f);
                }
            }
        }
    }
    if (env->num_bots > 0 && !any_bot_alive) {
        end_episode(env, +1);  // primary wiped all bots
        return;
    }
    agent_outcome = agent_terminal_outcome(env);
    if (agent_outcome != 2) {
        end_episode(env, agent_outcome);
        return;
    }
    for (int i = 0; i < env->num_agents; i++) {
        Robot* robot = &env->robots[i];
        float* atn = env->action_ptr[i];
        env->logs[i].episode_length += 1.0f;

        // Defensive guard; agent_terminal_outcome should have already ended
        // episodes for disabled agents.
        if (robot->energy <= 0.0f) {
            robot->v = 0;
            continue;
        }

        // Hist curriculum: slot 1 is frozen opponent when tag > 0. With prob
        // hist_cl_noise, replace its discrete action heads with uniform random
        // (same tables as bot_cl_noise). Does not touch GPU logprobs — frozen
        // rows have zero advantages so training is unaffected.
        if (i == 1 && env->tag > 0 && env->tag <= ROBOCODE_MAX_BANKS
                && env->hist_cl_noise > 0.0f
                && rand_unit(env) < env->hist_cl_noise) {
            atn[0] = (float)(rand_r(&env->rng) % 4);
            atn[1] = (float)(rand_r(&env->rng) % 9);
            atn[2] = (float)(rand_r(&env->rng) % 11);
            atn[3] = (float)(rand_r(&env->rng) % 11);
            atn[4] = (float)(rand_r(&env->rng) % 6);
        }

        // Cool down gun
        if (robot->gun_heat > 0) {
            robot->gun_heat -= 0.1f;
        }

        // Move
        float handling = fmaxf(robot->handling_mult, 0.0f);
        float move_atn = ACCEL_VALUES[(int)atn[0]] * handling;
        move(env, robot, move_atn);

        // Turn
        float turn_atn = TURN_VALUES[(int)atn[1]] * handling;

        float abs_v = fabs(robot->v);
        float max_turn = (10 - 0.75*abs_v) * handling;
        if (max_turn < 0.0f) max_turn = 0.0f;
        float body_turn_degrees = turn(&robot->heading, turn_atn, max_turn, 0);

        // Gun
        float gun_atn = GUN_TURN_VALUES[(int)atn[2]] * handling;
        float gun_degrees = turn(&robot->gun_heading, gun_atn, 20.0f * handling, body_turn_degrees);

        // Radar
        float radar_atn = RADAR_TURN_VALUES[(int)atn[3]] * handling;
        robot->radar_heading_prev = robot->radar_heading;
        turn(&robot->radar_heading, radar_atn, 45.0f * handling, body_turn_degrees+gun_degrees);

        // Fire
        float firepower = FIREPOWER_VALUES[(int)atn[4]] * fmaxf(robot->power_mult, 0.0f);
        if (firepower > 0) {
            fire(env, robot,i, firepower);
        }

        // Clip position
        float px = robot->x, py = robot->y;
        robot->x = fmaxf(16.0f, fminf(robot->x, env->width  - 16.0f));
        robot->y = fmaxf(16.0f, fminf(robot->y, env->height - 16.0f));
        int hit_wall = (robot->x != px) || (robot->y != py);
        
        // Damage from wall collisions & stop bot
        if(!hit_wall) continue;
        float wall_dmg = fabsf(robot->v)*0.5 - 1;
        if (wall_dmg < 0.0f){
            wall_dmg = 0.0f;
        }
        robot->energy -= wall_dmg;
        record_damage_taken(env, i, wall_dmg);
        robot->v = 0;
    }
    agent_outcome = agent_terminal_outcome(env);
    if (agent_outcome != 2) {
        end_episode(env, agent_outcome);
        return;
    }

    // bot step
    for (int b = env->num_agents; b < total_robots; b++) bot_step(env, b);
    if (env->num_bots > 0) {
        any_bot_alive = false;
        for (int b = env->num_agents; b < total_robots; b++) {
            if (env->robots[b].energy > 0.0f) {
                any_bot_alive = true;
                break;
            }
        }
        if (!any_bot_alive) {
            // All bots dead: agent wipe (if agents present) or bot-vs-bot draw.
            if (env->num_agents == 0) {
                env->bot_match_winner = -1;
                c_reset(env);
            } else {
                end_episode(env, +1);
            }
            return;
        }
        // Pure bot-vs-bot: end when only one bot remains.
        if (env->num_agents == 0 && env->num_bots >= 2) {
            int alive = 0;
            int winner = -1;
            for (int b = 0; b < env->num_bots; b++) {
                if (env->robots[b].energy > 0.0f) {
                    alive++;
                    winner = b;
                }
            }
            if (alive <= 1) {
                env->bot_match_winner = (alive == 1) ? winner : -1;
                c_reset(env);
                return;
            }
        }
    }
    compute_observations(env);
}

typedef struct Client Client;
struct Client {
    Texture2D atlas;
};

Client* make_client(Robocode* env) {
    // Window sized for the largest arena arena_dr can produce (see sample_episode_arena).
    int bw = env->base_width > 0 ? env->base_width : env->width;
    int bh = env->base_height > 0 ? env->base_height : env->height;
    float adr = env->arena_dr > 0.0f ? env->arena_dr : 0.0f;
    int max_w = (int)lroundf((float)bw * (1.0f + adr));
    int max_h = (int)lroundf((float)bh * (1.0f + adr));
    if (max_w < bw) max_w = bw;
    if (max_h < bh) max_h = bh;
    if (max_w > 1600) max_w = 1600;
    if (max_h > 1600) max_h = 1600;
    InitWindow(max_w + 80, max_h + 80, "PufferLib Ray Robocode");
    SetTargetFPS(60);
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->atlas = LoadTexture("resources/robocode/robocode.png");
    return client;
}

void close_client(Client* client) {
    UnloadTexture(client->atlas);
    CloseWindow();
}

void c_render(Robocode* env) {
    if(env->client == NULL){
        env->client = make_client(env);
    }
    Client* client = env->client;
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    // Fit the *current* episode arena (post arena_dr) into the window.
    float pad = 24.0f;
    float view_w = fmaxf(1.0f, (float)GetScreenWidth() - 2.0f * pad);
    float view_h = fmaxf(1.0f, (float)GetScreenHeight() - 2.0f * pad);
    float zoom_x = view_w / fmaxf(1.0f, (float)env->width);
    float zoom_y = view_h / fmaxf(1.0f, (float)env->height);
    float zoom = fminf(zoom_x, zoom_y);
    if (zoom <= 0.0f) zoom = 1.0f;

    Camera2D camera = {0};
    camera.target = (Vector2){env->width * 0.5f, env->height * 0.5f};
    camera.offset = (Vector2){GetScreenWidth() * 0.5f, GetScreenHeight() * 0.5f};
    camera.zoom = zoom;
    BeginMode2D(camera);

    // Floor tiles across the full current map.
    for (int x = 0; x < env->width; x+=64) {
        for (int y = 0; y < env->height; y+=64) {
            int src_x = 64 * ((x*33409 + y*30971) % 5);
            int w = (x + 64 > env->width)  ? env->width  - x : 64;
            int h = (y + 64 > env->height) ? env->height - y : 64;
            Rectangle src_rect = (Rectangle){src_x, 0, w, h};
            Rectangle dst_rect = (Rectangle){x, y, w, h};
            DrawTexturePro(client->atlas, src_rect, dst_rect, (Vector2){0, 0}, 0, WHITE);
        }
    }
    // Arena border so resized maps are obvious.
    DrawRectangleLinesEx((Rectangle){0, 0, (float)env->width, (float)env->height},
        2.0f / zoom, (Color){0, 187, 187, 255});

    int total_robots = env->num_agents + env->num_bots;
    for (int i = 0; i < total_robots; i++) {
        Robot robot = env->robots[i];
        if (robot.energy < 0) continue;
        bool is_agent = i < env->num_agents;
        Vector2 robot_pos = (Vector2){robot.x, robot.y};

        // Radar wedge: agents = green, bots = orange. Use the actual radar
        // sweep sign so vertex winding stays consistent (no backface culling).
        float sweep = robot.radar_heading - robot.radar_heading_prev;
        if (sweep >  180.0f) sweep -= 360.0f;
        else if (sweep < -180.0f) sweep += 360.0f;
        float a_left  = (sweep >= 0) ? robot.radar_heading      : robot.radar_heading_prev;
        float a_right = (sweep >= 0) ? robot.radar_heading_prev : robot.radar_heading;
        Vector2 p_left  = (Vector2){robot.x + 1200*cos_deg(a_left),  robot.y + 1200*sin_deg(a_left)};
        Vector2 p_right = (Vector2){robot.x + 1200*cos_deg(a_right), robot.y + 1200*sin_deg(a_right)};
        Color wedge_color = is_agent ? (Color){0, 255, 0, 128} : (Color){255, 140, 0, 128};
        DrawTriangle(robot_pos, p_left, p_right, wedge_color);

        int src_y = is_agent ? 64 : 128;  // blue row for agents, red row for bots
        Rectangle body_rect  = (Rectangle){0,   src_y, 64, 64};
        Rectangle radar_rect = (Rectangle){64,  src_y, 64, 64};
        Rectangle gun_rect   = (Rectangle){128, src_y, 64, 64};
        Rectangle dest_rect  = (Rectangle){robot.x, robot.y, 64, 64};
        Vector2 origin = (Vector2){32, 32};
        DrawTexturePro(client->atlas, body_rect,  dest_rect, origin, robot.heading+90,       WHITE);
        DrawTexturePro(client->atlas, radar_rect, dest_rect, origin, robot.radar_heading+90, WHITE);
        DrawTexturePro(client->atlas, gun_rect,   dest_rect, origin, robot.gun_heading+90,   WHITE);

        DrawText(TextFormat("%.1f", robot.energy), robot.x-16, robot.y-48, 12, WHITE);
    }

    for (int i = 0; i < (env->num_agents + env->num_bots)*NUM_BULLETS; i++) {
        Bullet bullet = env->bullets[i];
        if (!bullet.live) {
            continue;
        }
        Vector2 bullet_pos = (Vector2){bullet.x, bullet.y};
        DrawCircleV(bullet_pos, 4, WHITE);
    }

    EndMode2D();
    // Screen-space HUD (not affected by arena zoom).
    DrawText(TextFormat("tick=%i  arena=%ix%i  zoom=%.2f",
        env->tick, env->width, env->height, zoom), 10, 10, 16, WHITE);
    EndDrawing();
}
