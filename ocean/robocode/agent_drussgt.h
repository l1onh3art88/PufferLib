// Faithful structural adaptation of DrussGT 3.1.4159 (jk.mega.DrussGT)
// by Julian Kent / Skilgannon — open source (source included in the jar).
//
// Jar: http://robocode-archive.strangeautomata.com/robots/jk.mega.DrussGT_3.1.4159.jar
// Docs: https://robowiki.net/wiki/DrussGT/Understanding_DrussGT
//
// Not a line-by-line JVM port of the ~50k codesize mega-bot (100+ VCS buffers,
// full precise prediction, bullet shadows, shielding). Ported into our
// robocode.h tick model like Raiko / HawkOnFire / wave-surfer:
//
//   * Go-to wave surfing with kNN guess-factor danger (Manhattan-ish features)
//   * Gunheat-filtered energy-drop wave detection
//   * No-wave fallback: wall-smoothed lateral orbit choosing farther side
//   * DC-style GF gun: kNN kernel density over logged enemy hit GFs
//   * Fire-power schedule similar to DrussGT (base 1.95 / 2.95, energy clamps)
//
// License note (author terms): keep open-source; credit Skilgannon / DrussGT.

#ifndef ROBOCODE_AGENT_DRUSSGT_H
#define ROBOCODE_AGENT_DRUSSGT_H

#include "agent_common.h"

// DGT_WAVES / DGTWave / DGT_KNN_CAP / DGT_GUN_CAP / DGT_FEATS defined in bots.h.
#ifndef DGT_KNN_K
#define DGT_KNN_K 8
#endif
#ifndef DGT_GUN_K
#define DGT_GUN_K 12
#endif
#ifndef DGT_GOTO_CANDS
#define DGT_GOTO_CANDS 24
#endif

// Feature scales (DrussGT-style attributes, compressed).
static const float DGT_FEAT_W[DGT_FEATS] = {
    0.01f,  // distance
    1.00f,  // |lat vel| / 8
    0.50f,  // adv vel / 8
    0.04f,  // time since dir change
    0.50f,  // accel bucket-ish (normalized)
    2.00f,  // wall proximity
};

static inline void dgt_features(Robot* bot, float tgt_x, float tgt_y, Robocode* env,
                                int tick, int last_dir_change, float last_v,
                                float out[DGT_FEATS]) {
    float dx = bot->x - tgt_x, dy = bot->y - tgt_y;
    float dist = sqrtf(dx * dx + dy * dy);
    float ux = (dist > 1e-6f) ? dx / dist : 1.0f;
    float uy = (dist > 1e-6f) ? dy / dist : 0.0f;
    float bvx = cos_deg(bot->heading) * bot->v;
    float bvy = sin_deg(bot->heading) * bot->v;
    float adv_v = bvx * ux + bvy * uy;
    float lat_v = -bvx * uy + bvy * ux;
    float wall_min = fminf(fminf(bot->x, env->width - bot->x),
                           fminf(bot->y, env->height - bot->y));
    float wall_half = fmaxf(fminf(env->width, env->height) * 0.5f, 1.0f);
    float acc = bot->v - last_v;
    out[0] = dist;
    out[1] = fabsf(lat_v) / 8.0f;
    out[2] = adv_v / 8.0f;
    out[3] = (float)(tick - last_dir_change);
    out[4] = rb_clampf(acc / 2.0f, -1.0f, 1.0f);
    out[5] = wall_min / wall_half;
}

static inline float dgt_feat_dist2(const float* a, const float* b) {
    float d2 = 0.0f;
    for (int f = 0; f < DGT_FEATS; f++) {
        float d = (a[f] - b[f]) * DGT_FEAT_W[f];
        d2 += d * d;
    }
    return d2;
}

static float dgt_knn_danger(const float feats[DGT_FEATS], float gf,
                            const float knn_feats[][DGT_FEATS],
                            const float* knn_gf, int knn_n) {
    if (knn_n <= 0) return 0.0f;
    float best_d[DGT_KNN_K];
    int best_i[DGT_KNN_K];
    for (int k = 0; k < DGT_KNN_K; k++) {
        best_d[k] = 1e18f;
        best_i[k] = -1;
    }
    for (int n = 0; n < knn_n; n++) {
        float d2 = dgt_feat_dist2(feats, knn_feats[n]);
        for (int k = 0; k < DGT_KNN_K; k++) {
            if (d2 < best_d[k]) {
                for (int s = DGT_KNN_K - 1; s > k; s--) {
                    best_d[s] = best_d[s - 1];
                    best_i[s] = best_i[s - 1];
                }
                best_d[k] = d2;
                best_i[k] = n;
                break;
            }
        }
    }
    const float sigma = 0.18f;
    const float two_s2 = 2.0f * sigma * sigma;
    float danger = 0.0f;
    for (int k = 0; k < DGT_KNN_K; k++) {
        if (best_i[k] < 0) break;
        float w = 1.0f / (1.0f + best_d[k]);
        float dgf = gf - knn_gf[best_i[k]];
        danger += w * expf(-(dgf * dgf) / two_s2);
    }
    return danger;
}

static inline void dgt_knn_add(float knn_feats[][DGT_FEATS], float* knn_gf,
                               int* knn_n, int* knn_head, int cap,
                               const float feats[DGT_FEATS], float gf) {
    int slot = *knn_head;
    memcpy(knn_feats[slot], feats, DGT_FEATS * sizeof(float));
    knn_gf[slot] = gf;
    *knn_head = (slot + 1) % cap;
    if (*knn_n < cap) (*knn_n)++;
}

// Highest-density GF for the gun (kernel density over KNN neighbors).
static float dgt_gun_best_gf(const float feats[DGT_FEATS],
                             const float knn_feats[][DGT_FEATS],
                             const float* knn_gf, int knn_n) {
    if (knn_n <= 0) return 0.0f;
    float best_d[DGT_GUN_K];
    int best_i[DGT_GUN_K];
    for (int k = 0; k < DGT_GUN_K; k++) {
        best_d[k] = 1e18f;
        best_i[k] = -1;
    }
    for (int n = 0; n < knn_n; n++) {
        float d2 = dgt_feat_dist2(feats, knn_feats[n]);
        for (int k = 0; k < DGT_GUN_K; k++) {
            if (d2 < best_d[k]) {
                for (int s = DGT_GUN_K - 1; s > k; s--) {
                    best_d[s] = best_d[s - 1];
                    best_i[s] = best_i[s - 1];
                }
                best_d[k] = d2;
                best_i[k] = n;
                break;
            }
        }
    }
    // Evaluate density on a coarse GF grid in [-1, 1].
    float best_gf = 0.0f;
    float best_dens = -1.0f;
    const float sigma = 0.12f;
    const float two_s2 = 2.0f * sigma * sigma;
    for (int g = 0; g <= 20; g++) {
        float gf = -1.0f + 0.1f * (float)g;
        float dens = 0.0f;
        for (int k = 0; k < DGT_GUN_K; k++) {
            if (best_i[k] < 0) break;
            float w = 1.0f / (1.0f + best_d[k]);
            float dgf = gf - knn_gf[best_i[k]];
            dens += w * expf(-(dgf * dgf) / two_s2);
        }
        if (dens > best_dens) {
            best_dens = dens;
            best_gf = gf;
        }
    }
    return best_gf;
}

// DrussGT bullet-power schedule (compressed): base 1.95/2.95, energy clamps,
// snap toward classic "x.x5" values used by many rumble bots.
static inline float dgt_bullet_power(Robot* bot, float enemy_energy, float dist,
                                     int bullets_hit, int bullets_passed) {
    double base = 1.95;
    if (bullets_passed > 0 && bullets_hit * 3 > bullets_passed) base = 2.95;
    if (dist < 150.0f) base = 2.95;
    float energy_bp = fmaxf(0.0f, ((float)bot->energy - 20.0f) / 25.0f);
    float enemy_bp = enemy_energy / 4.0f;
    float power = (float)base;
    power = fminf(power, fmaxf(energy_bp, 0.15f));
    power = fminf(power, fmaxf(enemy_bp, 0.15f));
    power = fminf(power, (float)bot->energy / 4.0f);
    power = rb_clampf(power, 0.1f, 3.0f);
    static const float snaps[] = {
        0.15f, 0.25f, 0.35f, 0.45f, 0.65f, 0.85f, 0.95f,
        1.15f, 1.95f, 2.95f
    };
    float best = snaps[0];
    float best_d = fabsf(power - best);
    for (int i = 1; i < (int)(sizeof(snaps) / sizeof(snaps[0])); i++) {
        float d = fabsf(power - snaps[i]);
        if (d < best_d) {
            best_d = d;
            best = snaps[i];
        }
    }
    return rb_clampf(best, 0.1f, 3.0f);
}

// No-wave fallback: classic wall-smoothed lateral orbit; pick the side that
// increases distance from the enemy (DrussGT fallback / Raiko-like).
static void dgt_fallback_move(Robocode* env, Robot* bot, BotMem* m) {
    float abs_bearing = rb_abs_bearing_deg(bot->x, bot->y, m->last_x, m->last_y);
    float distance = rb_dist(bot->x, bot->y, m->last_x, m->last_y);
    float best_x = bot->x, best_y = bot->y;
    float best_score = -1e18f;
    float dirs[2] = {1.0f, -1.0f};
    for (int di = 0; di < 2; di++) {
        float dir = dirs[di];
        for (int t = 0; t < 8; t++) {
            float ang = (abs_bearing + dir * (70.0f + 8.0f * (float)t)) * RB_D2R;
            float x = bot->x + 120.0f * cosf(ang);
            float y = bot->y + 120.0f * sinf(ang);
            if (x < 28.0f || x > env->width - 28.0f ||
                    y < 28.0f || y > env->height - 28.0f) {
                continue;
            }
            float d = rb_dist(x, y, m->last_x, m->last_y);
            // Prefer distancing near ~400 like DrussGT bestDistance.
            float score = -fabsf(d - 400.0f) + 0.15f * d;
            if (distance < 200.0f) score += 0.5f * d;  // escape rammers
            if (score > best_score) {
                best_score = score;
                best_x = x;
                best_y = y;
            }
        }
    }
    rb_drive_to(env, bot, best_x, best_y);
}

// Go-to wave surfing: sample reachable lateral destinations and pick min danger.
static void dgt_surf_goto(Robocode* env, Robot* bot, BotMem* m,
                          DGTWave* waves, int num_waves,
                          const float knn_feats[][DGT_FEATS],
                          const float* knn_gf, int knn_n) {
    float abs_bearing = rb_abs_bearing_deg(bot->x, bot->y, m->last_x, m->last_y);
    float distance = fmaxf(rb_dist(bot->x, bot->y, m->last_x, m->last_y), 1.0f);
    float feats[DGT_FEATS];
    dgt_features(bot, m->last_x, m->last_y, env, m->tick, m->last_dir_change_tick,
                 m->dgt_last_v, feats);

    float best_x = bot->x;
    float best_y = bot->y;
    float best_danger = 1e18f;
    int any_wave = 0;
    for (int wi = 0; wi < num_waves; wi++) {
        if (waves[wi].active) {
            any_wave = 1;
            break;
        }
    }
    if (!any_wave || knn_n == 0) {
        dgt_fallback_move(env, bot, m);
        return;
    }

    for (int c = 0; c < DGT_GOTO_CANDS; c++) {
        // Mix orbit GFs: symmetric samples around current bearing.
        float t = (float)c / (float)(DGT_GOTO_CANDS - 1);  // 0..1
        float gf = -1.0f + 2.0f * t;
        float mea = asinf(fminf(8.0f / 14.0f, 1.0f)) * RB_R2D;  // ~MEA at mid power
        float orbit = abs_bearing + 90.0f * (gf >= 0.0f ? 1.0f : -1.0f)
                    + gf * mea * 0.85f;
        float radius = rb_clampf(distance * 0.35f + 80.0f, 60.0f, 180.0f);
        float x = bot->x + radius * cos_deg(orbit);
        float y = bot->y + radius * sin_deg(orbit);
        if (x < 28.0f || x > env->width - 28.0f ||
                y < 28.0f || y > env->height - 28.0f) {
            continue;
        }

        float danger = 0.0f;
        for (int wi = 0; wi < num_waves; wi++) {
            DGTWave* w = &waves[wi];
            if (!w->active) continue;
            float ddx = x - w->ox, ddy = y - w->oy;
            float dist_hit = fmaxf(sqrtf(ddx * ddx + ddy * ddy), 1.0f);
            float radius_now = (m->tick - w->fire_tick) * w->speed;
            // Approximate time-to-intersect if we head toward (x,y).
            float travel = rb_dist(bot->x, bot->y, x, y);
            float eta = travel / 8.0f;
            float wave_r = radius_now + w->speed * eta;
            if (wave_r < dist_hit - 40.0f) {
                // Wave still outside — light future danger
                float tti = (dist_hit - wave_r) / w->speed;
                if (tti > 40.0f) continue;
            }
            float bearing = rb_abs_bearing_deg(w->ox, w->oy, x, y);
            float gf_raw = rb_norm_deg(bearing - w->head_on);
            float mea_rad = asinf(fminf(8.0f / w->speed, 1.0f));
            float mea_deg = fmaxf(mea_rad * RB_R2D, 0.1f);
            float egf = (gf_raw / mea_deg) * (float)w->lat_sign;
            egf = rb_clampf(egf, -1.5f, 1.5f);
            float dmg = 4.0f * ((20.0f - w->speed) / 3.0f);  // approx power damage
            danger += (1.0f + 0.15f * dmg) * dgt_knn_danger(feats, egf, knn_feats, knn_gf, knn_n);
            // Prefer not closing too hard under fire.
            danger += 0.002f * fmaxf(0.0f, 350.0f - dist_hit);
        }
        if (danger < best_danger) {
            best_danger = danger;
            best_x = x;
            best_y = y;
        }
    }
    m->dgt_goto_x = best_x;
    m->dgt_goto_y = best_y;
    rb_drive_to(env, bot, best_x, best_y);
}

static void bot_drussgt_step(Robocode* env, int bot_idx, BotMem* m) {
    Robot* bot = &env->robots[bot_idx];
    int target_idx = rb_nearest_agent(env, bot);
    if (target_idx < 0) return;
    if (!rb_scan_target(env, bot, m, target_idx)) return;

    if (!m->dgt_initialized) {
        m->dgt_initialized = 1;
        m->dgt_enemy_energy = (float)m->last_energy_seen;
        m->dgt_enemy_firepower = 2.0f;
        m->dgt_enemy_gunheat = bot->gun_heat;
        m->dgt_lat_dir = 1.0f;
        m->dgt_last_v = bot->v;
        m->dgt_goto_x = bot->x;
        m->dgt_goto_y = bot->y;
        m->dgt_surf_n = 0;
        m->dgt_surf_head = 0;
        m->dgt_gun_n = 0;
        m->dgt_gun_head = 0;
        m->dgt_bullets_hit = 0;
        m->dgt_bullets_passed = 0;
        for (int i = 0; i < DGT_WAVES; i++) m->dgt_waves[i].active = 0;
    }

    // Cool enemy gunheat estimate (DrussGT movement gunheat tracking).
    m->dgt_enemy_gunheat = fmaxf(0.0f, m->dgt_enemy_gunheat - 0.1f);

    float drop = m->dgt_enemy_energy - (float)m->last_energy_seen;
    // Only treat energy drops as shots when gunheat allows (filters wall hits).
    if (drop >= 0.1f && drop <= 3.0f && m->dgt_enemy_gunheat <= 0.05f) {
        m->dgt_enemy_firepower = drop;
        m->dgt_enemy_gunheat = 1.0f + drop / 5.0f;
        DGTWave* w = &m->dgt_waves[m->dgt_wave_head];
        m->dgt_wave_head = (m->dgt_wave_head + 1) % DGT_WAVES;
        w->ox = m->last_x;
        w->oy = m->last_y;
        w->speed = rb_bullet_speed(drop);
        w->fire_tick = m->tick;
        w->head_on = rb_abs_bearing_deg(w->ox, w->oy, bot->x, bot->y);
        dgt_features(bot, w->ox, w->oy, env, m->tick, m->last_dir_change_tick,
                     m->dgt_last_v, w->feats);
        float bvx = cos_deg(bot->heading) * bot->v;
        float bvy = sin_deg(bot->heading) * bot->v;
        float dx = bot->x - w->ox, dy = bot->y - w->oy;
        float dist = fmaxf(sqrtf(dx * dx + dy * dy), 1.0f);
        float lat = (-bvx * dy + bvy * dx) / dist;
        w->lat_sign = (lat >= 0.0f) ? 1 : -1;
        w->active = 1;
    }
    m->dgt_enemy_energy = (float)m->last_energy_seen;

    // Lateral direction bookkeeping (for gun features).
    float abs_bearing = rb_abs_bearing_deg(bot->x, bot->y, m->last_x, m->last_y);
    float bvx = cos_deg(bot->heading) * bot->v;
    float bvy = sin_deg(bot->heading) * bot->v;
    float dxe = m->last_x - bot->x, dye = m->last_y - bot->y;
    float dist = fmaxf(sqrtf(dxe * dxe + dye * dye), 1.0f);
    float lat_v = (-bvx * dye + bvy * dxe) / dist;
    if (fabsf(lat_v) > 0.1f) {
        float new_dir = lat_v > 0.0f ? 1.0f : -1.0f;
        if (new_dir != m->dgt_lat_dir) {
            m->dgt_lat_dir = new_dir;
            m->last_dir_change_tick = m->tick;
        }
    }

    // Expire waves that have passed.
    for (int wi = 0; wi < DGT_WAVES; wi++) {
        DGTWave* w = &m->dgt_waves[wi];
        if (!w->active) continue;
        float radius = (m->tick - w->fire_tick) * w->speed;
        float ddx = bot->x - w->ox, ddy = bot->y - w->oy;
        float dist_now = sqrtf(ddx * ddx + ddy * ddy);
        if (radius >= dist_now + 36.0f || m->tick - w->fire_tick > 400) {
            w->active = 0;
        }
    }

    dgt_surf_goto(env, bot, m, m->dgt_waves, DGT_WAVES,
                  m->dgt_surf_feats, m->dgt_surf_gf, m->dgt_surf_n);

    // ---- Gun (DC GF / kNN density) ----
    float gfeats[DGT_FEATS];
    // Features from *enemy* frame for targeting (mirror of surf features).
    {
        Robot fake = *bot;
        // Treat enemy as "bot" for feature extraction relative to us.
        fake.x = m->last_x;
        fake.y = m->last_y;
        fake.heading = m->last_heading;
        fake.v = m->last_v;
        dgt_features(&fake, bot->x, bot->y, env, m->tick, m->last_dir_change_tick,
                     m->dgt_last_v, gfeats);
    }
    float power = dgt_bullet_power(bot, (float)m->last_energy_seen, dist,
                                   m->dgt_bullets_hit, m->dgt_bullets_passed);
    float bspeed = rb_bullet_speed(power);
    float max_escape = asinf(fminf(8.0f / bspeed, 1.0f)) * RB_R2D;
    float gf = dgt_gun_best_gf(gfeats, m->dgt_gun_feats, m->dgt_gun_gf, m->dgt_gun_n);
    float aim = abs_bearing + m->dgt_lat_dir * gf * max_escape;
    // Blend a little linear lead when gun is cold (early fight).
    if (m->dgt_gun_n < 8) {
        aim = 0.65f * aim + 0.35f * rb_linear_aim_deg(bot, &env->robots[target_idx], bspeed);
    }
    float gun_delta = rb_turn_gun_to(bot, aim);
    if (fabsf(gun_delta) < 2.5f && bot->gun_heat <= 0.0f && bot->energy > 1.0f
            && m->last_energy_seen > 0.0f) {
        fire(env, bot, bot_idx, power);
        m->dgt_bullets_passed++;
        // Store virtual fire snapshot for later GF logging when bullet hits.
        m->dgt_last_fire_x = bot->x;
        m->dgt_last_fire_y = bot->y;
        m->dgt_last_fire_bearing = abs_bearing;
        m->dgt_last_fire_lat_dir = m->dgt_lat_dir;
        m->dgt_last_fire_mea = max_escape;
        memcpy(m->dgt_last_fire_feats, gfeats, DGT_FEATS * sizeof(float));
        m->dgt_last_fire_valid = 1;
    }

    // Keep radar locked (scan already turned radar in rb_scan_target).
    rb_turn_radar_to(bot, abs_bearing, 8.0f);
    m->dgt_last_v = bot->v;
}

// Called when an enemy bullet hits this bot — log surfing GF sample.
static void dgt_on_hit_by_bullet(BotMem* m, float bullet_heading, float bullet_power) {
    float speed = rb_bullet_speed(bullet_power);
    DGTWave* best = NULL;
    int best_age = -1;
    for (int wi = 0; wi < DGT_WAVES; wi++) {
        DGTWave* w = &m->dgt_waves[wi];
        if (!w->active) continue;
        if (fabsf(w->speed - speed) > 0.6f) continue;
        int age = m->tick - w->fire_tick;
        if (age > best_age) {
            best_age = age;
            best = w;
        }
    }
    if (best == NULL) return;
    float gf_raw = rb_norm_deg(bullet_heading - best->head_on);
    float mea_rad = asinf(fminf(8.0f / best->speed, 1.0f));
    float mea_deg = fmaxf(mea_rad * RB_R2D, 0.1f);
    float gf = (gf_raw / mea_deg) * (float)best->lat_sign;
    gf = rb_clampf(gf, -1.5f, 1.5f);
    dgt_knn_add(m->dgt_surf_feats, m->dgt_surf_gf, &m->dgt_surf_n, &m->dgt_surf_head,
                DGT_KNN_CAP, best->feats, gf);
    best->active = 0;
}

// Called when our bullet hits the enemy — log gun GF sample.
static void dgt_on_bullet_hit(BotMem* m, float hit_x, float hit_y) {
    if (!m->dgt_last_fire_valid) return;
    m->dgt_bullets_hit++;
    float bearing = rb_abs_bearing_deg(m->dgt_last_fire_x, m->dgt_last_fire_y, hit_x, hit_y);
    float gf_raw = rb_norm_deg(bearing - m->dgt_last_fire_bearing);
    float mea = fmaxf(m->dgt_last_fire_mea, 0.1f);
    float gf = (gf_raw / mea) * m->dgt_last_fire_lat_dir;
    gf = rb_clampf(gf, -1.2f, 1.2f);
    dgt_knn_add(m->dgt_gun_feats, m->dgt_gun_gf, &m->dgt_gun_n, &m->dgt_gun_head,
                DGT_GUN_CAP, m->dgt_last_fire_feats, gf);
    m->dgt_last_fire_valid = 0;
}

#endif  // ROBOCODE_AGENT_DRUSSGT_H
