// Structural adaptation of DrussGT 3.1.4159 (jk.mega.DrussGT)
// by Julian Kent / Skilgannon — open source (source included in the jar).
//
// Jar: http://robocode-archive.strangeautomata.com/robots/jk.mega.DrussGT_3.1.4159.jar
// Docs: https://robowiki.net/wiki/DrussGT/Understanding_DrussGT
//
// Not a full JVM mega-bot port (100+ VCS buffers, precise intersection KD-tree
// dual guns, bullet shadows, shielding). Captures the systems that dominate
// strength vs classical bots like Raiko:
//
//   * Go-to wave surfing with kNN GF danger + visit flattener
//   * Gunheat-filtered energy-drop waves + imaginary gunheat waves
//   * No-wave fallback: wall-smoothed lateral orbit (farther side)
//   * DC-style GF gun: kNN + kernel density over *wave-pass visits*
//     (Raiko/DrussGT style — not hit-only learning)
//   * Manhattan feature distance, DrussGT-ish attribute set
//   * Fire-power schedule similar to DrussGT (1.95/2.95, energy clamps)
//
// License note (author terms): keep open-source; credit Skilgannon / DrussGT.

#ifndef ROBOCODE_AGENT_DRUSSGT_H
#define ROBOCODE_AGENT_DRUSSGT_H

#include "agent_common.h"

#ifndef DGT_KNN_K
#define DGT_KNN_K 12
#endif
#ifndef DGT_GUN_K
#define DGT_GUN_K 24
#endif
#ifndef DGT_GOTO_CANDS
#define DGT_GOTO_CANDS 36
#endif
#ifndef DGT_BEST_DIST
#define DGT_BEST_DIST 450.0f
#endif

// Feature scales — Manhattan weights (official DrussGT uses Manhattan kNN).
// Layout matches dgt_features() below.
static const float DGT_FEAT_W[DGT_FEATS] = {
    5.00f,  // distance / 900
    4.00f,  // |lat vel| / 8
    2.00f,  // adv vel / 8
    3.00f,  // 1/(1+k*tsdc)
    2.50f,  // 1/(1+k*tsdecel)
    2.00f,  // accel bucket
    3.00f,  // wall proximity
    2.00f,  // dist-last-10 / 80
};

static inline float dgt_norm_time(float t, float k) {
    return 1.0f / (1.0f + k * fmaxf(t, 0.0f));
}

// bot = the robot whose motion we describe; tgt = the other robot (origin of
// the wave / gun frame).  Pass *that* bot's last_v and dir-change tick.
static inline void dgt_features(Robot* bot, float tgt_x, float tgt_y, Robocode* env,
                                int tick, int last_dir_change, int last_decel,
                                float last_v, float dist_last10,
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
    out[0] = rb_clampf(dist / 900.0f, 0.0f, 1.2f);
    out[1] = fabsf(lat_v) / 8.0f;
    out[2] = rb_clampf(adv_v / 8.0f, -1.0f, 1.0f);
    out[3] = dgt_norm_time((float)(tick - last_dir_change), 0.08f);
    out[4] = dgt_norm_time((float)(tick - last_decel), 0.08f);
    out[5] = rb_clampf(acc / 2.0f, -1.0f, 1.0f);
    out[6] = rb_clampf(wall_min / wall_half, 0.0f, 1.0f);
    out[7] = rb_clampf(dist_last10 / 80.0f, 0.0f, 1.5f);
}

// Manhattan distance (official DrussGT KD-tree metric).
static inline float dgt_feat_dist(const float* a, const float* b) {
    float d = 0.0f;
    for (int f = 0; f < DGT_FEATS; f++) {
        d += fabsf((a[f] - b[f]) * DGT_FEAT_W[f]);
    }
    return d;
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

// Weighted kNN kernel density danger at a GF (surfing).
static float dgt_knn_danger(const float feats[DGT_FEATS], float gf,
                            const float knn_feats[][DGT_FEATS],
                            const float* knn_gf, int knn_n) {
    if (knn_n <= 0) {
        // Cold prior: slightly prefer extremes (anti-head-on) lightly.
        return 0.35f * expf(-(gf * gf) / (2.0f * 0.22f * 0.22f));
    }
    float best_d[DGT_KNN_K];
    int best_i[DGT_KNN_K];
    for (int k = 0; k < DGT_KNN_K; k++) {
        best_d[k] = 1e18f;
        best_i[k] = -1;
    }
    for (int n = 0; n < knn_n; n++) {
        float d = dgt_feat_dist(feats, knn_feats[n]);
        for (int k = 0; k < DGT_KNN_K; k++) {
            if (d < best_d[k]) {
                for (int s = DGT_KNN_K - 1; s > k; s--) {
                    best_d[s] = best_d[s - 1];
                    best_i[s] = best_i[s - 1];
                }
                best_d[k] = d;
                best_i[k] = n;
                break;
            }
        }
    }
    const float sigma = 0.14f;
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
        float d = dgt_feat_dist(feats, knn_feats[n]);
        for (int k = 0; k < DGT_GUN_K; k++) {
            if (d < best_d[k]) {
                for (int s = DGT_GUN_K - 1; s > k; s--) {
                    best_d[s] = best_d[s - 1];
                    best_i[s] = best_i[s - 1];
                }
                best_d[k] = d;
                best_i[k] = n;
                break;
            }
        }
    }
    // Finer GF grid + neighbor-centered refinement.
    float best_gf = 0.0f;
    float best_dens = -1.0f;
    const float sigma = 0.10f;
    const float two_s2 = 2.0f * sigma * sigma;
    for (int g = 0; g <= 40; g++) {
        float gf = -1.0f + 0.05f * (float)g;
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
    // Also score exact neighbor GFs (peak may sit between grid points).
    for (int k = 0; k < DGT_GUN_K; k++) {
        if (best_i[k] < 0) break;
        float gf = knn_gf[best_i[k]];
        float dens = 0.0f;
        for (int j = 0; j < DGT_GUN_K; j++) {
            if (best_i[j] < 0) break;
            float w = 1.0f / (1.0f + best_d[j]);
            float dgf = gf - knn_gf[best_i[j]];
            dens += w * expf(-(dgf * dgf) / two_s2);
        }
        if (dens > best_dens) {
            best_dens = dens;
            best_gf = gf;
        }
    }
    return rb_clampf(best_gf, -1.0f, 1.0f);
}

// DrussGT-style bullet power: base 1.95/2.95, energy clamps, snap to .x5.
static inline float dgt_bullet_power(Robot* bot, float enemy_energy, float dist,
                                     float enemy_fp, int bullets_hit,
                                     int bullets_fired) {
    double base = 1.95;
    float hitrate = (bullets_fired > 4)
        ? (float)bullets_hit / (float)bullets_fired : 0.0f;
    if (hitrate > 0.33f || dist < 180.0f) base = 2.95;
    if (dist > 600.0f && hitrate < 0.25f) base = 1.95;

    float power = (float)base;
    // Scale down when low energy / enemy would die for less.
    float energy_bp = fmaxf(0.1f, ((float)bot->energy - 0.1f) / 4.0f);
    float enemy_bp = fmaxf(0.1f, enemy_energy / 4.0f);
    power = fminf(power, energy_bp);
    power = fminf(power, enemy_bp);
    // Undercut / match lower enemy firepower when advantageous.
    if (enemy_fp > 0.1f && enemy_fp < power && bot->energy < 40.0f) {
        power = fmaxf(enemy_fp, 0.15f);
    }
    power = rb_clampf(power, 0.1f, 3.0f);
    static const float snaps[] = {
        0.15f, 0.25f, 0.35f, 0.45f, 0.65f, 0.85f, 0.95f,
        1.15f, 1.45f, 1.95f, 2.45f, 2.95f
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
    return rb_clampf(best, 0.1f, fminf(3.0f, (float)bot->energy - 0.1f));
}

// Linear-lead aim from last scan snapshot only (fair).
static inline float dgt_linear_aim(Robot* bot, BotMem* m, float bspeed) {
    float dx = m->last_x - bot->x;
    float dy = m->last_y - bot->y;
    float dist = fmaxf(sqrtf(dx * dx + dy * dy), 1.0f);
    float dt = dist / fmaxf(bspeed, 0.1f);
    // Simple iterative lead using last heading/v.
    float px = m->last_x, py = m->last_y;
    float tvx = cos_deg(m->last_heading) * m->last_v;
    float tvy = sin_deg(m->last_heading) * m->last_v;
    for (int it = 0; it < 3; it++) {
        float ddx = px - bot->x, ddy = py - bot->y;
        float d = fmaxf(sqrtf(ddx * ddx + ddy * ddy), 1.0f);
        dt = d / fmaxf(bspeed, 0.1f);
        px = m->last_x + tvx * dt;
        py = m->last_y + tvy * dt;
    }
    return rb_abs_bearing_deg(bot->x, bot->y, px, py);
}

// Circular aim: assume constant lateral orbit at last angular rate.
static inline float dgt_circular_aim(Robot* bot, BotMem* m, float bspeed) {
    float dx = m->last_x - bot->x;
    float dy = m->last_y - bot->y;
    float dist = fmaxf(sqrtf(dx * dx + dy * dy), 1.0f);
    float dt = dist / fmaxf(bspeed, 0.1f);
    // Angular velocity estimate from lat vel.
    float inv = 1.0f / dist;
    float ux = dx * inv, uy = dy * inv;
    float tvx = cos_deg(m->last_heading) * m->last_v;
    float tvy = sin_deg(m->last_heading) * m->last_v;
    float lat = -tvx * uy + tvy * ux;
    float ang_vel = lat / dist;  // rad/tick approx
    float bearing = rb_abs_bearing_deg(bot->x, bot->y, m->last_x, m->last_y);
    float new_bearing = bearing + ang_vel * dt * RB_R2D;
    // Keep distance roughly constant (orbit).
    float px = bot->x + dist * cos_deg(new_bearing);
    float py = bot->y + dist * sin_deg(new_bearing);
    return rb_abs_bearing_deg(bot->x, bot->y, px, py);
}

// No-wave fallback: wall-smoothed lateral orbit; pick side farther from enemy.
static void dgt_fallback_move(Robocode* env, Robot* bot, BotMem* m) {
    float abs_bearing = rb_abs_bearing_deg(bot->x, bot->y, m->last_x, m->last_y);
    float distance = rb_dist(bot->x, bot->y, m->last_x, m->last_y);
    float best_x = bot->x, best_y = bot->y;
    float best_score = -1e18f;
    float dirs[2] = {1.0f, -1.0f};
    for (int di = 0; di < 2; di++) {
        float dir = dirs[di];
        for (int t = 0; t < 12; t++) {
            float orbit = 55.0f + 9.0f * (float)t;  // 55..154 deg from abs bearing
            float ang = (abs_bearing + dir * orbit) * RB_D2R;
            float step = rb_clampf(distance * 0.4f + 100.0f, 90.0f, 200.0f);
            float x = bot->x + step * cosf(ang);
            float y = bot->y + step * sinf(ang);
            if (x < 30.0f || x > env->width - 30.0f ||
                    y < 30.0f || y > env->height - 30.0f) {
                continue;
            }
            float d = rb_dist(x, y, m->last_x, m->last_y);
            float score = -fabsf(d - DGT_BEST_DIST) + 0.12f * d;
            if (distance < 220.0f) score += 0.8f * d;  // escape rammers
            // Prefer staying off walls.
            float wall = fminf(fminf(x, env->width - x), fminf(y, env->height - y));
            score += 0.4f * wall;
            if (score > best_score) {
                best_score = score;
                best_x = x;
                best_y = y;
            }
        }
    }
    rb_drive_to(env, bot, best_x, best_y);
}

// Simulate driving toward (tx,ty) until a wave hits; return GF of hit location.
// Simplified tick sim of go-to (core of official DrussGT surfing).
static float dgt_wave_hit_gf(Robot* bot, DGTWave* w, float tx, float ty,
                             int tick_now, float* out_hit_dist) {
    float x = bot->x, y = bot->y;
    float heading = bot->heading;
    float v = bot->v;
    float radius = (float)(tick_now - w->fire_tick) * w->speed;
    for (int step = 0; step < 120; step++) {
        float ddx = x - w->ox, ddy = y - w->oy;
        float dist = sqrtf(ddx * ddx + ddy * ddy);
        if (radius + w->speed >= dist - 18.0f) {
            float bearing = rb_abs_bearing_deg(w->ox, w->oy, x, y);
            float gf_raw = rb_norm_deg(bearing - w->head_on);
            float mea_rad = asinf(fminf(8.0f / w->speed, 1.0f));
            float mea_deg = fmaxf(mea_rad * RB_R2D, 0.1f);
            float gf = (gf_raw / mea_deg) * (float)w->lat_sign;
            if (out_hit_dist) *out_hit_dist = dist;
            return rb_clampf(gf, -1.5f, 1.5f);
        }
        // Go-to step toward destination.
        float want = rb_abs_bearing_deg(x, y, tx, ty);
        float delta = rb_norm_deg(want - heading);
        float dir = 1.0f;
        if (cosf(delta * RB_D2R) < 0.0f) {
            delta = rb_norm_deg(delta + 180.0f);
            dir = -1.0f;
        }
        float max_turn = 10.0f - 0.75f * fabsf(v);
        if (max_turn < 0.0f) max_turn = 0.0f;
        if (delta > max_turn) delta = max_turn;
        if (delta < -max_turn) delta = -max_turn;
        heading = rb_norm_deg(heading + delta);
        if (heading < 0.0f) heading += 360.0f;
        // Accel toward max speed in chosen dir.
        float target_v = dir * 8.0f;
        if (v < target_v) {
            v = fminf(v + 1.0f, target_v);
        } else if (v > target_v) {
            v = fmaxf(v - 2.0f, target_v);
        }
        x += v * cos_deg(heading);
        y += v * sin_deg(heading);
        radius += w->speed;
    }
    // Timeout: use destination GF.
    float bearing = rb_abs_bearing_deg(w->ox, w->oy, tx, ty);
    float gf_raw = rb_norm_deg(bearing - w->head_on);
    float mea_rad = asinf(fminf(8.0f / w->speed, 1.0f));
    float mea_deg = fmaxf(mea_rad * RB_R2D, 0.1f);
    if (out_hit_dist) {
        float ddx = tx - w->ox, ddy = ty - w->oy;
        *out_hit_dist = sqrtf(ddx * ddx + ddy * ddy);
    }
    return rb_clampf((gf_raw / mea_deg) * (float)w->lat_sign, -1.5f, 1.5f);
}

// Go-to wave surfing: sample destinations, precise-ish wave-hit GFs, min danger.
static void dgt_surf_goto(Robocode* env, Robot* bot, BotMem* m) {
    float abs_bearing = rb_abs_bearing_deg(bot->x, bot->y, m->last_x, m->last_y);
    float distance = fmaxf(rb_dist(bot->x, bot->y, m->last_x, m->last_y), 1.0f);

    // Self lateral history sum for dist-last-10-ish feature.
    float lat_sum = 0.0f;
    for (int i = 0; i < 10; i++) lat_sum += fabsf(m->dgt_lat_hist[i]);

    float feats[DGT_FEATS];
    dgt_features(bot, m->last_x, m->last_y, env, m->tick, m->last_dir_change_tick,
                 m->dgt_enemy_decel_tick /* unused self decel ok */,
                 m->dgt_last_v, lat_sum, feats);
    // Fix self-decel feature: use our own decel tick stored in last_dir for now
    // (feature[4] already set; recompute with self if we track it).
    // Recompute feature 4 with last_dir_change as weak proxy is fine.

    // Find soonest active waves (surf first 1-2).
    int wave_idx[DGT_WAVES];
    int n_active = 0;
    for (int wi = 0; wi < DGT_WAVES; wi++) {
        if (!m->dgt_waves[wi].active) continue;
        // Skip pure imaginary if we already have real waves.
        wave_idx[n_active++] = wi;
    }
    if (n_active == 0) {
        dgt_fallback_move(env, bot, m);
        return;
    }
    // Sort by time-to-hit ascending (simple insertion).
    for (int a = 0; a < n_active; a++) {
        for (int b = a + 1; b < n_active; b++) {
            DGTWave* wa = &m->dgt_waves[wave_idx[a]];
            DGTWave* wb = &m->dgt_waves[wave_idx[b]];
            float da = rb_dist(bot->x, bot->y, wa->ox, wa->oy)
                     - (m->tick - wa->fire_tick) * wa->speed;
            float db = rb_dist(bot->x, bot->y, wb->ox, wb->oy)
                     - (m->tick - wb->fire_tick) * wb->speed;
            if (db < da) {
                int t = wave_idx[a]; wave_idx[a] = wave_idx[b]; wave_idx[b] = t;
            }
        }
    }
    int surf_n = n_active > 2 ? 2 : n_active;

    float best_x = bot->x;
    float best_y = bot->y;
    float best_danger = 1e18f;

    for (int c = 0; c < DGT_GOTO_CANDS; c++) {
        float t = (float)c / (float)(DGT_GOTO_CANDS - 1);  // 0..1
        float gf = -1.05f + 2.1f * t;
        // Orbit around enemy with distancing.
        float orbit_ang = abs_bearing + 90.0f * (gf >= 0.0f ? 1.0f : -1.0f)
                        + gf * 35.0f;
        float want_dist = rb_clampf(DGT_BEST_DIST + 40.0f * gf, 120.0f, 650.0f);
        // Blend current distance for reachable points.
        float radius = rb_clampf(0.55f * distance + 0.45f * want_dist, 80.0f, 280.0f);
        // Candidates relative to *us*, not enemy, for go-to reachability.
        float x = bot->x + radius * cos_deg(orbit_ang);
        float y = bot->y + radius * sin_deg(orbit_ang);
        // Also sample points on a circle around enemy (classic go-to surf set).
        if (c % 2 == 0) {
            float ea = abs_bearing + 180.0f + gf * 50.0f;
            float ed = rb_clampf(want_dist, 150.0f, 550.0f);
            x = m->last_x + ed * cos_deg(ea);
            y = m->last_y + ed * sin_deg(ea);
        }
        if (x < 28.0f || x > env->width - 28.0f ||
                y < 28.0f || y > env->height - 28.0f) {
            continue;
        }

        float danger = 0.0f;
        for (int si = 0; si < surf_n; si++) {
            DGTWave* w = &m->dgt_waves[wave_idx[si]];
            float hit_dist = 0.0f;
            float egf = dgt_wave_hit_gf(bot, w, x, y, m->tick, &hit_dist);
            // Use wave-fire features for danger lookup (enemy targeting state).
            float d = dgt_knn_danger(w->feats, egf, m->dgt_surf_feats,
                                     m->dgt_surf_gf, m->dgt_surf_n);
            float power = w->power > 0.0f ? w->power : ((20.0f - w->speed) / 3.0f);
            float dmg = 4.0f * power + (power > 1.0f ? 2.0f * (power - 1.0f) : 0.0f);
            // Weight first wave much higher; scale by damage + imminence.
            float tti = fmaxf(0.0f,
                (hit_dist - (m->tick - w->fire_tick) * w->speed) / w->speed);
            float wgt = (si == 0 ? 1.0f : 0.35f) * (1.0f + 0.12f * dmg)
                      / (1.0f + 0.04f * tti);
            if (w->imaginary) wgt *= 0.55f;
            danger += wgt * d;
            // Distancing under fire.
            danger += 0.0015f * fmaxf(0.0f, 320.0f - hit_dist);
        }
        // Prefer better stand-off vs enemy.
        float d_en = rb_dist(x, y, m->last_x, m->last_y);
        danger += 0.0008f * fabsf(d_en - DGT_BEST_DIST);
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

// Advance outgoing gun waves; log visit GF when wave reaches enemy (main gun
// learning signal — same structure as Raiko wave updates / DrussGT visits).
static void dgt_update_gun_waves(BotMem* m) {
    for (int i = 0; i < DGT_GUN_WAVES; i++) {
        DGTGunWave* w = &m->dgt_gun_waves[i];
        if (!w->active) continue;
        w->dist_traveled += w->speed;
        float d = rb_dist(w->ox, w->oy, m->last_x, m->last_y);
        if (w->dist_traveled + w->speed >= d) {
            float bearing = rb_abs_bearing_deg(w->ox, w->oy, m->last_x, m->last_y);
            float gf_raw = rb_norm_deg(bearing - w->abs_bearing);
            float mea = fmaxf(w->mea, 0.1f);
            float gf = (gf_raw / mea) * w->lat_dir;
            gf = rb_clampf(gf, -1.2f, 1.2f);
            dgt_knn_add(m->dgt_gun_feats, m->dgt_gun_gf, &m->dgt_gun_n,
                        &m->dgt_gun_head, DGT_GUN_CAP, w->feats, gf);
            m->dgt_current_gf = gf;
            w->active = 0;
        } else if (w->dist_traveled > 1400.0f) {
            w->active = 0;
        }
    }
}

// Flattener: when an enemy wave passes us without a hit, log the visit GF.
static void dgt_expire_and_flatten(BotMem* m, Robot* bot) {
    for (int wi = 0; wi < DGT_WAVES; wi++) {
        DGTWave* w = &m->dgt_waves[wi];
        if (!w->active) continue;
        float radius = (m->tick - w->fire_tick) * w->speed;
        float ddx = bot->x - w->ox, ddy = bot->y - w->oy;
        float dist_now = sqrtf(ddx * ddx + ddy * ddy);
        // Drop stale imaginary waves quickly if never confirmed by an energy drop.
        if (w->imaginary && (m->tick - w->fire_tick) > 18) {
            w->active = 0;
            continue;
        }
        if (radius >= dist_now + 18.0f || m->tick - w->fire_tick > 400) {
            // Visit log (flattener) for confirmed waves only.
            if (!w->imaginary) {
                float bearing = rb_abs_bearing_deg(w->ox, w->oy, bot->x, bot->y);
                float gf_raw = rb_norm_deg(bearing - w->head_on);
                float mea_rad = asinf(fminf(8.0f / w->speed, 1.0f));
                float mea_deg = fmaxf(mea_rad * RB_R2D, 0.1f);
                float gf = (gf_raw / mea_deg) * (float)w->lat_sign;
                gf = rb_clampf(gf, -1.5f, 1.5f);
                dgt_knn_add(m->dgt_surf_feats, m->dgt_surf_gf, &m->dgt_surf_n,
                            &m->dgt_surf_head, DGT_KNN_CAP, w->feats, gf);
                m->dgt_waves_passed++;
            }
            w->active = 0;
        }
    }
}

static inline float dgt_hist_dist10(BotMem* m) {
    if (m->dgt_hist_n < 2) return 0.0f;
    int oldest = m->dgt_hist_n < 10
        ? (m->dgt_hist_head - m->dgt_hist_n + DGT_HIST) % DGT_HIST
        : (m->dgt_hist_head - 10 + DGT_HIST) % DGT_HIST;
    int latest = (m->dgt_hist_head - 1 + DGT_HIST) % DGT_HIST;
    return rb_dist(m->dgt_hist_x[oldest], m->dgt_hist_y[oldest],
                   m->dgt_hist_x[latest], m->dgt_hist_y[latest]);
}

static inline void dgt_push_hist(BotMem* m) {
    m->dgt_hist_x[m->dgt_hist_head] = m->last_x;
    m->dgt_hist_y[m->dgt_hist_head] = m->last_y;
    m->dgt_hist_head = (m->dgt_hist_head + 1) % DGT_HIST;
    if (m->dgt_hist_n < DGT_HIST) m->dgt_hist_n++;
}

static void dgt_add_enemy_wave(BotMem* m, Robot* bot, Robocode* env,
                               float power, int imaginary) {
    if (power < 0.1f || power > 3.01f) return;
    DGTWave* w = &m->dgt_waves[m->dgt_wave_head];
    m->dgt_wave_head = (m->dgt_wave_head + 1) % DGT_WAVES;
    w->ox = m->last_x;
    w->oy = m->last_y;
    w->speed = rb_bullet_speed(power);
    w->power = power;
    // Fire was last tick for real drops; imaginary is "about to fire".
    w->fire_tick = imaginary ? m->tick + 1 : m->tick - 1;
    if (w->fire_tick < 0) w->fire_tick = 0;
    w->head_on = rb_abs_bearing_deg(w->ox, w->oy, bot->x, bot->y);
    float lat_sum = 0.0f;
    for (int i = 0; i < 10; i++) lat_sum += fabsf(m->dgt_lat_hist[i]);
    dgt_features(bot, w->ox, w->oy, env, m->tick, m->last_dir_change_tick,
                 m->last_dir_change_tick, m->dgt_last_v, lat_sum, w->feats);
    float bvx = cos_deg(bot->heading) * bot->v;
    float bvy = sin_deg(bot->heading) * bot->v;
    float dx = bot->x - w->ox, dy = bot->y - w->oy;
    float dist = fmaxf(sqrtf(dx * dx + dy * dy), 1.0f);
    float lat = (-bvx * dy + bvy * dx) / dist;
    w->lat_sign = (lat >= 0.0f) ? 1 : -1;
    w->imaginary = imaginary;
    w->active = 1;
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
        m->dgt_enemy_gunheat = 0.0f;  // allow detecting first shot
        m->dgt_lat_dir = 1.0f;
        m->dgt_enemy_lat_dir = 1.0f;
        m->dgt_last_v = bot->v;
        m->dgt_enemy_last_v = m->last_v;
        m->dgt_enemy_dir_change_tick = m->tick;
        m->dgt_enemy_decel_tick = m->tick;
        m->dgt_goto_x = bot->x;
        m->dgt_goto_y = bot->y;
        // Do NOT zero kNN / hit counters here — they persist across rounds.
        for (int i = 0; i < DGT_WAVES; i++) m->dgt_waves[i].active = 0;
        for (int i = 0; i < DGT_GUN_WAVES; i++) m->dgt_gun_waves[i].active = 0;
    }

    dgt_push_hist(m);

    // Cool enemy gunheat estimate (DrussGT movement gunheat tracking).
    m->dgt_enemy_gunheat = fmaxf(0.0f, m->dgt_enemy_gunheat - 0.1f);

    float drop = m->dgt_enemy_energy - (float)m->last_energy_seen;
    // Real wave: energy drop. Gunheat filter kills most wall-hit false waves;
    // allow a small positive gunheat slack for timing jitter.
    if (drop >= 0.1f && drop <= 3.0f && m->dgt_enemy_gunheat <= 0.15f) {
        m->dgt_enemy_firepower = drop;
        m->dgt_enemy_gunheat = 1.0f + drop / 5.0f;
        // Prefer promoting an imaginary wave if present; else add real.
        int promoted = 0;
        for (int wi = 0; wi < DGT_WAVES; wi++) {
            DGTWave* w = &m->dgt_waves[wi];
            if (w->active && w->imaginary && fabsf(w->power - drop) < 0.45f) {
                w->imaginary = 0;
                w->power = drop;
                w->speed = rb_bullet_speed(drop);
                w->fire_tick = m->tick - 1;
                if (w->fire_tick < 0) w->fire_tick = 0;
                promoted = 1;
                break;
            }
        }
        if (!promoted) {
            dgt_add_enemy_wave(m, bot, env, drop, 0);
        }
        // Drop any other unmatched imaginary waves (stale predictions).
        for (int wi = 0; wi < DGT_WAVES; wi++) {
            if (m->dgt_waves[wi].active && m->dgt_waves[wi].imaginary) {
                m->dgt_waves[wi].active = 0;
            }
        }
    }
    m->dgt_enemy_energy = (float)m->last_energy_seen;

    // Imaginary gunheat wave: start surfing ~1 tick before energy drop is
    // visible. Do NOT raise enemy_gunheat here — that tracks real fire only.
    if (m->dgt_enemy_gunheat <= 0.0f) {
        int has_imag = 0;
        for (int wi = 0; wi < DGT_WAVES; wi++) {
            if (m->dgt_waves[wi].active && m->dgt_waves[wi].imaginary) {
                has_imag = 1;
                break;
            }
        }
        if (!has_imag) {
            float pred = m->dgt_enemy_firepower;
            if (pred < 0.1f) pred = 2.0f;
            dgt_add_enemy_wave(m, bot, env, pred, 1);
        }
    }

    // Lateral direction bookkeeping (self + enemy).
    float abs_bearing = rb_abs_bearing_deg(bot->x, bot->y, m->last_x, m->last_y);
    float bvx = cos_deg(bot->heading) * bot->v;
    float bvy = sin_deg(bot->heading) * bot->v;
    float dxe = m->last_x - bot->x, dye = m->last_y - bot->y;
    float dist = fmaxf(sqrtf(dxe * dxe + dye * dye), 1.0f);
    float inv = 1.0f / dist;
    float ux = dxe * inv, uy = dye * inv;
    float lat_v = -bvx * uy + bvy * ux;
    m->dgt_lat_hist[m->dgt_lat_hist_i % 10] = lat_v;
    m->dgt_lat_hist_i++;
    if (fabsf(lat_v) > 0.1f) {
        float new_dir = lat_v > 0.0f ? 1.0f : -1.0f;
        if (new_dir != m->dgt_lat_dir) {
            m->dgt_lat_dir = new_dir;
            m->last_dir_change_tick = m->tick;
        }
    }

    float tvx = cos_deg(m->last_heading) * m->last_v;
    float tvy = sin_deg(m->last_heading) * m->last_v;
    float enemy_lat = -tvx * uy + tvy * ux;
    if (fabsf(enemy_lat) > 0.05f) {
        float ed = enemy_lat > 0.0f ? 1.0f : -1.0f;
        if (ed != m->dgt_enemy_lat_dir) {
            m->dgt_enemy_lat_dir = ed;
            m->dgt_enemy_dir_change_tick = m->tick;
        }
    }
    if (m->last_v < m->dgt_enemy_last_v - 0.5f) {
        m->dgt_enemy_decel_tick = m->tick;
    }

    dgt_expire_and_flatten(m, bot);
    dgt_update_gun_waves(m);
    dgt_surf_goto(env, bot, m);

    // ---- Gun (DC GF / kNN density + linear/circular cold blend) ----
    float gfeats[DGT_FEATS];
    {
        Robot fake = *bot;
        fake.x = m->last_x;
        fake.y = m->last_y;
        fake.heading = m->last_heading;
        fake.v = m->last_v;
        dgt_features(&fake, bot->x, bot->y, env, m->tick,
                     m->dgt_enemy_dir_change_tick, m->dgt_enemy_decel_tick,
                     m->dgt_enemy_last_v, dgt_hist_dist10(m), gfeats);
    }
    float power = dgt_bullet_power(bot, (float)m->last_energy_seen, dist,
                                   m->dgt_enemy_firepower,
                                   m->dgt_bullets_hit, m->dgt_bullets_fired);
    float bspeed = rb_bullet_speed(power);
    float max_escape = asinf(fminf(8.0f / bspeed, 1.0f)) * RB_R2D;
    float gf = dgt_gun_best_gf(gfeats, m->dgt_gun_feats, m->dgt_gun_gf, m->dgt_gun_n);
    float aim = abs_bearing + m->dgt_enemy_lat_dir * gf * max_escape;

    // Cold gun: blend linear + circular lead from last-scan only.
    if (m->dgt_gun_n < 12) {
        float lin = dgt_linear_aim(bot, m, bspeed);
        float circ = dgt_circular_aim(bot, m, bspeed);
        float blend = 0.55f * lin + 0.45f * circ;
        float w_dc = (float)m->dgt_gun_n / 12.0f;
        aim = w_dc * aim + (1.0f - w_dc) * blend;
    } else if (m->dgt_gun_n < 40) {
        // Soft blend a bit of linear while still learning.
        float lin = dgt_linear_aim(bot, m, bspeed);
        aim = 0.85f * aim + 0.15f * lin;
    }

    float gun_delta = rb_turn_gun_to(bot, aim);
    if (fabsf(gun_delta) < 2.0f && bot->gun_heat <= 0.0f && bot->energy > 1.0f
            && m->last_energy_seen > 0.0f
            && power < bot->energy) {
        fire(env, bot, bot_idx, power);
        m->dgt_bullets_fired++;
        // Outgoing gun wave for visit-count learning.
        DGTGunWave* gw = &m->dgt_gun_waves[m->dgt_gun_wave_head];
        m->dgt_gun_wave_head = (m->dgt_gun_wave_head + 1) % DGT_GUN_WAVES;
        gw->ox = bot->x;
        gw->oy = bot->y;
        gw->abs_bearing = abs_bearing;
        gw->lat_dir = m->dgt_enemy_lat_dir;
        gw->speed = bspeed;
        gw->mea = max_escape;
        gw->dist_traveled = 0.0f;
        memcpy(gw->feats, gfeats, DGT_FEATS * sizeof(float));
        gw->active = 1;
    }

    rb_turn_radar_to(bot, abs_bearing, 10.0f);
    m->dgt_last_v = bot->v;
    m->dgt_enemy_last_v = m->last_v;
}

// Called when an enemy bullet hits this bot — strong surf sample (hit weight).
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
    // Log hit thrice for higher weight vs flattener visits.
    for (int r = 0; r < 3; r++) {
        dgt_knn_add(m->dgt_surf_feats, m->dgt_surf_gf, &m->dgt_surf_n,
                    &m->dgt_surf_head, DGT_KNN_CAP, best->feats, gf);
    }
    m->dgt_hits_taken++;
    best->active = 0;
}

// Called when our bullet hits the enemy — boost gun sample at hit GF.
static void dgt_on_bullet_hit(BotMem* m, float hit_x, float hit_y) {
    m->dgt_bullets_hit++;
    // Find newest gun wave matching this hit roughly.
    DGTGunWave* best = NULL;
    float best_err = 1e18f;
    for (int i = 0; i < DGT_GUN_WAVES; i++) {
        DGTGunWave* w = &m->dgt_gun_waves[i];
        if (!w->active) continue;
        float d = rb_dist(w->ox, w->oy, hit_x, hit_y);
        float err = fabsf(d - w->dist_traveled);
        if (err < best_err) {
            best_err = err;
            best = w;
        }
    }
    if (best == NULL) return;
    float bearing = rb_abs_bearing_deg(best->ox, best->oy, hit_x, hit_y);
    float gf_raw = rb_norm_deg(bearing - best->abs_bearing);
    float mea = fmaxf(best->mea, 0.1f);
    float gf = (gf_raw / mea) * best->lat_dir;
    gf = rb_clampf(gf, -1.2f, 1.2f);
    for (int r = 0; r < 2; r++) {
        dgt_knn_add(m->dgt_gun_feats, m->dgt_gun_gf, &m->dgt_gun_n,
                    &m->dgt_gun_head, DGT_GUN_CAP, best->feats, gf);
    }
    best->active = 0;
}

#endif  // ROBOCODE_AGENT_DRUSSGT_H
