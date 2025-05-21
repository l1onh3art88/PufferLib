#include "tradesim.h"

#define Env TradeSim
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->reward_pnl_scale = unpack(kwargs, "reward_pnl_scale");
    env->reward_illegal_move = unpack(kwargs, "reward_illegal_move");
    char data_path[100];
    sprintf(data_path, "resources/tradesim/data.bin");
    env->data_path = data_path;
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "long_win_pct", log->long_win_pct);
    assign_to_dict(dict, "short_win_pct", log->short_win_pct);
    assign_to_dict(dict, "overall_win_pct", log->overall_win_pct);
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "realized_pnl", log->realized_pnl);
    assign_to_dict(dict, "capital", log->capital);
    assign_to_dict(dict, "illegal_move_pct", log->illegal_move_pct);
    return 0;
}
