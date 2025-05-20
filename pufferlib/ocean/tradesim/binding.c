#include "tradesim.h"

#define Env TradeSim
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->data_path = unpack(kwargs, "data_path");
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
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
    return 0;
}
