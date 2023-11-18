
typedef struct {
    s32 min_x, min_y;
    s32 max_x, max_y;
    volatile uint64_t finished;
} thread_params;