typedef struct {
  int depth_idx;
  int color_idx;
  int water_version;
} map_info;

map_info map_table[32] = {
  {.color_idx=1,.depth_idx=1, .water_version = 1},
  {.color_idx=2,.depth_idx=2, .water_version = 1},
  {.color_idx=3,.depth_idx=3},
  {.color_idx=4,.depth_idx=4},
  {.color_idx=5,.depth_idx=5, .water_version = 1},
  {.color_idx=6,.depth_idx=6, .water_version = 1},
  {.color_idx=7,.depth_idx=7, .water_version = 1},
  {.color_idx=8,.depth_idx=6},
  {.color_idx=9,.depth_idx=9, .water_version = 1},
  {.color_idx=10,.depth_idx=10, .water_version = 1},
  {.color_idx=11,.depth_idx=11, .water_version = 1},
  {.color_idx=12,.depth_idx=11, .water_version = 1},
  {.color_idx=13,.depth_idx=13},
  {.color_idx=14,.depth_idx=14},
  {.color_idx=14,.depth_idx=14, .water_version = 1},
  {.color_idx=15,.depth_idx=15},
  {.color_idx=16,.depth_idx=16, .water_version = 1},
  {.color_idx=17,.depth_idx=17, .water_version = 1},
  {.color_idx=18,.depth_idx=18, .water_version = 1},
  {.color_idx=19,.depth_idx=19, .water_version = 1},
  {.color_idx=20,.depth_idx=20, .water_version = 1},
  {.color_idx=21,.depth_idx=21},
  {.color_idx=22,.depth_idx=22, .water_version = 1},
  {.color_idx=23,.depth_idx=21, .water_version = 1},
  {.color_idx=24,.depth_idx=24, .water_version = 1},
  {.color_idx=25,.depth_idx=25, .water_version = 1},
  {.color_idx=26,.depth_idx=18, .water_version = 1},
  {.color_idx=27,.depth_idx=15, .water_version = 1},
  {.color_idx=28,.depth_idx=25, .water_version = 1},
  {.color_idx=29,.depth_idx=16, .water_version = 1},
  {.color_idx=30,.depth_idx=30, .water_version = 0},
  {.color_idx=30,.depth_idx=30, .water_version = 0},
};