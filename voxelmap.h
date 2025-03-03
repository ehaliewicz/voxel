#ifndef VOXELMAP_H
#define VOXELMAP_H

#include "config.h"
#include "types.h"

typedef struct {
    u8 top_y;     // duplicate of top in the topmost span of this column
    u8 num_runs;
} column_header;


typedef struct {
    u32 colors[MAP_Z_SIZE];
} column_colors;

typedef struct {
    u32 colors[MIP_MAP_Z_SIZE];
} mip_column_colors;

typedef struct {
    u32 colors[MIP2_MAP_Z_SIZE];
} mip2_column_colors;

typedef struct {
    f32 norm_pt1[MAP_Z_SIZE];
    f32 norm_pt2[MAP_Z_SIZE];
} column_normals;

typedef struct {
    f32 norm_pt1[MIP_MAP_Z_SIZE];
    f32 norm_pt2[MIP_MAP_Z_SIZE];
} mip_column_normals;

// 4 bytes
typedef struct {
    u8 top_surface_start;
    u8 top_surface_end;

    // bot surface exists if end > start
    // ALL bot surfaces must be below top surfaces
    u8 bot_surface_start;
    u8 bot_surface_end;
} span;


// each column is 512 bytes of spans :(
// now 256 bytes :shrug:
typedef struct {
    span runs_info[128]; //MAP_Z_SIZE/2];//128];
} column_runs;

typedef struct {
    span runs_info[64]; //MAP_Z_SIZE/2];//64];
} mip_column_runs;

typedef struct {
    span runs_info[64]; //MAP_Z_SIZE/2];//32];
} mip2_column_runs;


typedef struct {
    u64 bits[4]; // 4
} column_bitmaps;


typedef struct {
    u64 bits[2]; // 
} mip_column_bitmaps;

typedef struct {
    u64 bits[2];
} mip2_column_bitmaps;



// 1024 -> 10 bits
// 1024 -> 10 bits
// 128 -> 7 bits
// 27 bits


// sizeof(column_runs)

int cur_map_max_height; // usually 63 but not always

column_header* columns_header_data = NULL;
column_colors* columns_colors_data = NULL;
column_runs* columns_runs_data = NULL;
column_normals* columns_norm_data = NULL;
column_bitmaps* columns_bitmaps_data = NULL;
u8* map_file_buffer = NULL;


column_header* mip_columns_header_data = NULL;
mip_column_colors* mip_columns_colors_data = NULL;
mip_column_runs* mip_columns_runs_data = NULL;
mip_column_normals* mip_columns_norm_data = NULL;
mip_column_bitmaps* mip_columns_bitmaps_data = NULL;
//column_bitmaps* mip_columns_bitmaps_data = NULL;

column_header* mip2_columns_header_data = NULL;
mip_column_colors* mip2_columns_colors_data = NULL;
mip_column_runs* mip2_columns_runs_data = NULL;
mip_column_normals* mip2_columns_norm_data = NULL;
mip_column_bitmaps* mip2_columns_bitmaps_data = NULL;
//column_bitmaps* mip2_columns_bitmaps_data = NULL;

#endif