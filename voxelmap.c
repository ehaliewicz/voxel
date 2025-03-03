#ifndef VOXELMAP_C
#define VOXELMAP_C

#include "stdio.h"

#include <dirent.h> 
#include "map_table.c"

#include "types.h"
#include "config.h"


// 4 bytes per run entry

// 4x4 chunk
// currently it'll load half of an 8x8 tile i think?

// 4x4 chunks

// 32 entry cache line


#include "voxelmap.h"

static int map_data_allocated = 0;

typedef enum {
    IN_RUN,
    OUT_OF_RUN
} run_state;

void allocate_map_data() {
    columns_header_data = malloc_wrapper(sizeof(column_header)*MAP_X_SIZE*MAP_Y_SIZE, "column headers");
    columns_colors_data = malloc_wrapper(sizeof(column_colors)*MAP_X_SIZE*MAP_Y_SIZE, "column colors");
    columns_runs_data = malloc_wrapper(sizeof(column_runs)*MAP_X_SIZE*MAP_Y_SIZE, "column runs");
    //columns_norm_data = malloc_wrapper(sizeof(column_normals)*MAP_X_SIZE*MAP_Y_SIZE, "column normals");
    columns_bitmaps_data = malloc_wrapper(sizeof(column_bitmaps)*MAP_X_SIZE*MAP_Y_SIZE, "column bitmaps");
    map_file_buffer = malloc_wrapper(32768*1024, "map file buffer");

    mip_columns_header_data = malloc_wrapper(sizeof(column_header)*MIP_MAP_X_SIZE*MIP_MAP_Y_SIZE, "mip column headers");
    mip_columns_colors_data = malloc_wrapper(sizeof(mip_column_colors)*MIP_MAP_X_SIZE*MIP_MAP_Y_SIZE, "mip column colors");
    mip_columns_runs_data = malloc_wrapper(sizeof(mip_column_runs)*MIP_MAP_X_SIZE*MIP_MAP_Y_SIZE, "mip column runs");
    //mip_columns_norm_data = malloc_wrapper(sizeof(mip_column_normals)*MIP_MAP_X_SIZE*MIP_MAP_Y_SIZE, "mip column normals");
    mip_columns_bitmaps_data = malloc_wrapper(sizeof(mip_column_bitmaps)*MIP_MAP_X_SIZE*MIP_MAP_Y_SIZE, "mip column bitmaps");

    
    //mip2_columns_header_data = malloc_wrapper(sizeof(column_header)*MIP2_MAP_X_SIZE*MIP2_MAP_Y_SIZE, "mip2 column headers");
    //mip2_columns_colors_data = malloc_wrapper(sizeof(mip_column_colors)*MIP2_MAP_X_SIZE*MIP2_MAP_Y_SIZE, "mip2 column colors");
    //mip2_columns_runs_data = malloc_wrapper(sizeof(mip_column_runs)*MIP2_MAP_X_SIZE*MIP2_MAP_Y_SIZE, "mip2 column runs");
    //mip2_columns_norm_data = malloc_wrapper(sizeof(mip_column_normals)*MIP2_MAP_X_SIZE*MIP2_MAP_Y_SIZE, "mip2 column normals");
    //mip2_columns_bitmaps_data = malloc_wrapper(sizeof(mip_column_bitmaps)*MIP2_MAP_X_SIZE*MIP2_MAP_Y_SIZE, "mip2 column bitmaps");
    //mip_columns_header_data = malloc(sizeof(column_header)*512*512);
    //mip_columns_colors_data = malloc(sizeof(column_colors)*512*512);
    //mip_columns_runs_data = malloc(sizeof(column_runs)*512*512);
    //mip_columns_norm_data = malloc(sizeof(column_normals)*512*512);
    map_data_allocated = 1;

}



u32 get_voxelmap_idx(s32 x, s32 y) {
    //return y*MAP_X_SIZE+x;
    //return (y*MAP_X_SIZE)+x;
    //return m2D_e_BMI(x, y);
    y &= (MAP_Y_SIZE-1);
    x &= (MAP_X_SIZE-1);

    // 2 bits each
    u32 tile_x = x >> 8; // 2 bits each 
    u32 tile_y = y >> 8;

    // 256x256 chunks
    y &= 255;
    x &= 255;
    //y &= 1023;
    //x &= 1023;

    

    // 4x4 chunks gathered into groups of 64x64,  so the tiles are 256x256
    u32 low_x = x&0b11; // 2 bits
    u32 low_y = y&0b11;  
    u32 high_x = x>>2;   // 6 bits
    u32 high_y = y>>2;
    // 20 bit result
    return (tile_y<<18)|(tile_x<<16)|(high_y<<10)|(high_x<<4)|(low_y<<2)|low_x;
}

u32 get_mip_voxelmap_idx(s32 x, s32 y) {

    //x >>= 1;
    //y >>= 1;
    //return (y*MIP_MAP_X_SIZE) + x;

    // x and y only go up to 512
    // tiles are 4x4
    //if(x > 0 && y > 0 ) {
    //    x++;
    //    x--;
    //}
    x >>= 1;
    y >>= 1;

    // chunks are 128x128 chunks of 4x4 tiles (64x64 in tiles, so 6 bits)
    x &= (MIP_MAP_X_SIZE-1);
    y &= (MIP_MAP_Y_SIZE-1);

    u32 tile_x = x >> 7; // 2 bits
    u32 tile_y = y >> 7; // 2 bits 

    y &= 127;
    x &= 127;

    // 4x4 chunks gathered into groups of 64x64,  so the tiles are 256x256
    u32 low_x = x&0b11; // 2 bits
    u32 low_y = y&0b11;  
    u32 high_x = x>>2;   // 5 bits
    u32 high_y = y>>2;
    // 18 bit result
    return (tile_y<<16)|(tile_x<<14)|(high_y<<9)|(high_x<<4)|(low_y<<2)|low_x;
}


u32 get_mip2_voxelmap_idx(s32 x, s32 y) {
    // x and y only go up to 256
    x >>= 2;
    y >>= 2;

    // chunks are 64x64 chunks of 4x4 tiles
    x &= (MIP2_MAP_X_SIZE-1);
    y &= (MIP2_MAP_Y_SIZE-1);

    u32 tile_x = x >> 6; // shift off the 6 bits because chunks are 64x64 at this level
    u32 tile_y = y >> 6;

    x &= 63;
    y &= 63;

    u32 low_x = x&0b11; // 2 bits
    u32 low_y = y&0b11;  
    u32 high_x = x>>2;   // 4 bits
    u32 high_y = y>>2;

    // 16 bit result
    return (tile_y<<14)|(tile_x<<12)|(high_y<<8)|(high_x<<4)|(low_y<<2)|low_x;
}


void set_bit_in_bitmap(s32 map_x, s32 map_y, s32 map_z) {
    u32 idx = get_voxelmap_idx(map_x, map_y);
    int qw_idx = map_z >> 6;
    int qw_bit = map_z & 63;
    columns_bitmaps_data[idx].bits[qw_idx] |= (((u64)1) << qw_bit);
}

void clear_bit_in_bitmap(s32 map_x, s32 map_y, s32 map_z) {
    u32 idx = get_voxelmap_idx(map_x, map_y);
    int qw_idx = map_z >> 6;
    int qw_bit = map_z & 63;
    columns_bitmaps_data[idx].bits[qw_idx] &= ((~((u64)1)) << qw_bit);
}

u64 get_bit_in_bitmap(int z, column_bitmaps* bmp) {
    int qw_idx = z >> 6;
    int qw_bit = z & 63;
    return bmp->bits[qw_idx] & ((u64)((u64)1) << qw_bit);
}

u64 get_bit_in_mip1_bitmap(int z, mip_column_bitmaps* bmp) {
    int qw_idx = z >> 6;
    int qw_bit = z & 63;
    return bmp->bits[qw_idx] & ((u64)((u64)1) << qw_bit);
}

u64 get_bit_in_mip2_bitmap(int z, mip2_column_bitmaps* bmp) {
    int qw_idx = z >> 6;
    int qw_bit = z & 63;
    return bmp->bits[qw_idx] & ((u64)((u64)1) << qw_bit);
}

u64 count_set_bits_in_voxelmap() {
    u64 tot = 0;
    for(int i = 0; i < 1024*1024; i++) {
        for(int j = 0; j < 4; j++) {
            tot += __builtin_popcountll(columns_bitmaps_data[i].bits[j]);
        }
    }
    return tot;
}



void col_to_surf_bitmap(u32 voxelmap_idx, column_bitmaps* bmp) {
    column_header header = columns_header_data[voxelmap_idx];
    span* runs = columns_runs_data[voxelmap_idx].runs_info;
    memset(bmp->bits, 0, sizeof(bmp->bits));
    for(int i = 0; i < header.num_runs; i++) {
        for(int z = runs[i].top_surface_start; z < runs[i].top_surface_end+1; z++) {
            int qw_idx = z >> 6;
            int qw_bit = (z & 0b111111);
            bmp->bits[qw_idx] |= ((u64)((u64)1) << qw_bit);
        }
        for(int z = runs[i].bot_surface_start; z < runs[i].bot_surface_end; z++) {
            int qw_idx = z >> 6;
            int qw_bit = (z & 0b111111);
            bmp->bits[qw_idx] |= ((u64)((u64)1) << qw_bit);
        }
    }
}

void col_to_solid_bitmap(u32 voxelmap_idx, column_bitmaps* bmp) {
    column_header header = columns_header_data[voxelmap_idx];
    span* runs = columns_runs_data[voxelmap_idx].runs_info;
    memset(bmp->bits, 0, sizeof(bmp->bits));
    for(int i = 0; i < header.num_runs; i++) {
        for(int z = runs[i].top_surface_start; z < runs[i].bot_surface_end; z++) {
            int qw_idx = z >> 6;
            int qw_bit = (z & 0b111111);
            bmp->bits[qw_idx] |= ((u64)((u64)1) << qw_bit);
        }
    }
}



#ifdef AVX2

__m256i get_voxelmap_idx_256(__m256i xs, __m256i ys) {
    __m256i ten_twenty_three_vec = _mm256_set1_epi32(1023);
    __m256i two_fifty_five_vec = _mm256_set1_epi32(255);
    __m256i low_two_bits_vec = _mm256_set1_epi32(0b11);
    __m256i wrapped_xs = _mm256_and_si256(xs, ten_twenty_three_vec);
    __m256i wrapped_ys = _mm256_and_si256(ys, ten_twenty_three_vec);
    //return (y<<1024)|x;

    // 2 bits each
    __m256i tile_xs = _mm256_srli_epi32(wrapped_xs, 8); // 9 for tiles of 512x512, 8 for 2x2 tiles of 256x256, 7 for tiles of 128x128
    __m256i tile_ys = _mm256_srli_epi32(wrapped_ys, 8);


    // 256x256 chunks
    //__m256i wrapped_tile_xs = wrapped_xs & 255;
    //__m256i wrapped_tile_ys = wrapped_ys & 255;
    __m256i wrapped_tile_xs = _mm256_and_si256(wrapped_xs, two_fifty_five_vec);
    __m256i wrapped_tile_ys = _mm256_and_si256(wrapped_ys, two_fifty_five_vec);
    //y &= 1023;
    //x &= 1023;
    __m256i tile_low_xs = _mm256_and_si256(wrapped_tile_xs, low_two_bits_vec); //wrapped_tile_xs & 0b111;
    __m256i tile_low_ys = _mm256_and_si256(wrapped_tile_ys, low_two_bits_vec); //wrapped_tile_ys & 0b111;
    __m256i tile_high_xs = _mm256_srli_epi32(wrapped_tile_xs, 2);
    __m256i tile_high_ys = _mm256_srli_epi32(wrapped_tile_ys, 2);
    return (tile_ys<<18)|(tile_xs<<16)|(tile_high_ys<<10)|(tile_high_xs<<4)|(tile_low_ys<<2)|tile_low_xs;
}
#endif

typedef enum {
    CONTINUE_ITER = 0,
    STOP_ITER = 1,
} iter_res;

typedef iter_res (*rle_solid_chunk_func)(int top, int exclusive_bot, int span_idx);
typedef iter_res (*rle_surface_chunk_func)(int top, int exclusive_bot, int span_idx, int cumulative_skipped_surface_voxels);


static inline void for_each_surface_chunk_in_column(u32 map_x, u32 map_y, rle_surface_chunk_func fp) {
    u32 idx = get_voxelmap_idx(map_x, map_y);
    span* runs = &columns_runs_data[idx].runs_info[0];
    int cumulative_skipped_surface_voxels = 0;
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        int top = runs[i].top_surface_start;
        int exclusive_bot = runs[i].top_surface_end+1;
        if(fp(top, exclusive_bot, i, cumulative_skipped_surface_voxels) == STOP_ITER) {
            break;
        }
        cumulative_skipped_surface_voxels += (exclusive_bot - top);
        top = runs[i].bot_surface_start;
        exclusive_bot = runs[i].bot_surface_end;
        if(exclusive_bot > top) {
            if(fp(top, exclusive_bot, i, cumulative_skipped_surface_voxels) == STOP_ITER) {
                break;
            }
            cumulative_skipped_surface_voxels += (exclusive_bot - top);
        }
    }
}

static inline void for_each_solid_chunk_in_column(u32 map_x, u32 map_y, rle_solid_chunk_func fp) {
    u32 idx = get_voxelmap_idx(map_x, map_y);
    span* runs = &columns_runs_data[idx].runs_info[0];
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        int has_bot_surf = (runs[i].bot_surface_end > runs[i].bot_surface_start);

        // NOTE: this expects all bottom surfaces to be below top surfaces
        // this probably should always be the case, 
        int bot = has_bot_surf ? runs[i].bot_surface_end : (runs[i].top_surface_end+1);
        if(fp(runs[i].top_surface_start, bot, i) == STOP_ITER) {
            break;
        }
    }
}


s32 get_world_pos_for_color_slot(u32 map_x, u32 map_y, int voxel_slot) {    
    int result = -1;
    iter_res span_check_func(int top, int exclusive_bot, int span_idx, int cumulative_skipped_surface_voxels) {
        int surf_len = exclusive_bot - top;
        int slot_relative_to_this_chunk = (voxel_slot - cumulative_skipped_surface_voxels);
        if(slot_relative_to_this_chunk < surf_len) {
            result = top + slot_relative_to_this_chunk;
            return STOP_ITER;
        }
    }

    for_each_surface_chunk_in_column(map_x, map_y, span_check_func);
    
    return result;
}

s32 get_color_slot_for_world_pos(s32 x, s32 y, s32 z) {
    //s16 color_slot_base_offset = 0;
    s32 slot_idx = -1;
    iter_res span_check_func(int top, int exclusive_bot, int span_idx, int cumulative_skipped_surface_voxels) {
        if(z >= top && z < exclusive_bot) {
            slot_idx = cumulative_skipped_surface_voxels + (z - top);
            return STOP_ITER;
        }
        //color_slot_base_offset += (exclusive_bot - top);
    }

    for_each_surface_chunk_in_column(x, y, &span_check_func);
    return slot_idx;
}

u32 get_voxel_color(s32 x, s32 y, s32 z) {
    u32 idx = get_voxelmap_idx(x, y);
    u32* color_ptr = &columns_colors_data[idx].colors[0];
    span* runs = &columns_runs_data[idx].runs_info[0];
    int cumulative_skipped_voxels = 0;

    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        int top = runs[i].top_surface_start;
        int exclusive_bot = runs[i].top_surface_end+1;
        if(z >= top && z < exclusive_bot) {
            return color_ptr[cumulative_skipped_voxels + (z - top)];  // used to be color_slot_base_offset + (z-top)
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
        top = runs[i].bot_surface_start;
        exclusive_bot = runs[i].bot_surface_end;
        if(z >= top && z < exclusive_bot) {
            return color_ptr[cumulative_skipped_voxels + (z - top)];  // used to be color_slot_base_offset + (z-top)
        }
        if(exclusive_bot > top) {
            cumulative_skipped_voxels += (exclusive_bot - top);
        }
    }
    return 0;
}


u32 get_mip1_voxel_color(s32 x, s32 y, s32 z) {
    u32 idx = get_mip_voxelmap_idx(x, y);
    u32* color_ptr = &mip_columns_colors_data[idx].colors[0];
    span* runs = &mip_columns_runs_data[idx].runs_info[0];
    int cumulative_skipped_voxels = 0;

    for(int i = 0; i < mip_columns_header_data[idx].num_runs; i++) {
        int top = runs[i].top_surface_start;
        int exclusive_bot = runs[i].top_surface_end+1;
        if(z >= top && z < exclusive_bot) {
            return color_ptr[cumulative_skipped_voxels + (z - top)];  // used to be color_slot_base_offset + (z-top)
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
        top = runs[i].bot_surface_start;
        exclusive_bot = runs[i].bot_surface_end;
        if(z >= top && z < exclusive_bot) {
            return color_ptr[cumulative_skipped_voxels + (z - top)];  // used to be color_slot_base_offset + (z-top)
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
    }
    return 0;
}



u32 get_mip2_voxel_color(s32 x, s32 y, s32 z) {
    u32 idx = get_mip2_voxelmap_idx(x, y);
    u32* color_ptr = &mip2_columns_colors_data[idx].colors[0];
    span* runs = &mip2_columns_runs_data[idx].runs_info[0];
    int cumulative_skipped_voxels = 0;

    for(int i = 0; i < mip2_columns_header_data[idx].num_runs; i++) {
        int top = runs[i].top_surface_start;
        int exclusive_bot = runs[i].top_surface_end+1;
        if(z >= top && z < exclusive_bot) {
            return color_ptr[cumulative_skipped_voxels + (z - top)];  // used to be color_slot_base_offset + (z-top)
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
        top = runs[i].bot_surface_start;
        exclusive_bot = runs[i].bot_surface_end;
        if(z >= top && z < exclusive_bot) {
            return color_ptr[cumulative_skipped_voxels + (z - top)];  // used to be color_slot_base_offset + (z-top)
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
    }
    return 0;
}


void voxel_set_color(s32 x, s32 y, s32 z, u32 color) {
    u32 idx = get_voxelmap_idx(x, y);
    u32* color_ptr = &columns_colors_data[idx].colors[0];
    span* runs = &columns_runs_data[idx].runs_info[0];
    int cumulative_skipped_voxels = 0;

    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        int top = runs[i].top_surface_start;
        int exclusive_bot = runs[i].top_surface_end+1;
        if(z >= top && z < exclusive_bot) {
            color_ptr[cumulative_skipped_voxels + (z - top)] = color;  // used to be color_slot_base_offset + (z-top)
            break;
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
        top = runs[i].bot_surface_start;
        exclusive_bot = runs[i].bot_surface_end;
        if(z >= top && z < exclusive_bot) {
            color_ptr[cumulative_skipped_voxels + (z - top)] = color;  // used to be color_slot_base_offset + (z-top)
            break;
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
    }
}


void voxel_mip1_set_color(s32 x, s32 y, s32 z, u32 color) {
    u32 idx = get_mip_voxelmap_idx(x, y);
    u32* color_ptr = &mip_columns_colors_data[idx].colors[0];
    span* runs = &mip_columns_runs_data[idx].runs_info[0];
    int cumulative_skipped_voxels = 0;

    for(int i = 0; i < mip_columns_header_data[idx].num_runs; i++) {
        int top = runs[i].top_surface_start;
        int exclusive_bot = runs[i].top_surface_end+1;
        if(z >= top && z < exclusive_bot) {
            color_ptr[cumulative_skipped_voxels + (z - top)] = color;  // used to be color_slot_base_offset + (z-top)
            break;
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
        top = runs[i].bot_surface_start;
        exclusive_bot = runs[i].bot_surface_end;
        if(z >= top && z < exclusive_bot) {
            color_ptr[cumulative_skipped_voxels + (z - top)] = color;  // used to be color_slot_base_offset + (z-top)
            break;
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
    }
}


void voxel_mip2_set_color(s32 x, s32 y, s32 z, u32 color) {
    u32 idx = get_mip2_voxelmap_idx(x, y);
    u32* color_ptr = &mip2_columns_colors_data[idx].colors[0];
    span* runs = &mip2_columns_runs_data[idx].runs_info[0];
    int cumulative_skipped_voxels = 0;

    for(int i = 0; i < mip2_columns_header_data[idx].num_runs; i++) {
        int top = runs[i].top_surface_start;
        int exclusive_bot = runs[i].top_surface_end+1;
        if(z >= top && z < exclusive_bot) {
            color_ptr[cumulative_skipped_voxels + (z - top)] = color;  // used to be color_slot_base_offset + (z-top)
            break;
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
        top = runs[i].bot_surface_start;
        exclusive_bot = runs[i].bot_surface_end;
        if(z >= top && z < exclusive_bot) {
            color_ptr[cumulative_skipped_voxels + (z - top)] = color;  // used to be color_slot_base_offset + (z-top)
            break;
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
    }
}


void voxel_set_normal(s32 x, s32 y, s32 z, f32 norm_pt1, f32 norm_pt2) {
    u32 idx = get_voxelmap_idx(x, y);
    f32* norm_pt1_ptr = &columns_norm_data[idx].norm_pt1[0];
    f32* norm_pt2_ptr = &columns_norm_data[idx].norm_pt2[0];
    span* runs = &columns_runs_data[idx].runs_info[0];
    int cumulative_skipped_voxels = 0;

    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        int top = runs[i].top_surface_start;
        int exclusive_bot = runs[i].top_surface_end+1;
        if(z >= top && z < exclusive_bot) {
            norm_pt1_ptr[cumulative_skipped_voxels + (z-top)] = norm_pt1; // used to be norm_slot_base_offset
            norm_pt2_ptr[cumulative_skipped_voxels + (z-top)] = norm_pt2;
            break;
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
        top = runs[i].bot_surface_start;
        exclusive_bot = runs[i].bot_surface_end;
        if(z >= top && z < exclusive_bot) {
            norm_pt1_ptr[cumulative_skipped_voxels + (z-top)] = norm_pt1; // used to be norm_slot_base_offset
            norm_pt2_ptr[cumulative_skipped_voxels + (z-top)] = norm_pt2;
            break;
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
    }
}

void voxel_mip1_set_normal(s32 x, s32 y, s32 z, f32 norm_pt1, f32 norm_pt2) {
    u32 idx = get_mip_voxelmap_idx(x, y);
    f32* norm_pt1_ptr = &mip_columns_norm_data[idx].norm_pt1[0];
    f32* norm_pt2_ptr = &mip_columns_norm_data[idx].norm_pt2[0];
    span* runs = &mip_columns_runs_data[idx].runs_info[0];
    int cumulative_skipped_voxels = 0;

    for(int i = 0; i < mip_columns_header_data[idx].num_runs; i++) {
        int top = runs[i].top_surface_start;
        int exclusive_bot = runs[i].top_surface_end+1;
        if(z >= top && z < exclusive_bot) {
            norm_pt1_ptr[cumulative_skipped_voxels + (z-top)] = norm_pt1; // used to be norm_slot_base_offset
            norm_pt2_ptr[cumulative_skipped_voxels + (z-top)] = norm_pt2;
            break;
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
        top = runs[i].bot_surface_start;
        exclusive_bot = runs[i].bot_surface_end;
        if(z >= top && z < exclusive_bot) {
            norm_pt1_ptr[cumulative_skipped_voxels + (z-top)] = norm_pt1; // used to be norm_slot_base_offset
            norm_pt2_ptr[cumulative_skipped_voxels + (z-top)] = norm_pt2;
            break;
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
    }
}

void voxel_mip2_set_normal(s32 x, s32 y, s32 z, f32 norm_pt1, f32 norm_pt2) {
    u32 idx = get_mip2_voxelmap_idx(x, y);
    f32* norm_pt1_ptr = &mip2_columns_norm_data[idx].norm_pt1[0];
    f32* norm_pt2_ptr = &mip2_columns_norm_data[idx].norm_pt2[0];
    span* runs = &mip2_columns_runs_data[idx].runs_info[0];
    int cumulative_skipped_voxels = 0;

    for(int i = 0; i < mip2_columns_header_data[idx].num_runs; i++) {
        int top = runs[i].top_surface_start;
        int exclusive_bot = runs[i].top_surface_end+1;
        if(z >= top && z < exclusive_bot) {
            norm_pt1_ptr[cumulative_skipped_voxels + (z-top)] = norm_pt1; // used to be norm_slot_base_offset
            norm_pt2_ptr[cumulative_skipped_voxels + (z-top)] = norm_pt2;
            break;
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
        top = runs[i].bot_surface_start;
        exclusive_bot = runs[i].bot_surface_end;
        if(z >= top && z < exclusive_bot) {
            norm_pt1_ptr[cumulative_skipped_voxels + (z-top)] = norm_pt1; // used to be norm_slot_base_offset
            norm_pt2_ptr[cumulative_skipped_voxels + (z-top)] = norm_pt2;
            break;
        }
        cumulative_skipped_voxels += (exclusive_bot - top);
    }
}

int voxel_is_solid(s32 map_x, s32 map_y, s32 map_z) {
    // TODO: maybe binary search
    if(map_x < 0 || map_x > MAP_X_SIZE || map_y < 0 || map_y >= MAP_Y_SIZE || map_z < 0 || map_z >= (cur_map_max_height+1)) {
        return 0;
    }

    u32 idx = get_voxelmap_idx(map_x, map_y);
    span* runs = &columns_runs_data[idx].runs_info[0];
    int cumulative_skipped_voxels = 0;

    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        int top = runs[i].top_surface_start;
        int exclusive_bot = max(runs[i].top_surface_end+1, runs[i].bot_surface_end);
        if(map_z >= top && map_z < exclusive_bot) {
            return 1;
        }
    }
    return 0;
}

int check_for_solid_voxel_in_aabb(s32 x, s32 y, s32 z, s32 xsize, s32 ysize, s32 zsize) {
    for(int ty = y-ysize; ty < y+ysize+1; ty++) {
        if(ty < 0 || ty >= MAP_Y_SIZE) {
            return 0;
        }
        for(int tx = x-xsize; tx < x+xsize+1; tx++) {
            if(tx < 0 || tx >= MAP_X_SIZE) {
                return 0;
            }
            int idx = get_voxelmap_idx(tx,ty);
            // skip empty columns
            if(columns_header_data[idx].num_runs == 0) {
                continue;
            }
            for(int tz = z-zsize; tz < z+zsize+1; tz++) {
                if(voxel_is_solid(tx, ty, tz)) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

int find_first_solid_span_gte_z(s32 x, s32 y, s32 z) {    
    int res = -1;
    if(x < 0 || x >= MAP_X_SIZE || y < 0 || y >= MAP_Y_SIZE || z < 0 || z >= (cur_map_max_height+1)) {
        return 1;
    }
    iter_res span_check_func(int top, int exclusive_bot, int span_idx) {
        if(z >= top && z < exclusive_bot) {
            res = span_idx;
            return STOP_ITER;
        }
    }
    for_each_solid_chunk_in_column(x, y, &span_check_func);
    return res;
}

int voxel_is_surface(s32 map_x, s32 map_y, s32 map_z) {
    if(map_x < 0 || map_x >= MAP_X_SIZE || map_y < 0 || map_y > MAP_Y_SIZE || map_z < 0 || map_z >= (cur_map_max_height+1)) {
        return 0;
    }
    // TODO: maybe binary search
    int res = 0;
    iter_res span_check_func(int top, int exclusive_bot, int span_idx, int cumulative_skipped_surface_voxels) {
        if(map_z >= top && map_z < exclusive_bot) {
            res = 1;
            return STOP_ITER;
        }
    }
    for_each_surface_chunk_in_column(map_x, map_y, &span_check_func);
    return res;
}

void set_voxel_to_surface(s32 x, s32 y, s32 z) {
    //printf("TODO: add support for exposing voxels as surfaces\n");
    //assert(0);

#if 0
    // only applies if there is a voxel here :)
    if(x < 0 || x >= MAP_X_SIZE || y < 0 || y >= MAP_Y_SIZE || z < 0 || z > 64) {
        return;
    }
    u32 voxelmap_idx = get_voxelmap_idx(x, y);
    int num_surface_runs = columns_header_data[voxelmap_idx].num_runs;
    int num_solid_runs = columns_header_data[voxelmap_idx].num_runs;

    span* surface_runs = &columns_runs_data[voxelmap_idx].runs_info[0];
    solid_span* solid_runs = &columns_solid_runs_data[voxelmap_idx].runs_info[0];

    // handle the easy cases
    int solid_run_idx = -1;
    for(int i = 0; i < num_solid_runs; i++) {
        if(z >= solid_runs[i].top && z < solid_runs[i].bot) {
            solid_run_idx = i;
            break;
        }
    }

    // case 0: not a solid voxel
    if(solid_run_idx == -1) {
        return;
    }
    solid_span* solid_run = &solid_runs[solid_run_idx];    
    // case 1: is already a surface voxel 
    int top_surf_run_idx = solid_run->top_surface_run;
    int bot_surf_run_idx = solid_run->bot_surface_run;

    if(top_surf_run_idx != -1 && z >= surface_runs[top_surf_run_idx].top && z < surface_runs[top_surf_run_idx].bot) {
        return;
    }

    if(bot_surf_run_idx != -1 && z >= surface_runs[bot_surf_run_idx].top && z < surface_runs[bot_surf_run_idx].bot) {
        return;
    }



    //span* top_surf_span = &surface_runs[top_surf_run_idx];
    //span* bot_surf_span = &surface_runs[bot_surf_run_idx];


    // these are only valid if their corresponding indexes are also valid 
    int runs_after_top_surf = top_surf_run_idx != -1 ? (num_surface_runs-1)-top_surf_run_idx : -1;
    int runs_after_bot_surf = bot_surf_run_idx != -1 ? (num_surface_runs-1)-bot_surf_run_idx : -1;

    // case 0:
    // in between two surfaces that only have a gap of one voxel
    if(top_surf_run_idx != -1 && z == surface_runs[top_surf_run_idx].bot && surface_runs[top_surf_run_idx].bot+1 == surface_runs[bot_surf_run_idx].top) {
        // count how many colors to move
        // move remaining colors down a slot
        // set color of new surface voxel
        // set bottom of top surface to bottom of old bottom surface         
    }

    // case 1:
    // right below the top surface, but above the bottom surface
    if(top_surf_run_idx != -1 && z == surface_runs[top_surf_run_idx].bot && bot_surf_run_idx != -1 && z < surface_runs[bot_surf_run_idx].top) {
        // count how many colors to move
        int colors_to_move = 0;
        u32* color_ptr = &columns_colors_data[voxelmap_idx].colors[0];
        for(int i = 0; i <= top_surf_run_idx; i++) {
            color_ptr += (surface_runs[i].bot - surface_runs[i].top);
        }
        // color_ptr now points past the color data for this surface 
        for(int i = top_surf_run_idx+1; i < num_surface_runs; i++) {
            // top = 0, bot = 1, that's 1
            colors_to_move += (surface_runs[i].bot - surface_runs[i].top);
        }
        // move the remaining colors in this column down a slot 
        memmove(color_ptr+1, color_ptr, sizeof(u32)*colors_to_move);
        // set color of new surface voxel
        color_ptr[0] = color;
        // at this point, increment the top surface
        surface_runs[bot_surf_run_idx].bot++;

        return;
    }

    // case 2:
    // right above the bottom surface, but below the top surface
    //if(z == bot_surf_span->top-1 && z >= top_surf_span->bot) {
    //
    //}
#endif
}


#if 0
void add_sphere(s32 sx, s32 sy, s32 sz, s32 radius) {
    // iterate through the bounding box of the sphere
    // for each column, find the min and max points which are in the spere
    // split the column

    s32 bb_min_x = sx-radius;
    s32 bb_min_y = sy-radius;
    s32 bb_max_x = sx+radius;
    s32 bb_max_y = sx+radius;
    s32 bb_min_z = sz-radius;
    f32 bb_max_z = sz+radius;


    for(int y = bb_min_y; y < bb_max_y; y++) {
        for(int x = bb_min_x; x < bb_max_x; x++) {
            u32 idx = get_voxelmap_idx(x, y);
            column_header* header = &columns_header_data[idx];
            // check if we're within the sphere at z 
            // check if 3d distance from the center of the sphere is less than the radius
            // no z distance
            s32 dx = sx-x;
            s32 dy = sy-y;
            f32 dist = sqrtf((dx*dx)+(dy*dy));
            if(dist > radius) {
                continue;
            }
            // iterate through z to find the min and max y of the sphere in this column
            s32 min_z = SDL_MAX_SINT32;
            s32 max_z = SDL_MIN_SINT32;
            for(int z = bb_min_z; z < bb_max_z; z++) {
                s32 dz = sz-z;
                dist = sqrtf((dx*dx)+(dy*dy)+(dz*dz));
                if(dist > radius) { 
                    continue;
                }
                min_z = z < min_z ? z : min_z;
                max_z = z > max_z ? z : max_z;
            }
            
            // is min_z complete above the column?  skip
            if(min_z > columns_max_y[idx]) {
                continue;
            }

            // is min_z 0 and max_z above the column? clear the column
            if(min_z == 0 && max_z > columns_max_y[idx]) { 
                header->num_runs = 0;
                continue;
            }

            // is min_z above zero, and max_z above the column? shorten the column
            if(min_z > 0 && max_z >= columns_max_y[idx]) {
                columns_max_y[idx] = min_z;
                header->first_three_runs[1] = min_z-1;
                continue;
            }
            // is min_z above zero, and max_z below the column?  split the column
            if(min_z > 0 && max_z < columns_max_y[idx]) {
                // don't change max y voxels
                u32 prev_col_height = header->first_three_runs[1];
                u32 max_y = columns_max_y[idx];
                u32 second_run_start = (max_z+1);
                u32 second_run_len = prev_col_height-second_run_start;
                columns_max_y[idx] = second_run_start+second_run_len+1;
                header->num_runs = 2;
                header->first_three_runs[0] = 0; // zero skip
                header->first_three_runs[1] = min_z-1; // length is from 0 up to min_z-1
                header->first_three_runs[2] = second_run_start-min_z; // then we skip the size of the sphere?
                header->first_three_runs[3] = max(2, second_run_len); // then then length should be 
                header->first_three_run_colors[1] = header->first_three_run_colors[0];
                continue;
            }



        }
    }

}
#endif

u32 count_colors_in_span(span sp) {
    u32 num_colors = ((sp.top_surface_end+1)-sp.top_surface_start);
    if(sp.bot_surface_end > sp.bot_surface_start) {
        num_colors += (sp.bot_surface_end - sp.bot_surface_start);
    }
    return num_colors;
}

void remove_voxel_at(s32 map_x, s32 map_y, s32 map_z) {
    // we either 
    //
    if(map_x < 0 || map_x >= MAP_X_SIZE || map_y < 0 || map_y >= MAP_Y_SIZE) {
        return;
    }
    if(map_z >= cur_map_max_height-1) {
        return;
    }
    clear_bit_in_bitmap(map_x, map_y, map_z);
    u32 idx = get_voxelmap_idx(map_x, map_y);
    
    // tracks color slot
    u32 cumulative_skipped_surface_voxels = 0;
    u32* color_ptr = columns_colors_data[idx].colors;

    u8 num_runs = columns_header_data[idx].num_runs;

    for(int i = 0; i < num_runs; i++) {
        u8 top_of_top_surf = columns_runs_data[idx].runs_info[i].top_surface_start;
        u8 bot_of_top_surf = columns_runs_data[idx].runs_info[i].top_surface_end;
        u8 top_of_bot_surf = columns_runs_data[idx].runs_info[i].bot_surface_start;
        u8 bot_of_bot_surf_exclusive = columns_runs_data[idx].runs_info[i].bot_surface_end;
        if(map_z == top_of_top_surf && top_of_top_surf < bot_of_top_surf) {
            // case 0: top voxel of a top surface
            columns_runs_data[idx].runs_info[i].top_surface_start++;
            if(i == 0) {
                columns_header_data[idx].top_y++;
            }
            set_voxel_to_surface(map_x-1, map_y, map_z);
            set_voxel_to_surface(map_x+1, map_y, map_z);
            set_voxel_to_surface(map_x, map_y-1, map_z);
            set_voxel_to_surface(map_x, map_y+1, map_z);


            // move colors up for this column
            u32 colors_to_move = 0;
            for(int j = i; j < num_runs; j++) {
                colors_to_move += count_colors_in_span(columns_runs_data[idx].runs_info[j]);
            }
            memmove(color_ptr, color_ptr+1, sizeof(u32)*colors_to_move);

            return;
        }

        if(map_z == top_of_top_surf && top_of_top_surf == bot_of_top_surf) {
            // case 1: inside top surf, which has a length of 1
            if(top_of_bot_surf > bot_of_top_surf && bot_of_bot_surf_exclusive > top_of_bot_surf) {

                // there are solid, non-surface voxels to handle..
                // let's just give it the same color
                
                columns_runs_data[idx].runs_info[i].top_surface_start++;
                columns_runs_data[idx].runs_info[i].top_surface_end++;
                
                set_voxel_to_surface(map_x-1, map_y, map_z);
                set_voxel_to_surface(map_x+1, map_y, map_z);
                set_voxel_to_surface(map_x, map_y-1, map_z);
                set_voxel_to_surface(map_x, map_y+1, map_z);
                return;
            } else {
                // there are no solid voxels below 
                // in fact there is no bottom surface either
                // just remove this span, copy colors 1 up, and copy spans up as well
                u32 colors_to_move = 0;
                for(int j = i+1; j < num_runs; j++) {
                    colors_to_move += count_colors_in_span(columns_runs_data[idx].runs_info[j]);
                    columns_runs_data[idx].runs_info[j-1] = columns_runs_data[idx].runs_info[j];
                }
                memmove(color_ptr, color_ptr+1, sizeof(u32)*colors_to_move);
                columns_header_data[idx].num_runs--;
                //if(i == 0) {
                    // the top y is now the top y of the next run
                    // just always update, in case to avoid a branch 
                    columns_header_data[idx].top_y = columns_runs_data[idx].runs_info[0].top_surface_start;
                //}
                
                set_voxel_to_surface(map_x-1, map_y, map_z);
                set_voxel_to_surface(map_x+1, map_y, map_z);
                set_voxel_to_surface(map_x, map_y-1, map_z);
                set_voxel_to_surface(map_x, map_y+1, map_z);
                return;
            }
        }

        
        if(map_z > top_of_top_surf && map_z < bot_of_top_surf) {
            // case 3: middle of top surface

            // patch surface
            // copy spans down
            for(int j = num_runs; j > i; j--) {
                columns_runs_data[idx].runs_info[j] = columns_runs_data[idx].runs_info[j-1];
            }
            num_runs++;
            columns_header_data[idx].num_runs = num_runs;
            // now the current run and next run are identical
            // modify them
            columns_runs_data[idx].runs_info[i].top_surface_end = map_z-1;
            columns_runs_data[idx].runs_info[i].bot_surface_start = map_z;
            columns_runs_data[idx].runs_info[i].bot_surface_end = map_z;

            columns_runs_data[idx].runs_info[i+1].top_surface_start = map_z+1;

            // move colors up for this column
            u32 colors_to_move = 0;
            for(int j = i+1; j < num_runs; j++) {
                colors_to_move += count_colors_in_span(columns_runs_data[idx].runs_info[j]);
            }
            
            // skip past new top
            color_ptr += (columns_runs_data[idx].runs_info[i].top_surface_end+1)-columns_runs_data[idx].runs_info[i].top_surface_start;

            memmove(color_ptr, color_ptr+1, sizeof(u32)*colors_to_move);

            set_voxel_to_surface(map_x-1, map_y, map_z);
            set_voxel_to_surface(map_x+1, map_y, map_z);
            set_voxel_to_surface(map_x, map_y-1, map_z);
            set_voxel_to_surface(map_x, map_y+1, map_z);

            return;
        }
        
        if(map_z == bot_of_top_surf) {
            // case 3: bottom of top surface
            return;
        }

        cumulative_skipped_surface_voxels += ((bot_of_top_surf+1) - top_of_top_surf);
        color_ptr += ((bot_of_top_surf+1) - top_of_top_surf);

        if(map_z == bot_of_bot_surf_exclusive-1 && bot_of_bot_surf_exclusive != bot_of_top_surf && bot_of_bot_surf_exclusive > top_of_bot_surf) {
            // case 1: bottom voxel of the bottom surface
            columns_runs_data[idx].runs_info[i].bot_surface_end--;
            // we need to delete this color
            // this is the bug for sure
            u32 colors_to_move = 0;
            for(int j = i+1; j < num_runs; j++) {
                colors_to_move += count_colors_in_span(columns_runs_data[idx].runs_info[j]);
            }
            u32 colors_in_bottom_block = (bot_of_bot_surf_exclusive-top_of_bot_surf);
            color_ptr += colors_in_bottom_block-1;
            memmove(color_ptr, color_ptr+1, sizeof(u32)*colors_to_move);

            set_voxel_to_surface(map_x-1, map_y, map_z);
            set_voxel_to_surface(map_x+1, map_y, map_z);
            set_voxel_to_surface(map_x, map_y-1, map_z);
            set_voxel_to_surface(map_x, map_y+1, map_z);
            return;
        }

        if(map_z > bot_of_top_surf && map_z < top_of_bot_surf) {
            // split in middle of non-surface

        }
        cumulative_skipped_surface_voxels += (bot_of_bot_surf_exclusive - top_of_bot_surf);
        color_ptr += (bot_of_bot_surf_exclusive - top_of_bot_surf);
    }

    // case 2: middle voxel of top surface
    // split into two runs
    
    // first run is top_top: top_top, top_bot: N-1, bot_top = N-1, bot_bot = N-1
    // colors are the same..

    // second run is top_top: N+1, top_bot: top_bot, bot_top: bot_top, bot_bot
    // delete color N and move remaining colors in column up by one

    // case 3: bottom voxel of top surface
    // split into two runs

    // fiarst run is top
}

u8 got_lighting_info = 0;
u32 convert_voxlap_color_to_transparent_abgr(u32 argb) {        
    u8 a = argb>>24;
    got_lighting_info = (got_lighting_info || a != 255);
    a >>= 2; a <<= 2;
    a |= 0b01;
    //u8 a = 0b11111101;//(argb>>26)<<2;// | 0b11);
    u8 r = (argb>>16)&0xFF;
    u8 g = (argb>>8)&0xFF;
    u8 b = argb&0xFF;
    f32 fr = r; fr *= 0.333f;
    f32 fg = g; fg *= 0.333f;
    f32 fb = b; fb *= 0.333f;
    r = fr;
    g = fg;
    b = fb;
    return (a<<24)|(b<<16)|(g<<8)|r;
}

u32 convert_voxlap_color_to_abgr(u32 argb) {
    if(transparency) { return convert_voxlap_color_to_transparent_abgr(argb); }
    u8 a = argb>>24;
    got_lighting_info = (got_lighting_info || a != 255);
    a >>= 2; a <<= 2;
    a |= 0b11;

    //a = 0b11111111; // the bottom two bits are transparency level
    // normally this isn't used :)
    //u8 a = 0b11111101;//0xFF;//(argb>>26)<<2;// | 0b11);
    //a = 0xFF;//(argb>>26)<<2;// | 0b11);
    u8 r = ((argb>>16)&0xFF);
    u8 g = ((argb>>8)&0xFF);
    u8 b = (argb&0xFF);
    return (a<<24)|(b<<16)|(g<<8)|r;
    //return (0b11<<24)|(a<<24)|(b<<16)|(g<<8)|r;
}

enum map_load_error_type {
    BAD_DIMENSIONS = 1,
    INVALID_MAX_EXPECTED_HEIGHT = 2,
} map_load_error_type;

char* map_load_error_strings[3] = {
    "No error",
    "Bad map xy dimensions",
    "Bad map height"
};

int load_error_value;

void register_unique_color(u32 col) {

}



int load_voxlap_map(char* file, int expected_height, int double_height) {
    got_lighting_info = 0;
    FILE* f = fopen(file, "rb");
    if(f == NULL) {
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);  /* same as rewind(f); */

    if(len > 32768*1024) {
        printf("Map file is too large for 32MB buffer!\n");
        exit(1);
    }

    fread(map_file_buffer, len, 1, f);
    fclose(f);

    u8 *base = (u8*)map_file_buffer;
    int x,y,z;
    printf("Loading map...\n");

    u8 *v = (u8*)map_file_buffer;
    for(y = 511; y >= 0; y--) {
    //for (y=0; y <= 511; y++) {
        int yy = y*2;
        for(x = 0; x <= 511; x++) {
            if(x == (8>>1) && y == (512>>1)) {
                printf("whoa!!\n");
            }
    //    for (x=511; x >= 0; x--) { 
            int xx = x*2;
            z = 0;
            u32 idx = get_voxelmap_idx(x, y);
            column_header *header = &columns_header_data[idx];

            span *runs = &columns_runs_data[idx].runs_info[0];
            u32 *output_color_ptr = &columns_colors_data[idx].colors[0];
            int num_runs = 0;     
            int double_factor = double_height ? 2 : 1;
            // parse this column
            for(;;) {
                u32 *color;
                int i;
                int number_4byte_chunks = v[0];
                int top_color_start = v[1];
                int top_color_end   = v[2]; // inclusive
                int top_color_end_exclusive = top_color_end+1;
                //assert(top_color_end >= top_color_start);
                if(top_color_start > expected_height) { load_error_value = top_color_start; return INVALID_MAX_EXPECTED_HEIGHT; }
                if(top_color_end > expected_height) { load_error_value = top_color_end; return INVALID_MAX_EXPECTED_HEIGHT; }
                int bottom_color_start;
                int bottom_color_end_exclusive; // exclusive
                int len_top;
                int len_bottom;
                
                
                color = (u32 *) (v+4);
                int actual_top_color_end = (top_color_end < top_color_start ? top_color_start : top_color_end);
                for(z=top_color_start; z < top_color_end_exclusive; z++) {
                    u32 col = *color;
                    if(double_height) {
                        *output_color_ptr++ = convert_voxlap_color_to_abgr(col);
                    }
                    *output_color_ptr++ = convert_voxlap_color_to_abgr(col);
                    color++;
                }
                
                len_top = top_color_end - top_color_start + 1;

                // check for end of data marker
                if (number_4byte_chunks == 0) {
                    // infer ACTUAL number of 4-byte chunks from the length of the color data
                    v += 4 * (len_top + 1);
                    runs[num_runs].top_surface_start = top_color_start*double_factor; 
                    runs[num_runs].top_surface_end = (top_color_end_exclusive*double_factor); //(top_color_end_exclusive*double_factor); //(top_color_end_exclusive*double_factor)-1; //expected_height*double_factor; // top_color_end // NOTE: this here is important!
                    runs[num_runs].bot_surface_start = expected_height*double_factor;
                    runs[num_runs++].bot_surface_end = expected_height*double_factor;
                    // fill empty colors
                    u32 prev_color = *(output_color_ptr-1);
                    while(z <= expected_height) {
                        if(double_height) {
                            *output_color_ptr++ = prev_color;
                        }
                        *output_color_ptr++ = prev_color;
                        z++;
                    }
                    //int num_visible_voxels = (runs[num_runs-1].top_surface_end+1)-runs[num_runs-1].top_surface_start;

                    //for(int k = 0; k < num_visible_voxels; k++) {
                    //while(z < runs[num_runs-1].bot_surface_end) {//<= expected_height) { 
                        //if(double_height) {
                        //    *output_color_ptr++ = prev_color;
                        //}
                        //*output_color_ptr++ = prev_color;
                        //z++;
                    //}
                    break;
                }

                // infer the number of bottom colors in next span from chunk length
                len_bottom = (number_4byte_chunks-1) - len_top;

                // now skip the v pointer past the data to the beginning of the next span
                v += v[0]*4;

                bottom_color_end_exclusive   = v[3]; // aka air start
                if(bottom_color_end_exclusive > expected_height) { load_error_value = bottom_color_end_exclusive; return INVALID_MAX_EXPECTED_HEIGHT; }
                bottom_color_start = bottom_color_end_exclusive - len_bottom;

                for(z=bottom_color_start; z < bottom_color_end_exclusive; ++z) {
                    if(double_height) {
                        *output_color_ptr++ = convert_voxlap_color_to_abgr(*color);
                    }
                    *output_color_ptr++ = convert_voxlap_color_to_abgr(*color++);
                }

                runs[num_runs].top_surface_start = top_color_start*double_factor;
                runs[num_runs].top_surface_end = (top_color_end_exclusive*double_factor)-1; //top_color_end; // NOTE: this here is important!
                runs[num_runs].bot_surface_start = bottom_color_start*double_factor;
                runs[num_runs++].bot_surface_end = bottom_color_end_exclusive*double_factor;

            }
            assert(num_runs != 0);
            //if(runs[num_runs-1].top_surface_end != expected_height) {
            //    printf("Expected bottom run height of %i, got %i\n", expected_height, runs[num_runs-1].top_surface_end);
            //    exit(1);
            //}
            //assert(runs[num_runs-1].bot == expected_height+1);
            header->num_runs = num_runs;
            header->top_y = runs[0].top_surface_start;
        }
    }
    if(v-base != len) {
        goto bad_dimensions;
    }

    return 0;
bad_dimensions:
    return BAD_DIMENSIONS;
    

}



/*!
 * \brief A parallel full-adder.
 * \param s0 Bit 0 of the sum
 * \param s1 Bit 1 of the sum
 * \param a0 First argument
 * \param a1 Second argument
 */
#define ADD2(s0,s1,a0,a1) do {		\
	    s1 = (a0) & (a1);		\
	    s0 = (a0) ^ (a1);		\
	} while(0)

/*!
 * \brief A parallel full-adder.
 * \param s0 Bit 0 of the sum
 * \param s1 Bit 1 of the sum
 * \param a0 First argument
 * \param a1 Second argument
 * \param a2 Third argument
 */
#define ADD3(s0,s1,a0,a1,a2) do {	\
	    u64 c0, c1;		\
        u64 tmp;        \
	    ADD2(tmp,c0,a0,a1);		\
	    ADD2(s0,c1,tmp,a2);		\
	    s1 = c0 | c1;		\
	} while(0)

/*
    Add together 2 2-bit numbers, 3 bits but only up to 6 
*/
#define ADD2_2(s0,s1,s2, a00,a01, a10,a11) do {     \
        u64 carry;                                  \
        ADD2(s0, carry, a00, a10);                  \
        ADD3(s1, s2, a01, a11, carry);              \
} while(0);


#define ADD3_3(s0,s1,s2,s3, a00,a01,a02, a10,a11,a12) do {     \
        u64 carry;                                  \
        ADD2(s0, carry, a00, a10);                  \
        ADD3(s1, carry, a01, a11, carry);           \
        ADD3(s2, s3, a02, a12, carry);              \
} while(0);

void double_map() {
    for(int y = 511; y >= 0; y--) {
        int output_y = y*2;
        for(int x = 511; x >= 0; x--) {
            int output_x = x*2;
            int src_idx = get_voxelmap_idx(x,y);
            int dst1_idx = get_voxelmap_idx(output_x,output_y);
            int dst2_idx = get_voxelmap_idx(output_x+1,output_y);
            int dst3_idx = get_voxelmap_idx(output_x,output_y+1);
            int dst4_idx = get_voxelmap_idx(output_x+1,output_y+1);

            columns_header_data[dst1_idx] = columns_header_data[src_idx];
            columns_header_data[dst2_idx] = columns_header_data[src_idx];
            columns_header_data[dst3_idx] = columns_header_data[src_idx];
            columns_header_data[dst4_idx] = columns_header_data[src_idx];
            u32 colors_to_copy = 0;
            for(int j = 0; j < columns_header_data[src_idx].num_runs; j++) {
                colors_to_copy += count_colors_in_span(columns_runs_data[src_idx].runs_info[j]);
                
            }
            memcpy(columns_runs_data[dst1_idx].runs_info, columns_runs_data[src_idx].runs_info, sizeof(span)*columns_header_data[src_idx].num_runs);
            memcpy(columns_runs_data[dst2_idx].runs_info, columns_runs_data[src_idx].runs_info, sizeof(span)*columns_header_data[src_idx].num_runs);
            memcpy(columns_runs_data[dst3_idx].runs_info, columns_runs_data[src_idx].runs_info, sizeof(span)*columns_header_data[src_idx].num_runs);
            memcpy(columns_runs_data[dst4_idx].runs_info, columns_runs_data[src_idx].runs_info, sizeof(span)*columns_header_data[src_idx].num_runs);

            memcpy(columns_colors_data[dst1_idx].colors, columns_colors_data[src_idx].colors, sizeof(u32)*colors_to_copy);
            memcpy(columns_colors_data[dst2_idx].colors, columns_colors_data[src_idx].colors, sizeof(u32)*colors_to_copy);
            memcpy(columns_colors_data[dst3_idx].colors, columns_colors_data[src_idx].colors, sizeof(u32)*colors_to_copy);
            memcpy(columns_colors_data[dst4_idx].colors, columns_colors_data[src_idx].colors, sizeof(u32)*colors_to_copy);
            
            //memcpy(columns_norm_data[dst1_idx].norm_pt1, columns_norm_data[src_idx].norm_pt1, sizeof(f32)*colors_to_copy);
            //memcpy(columns_norm_data[dst2_idx].norm_pt1, columns_norm_data[src_idx].norm_pt1, sizeof(f32)*colors_to_copy);
            //memcpy(columns_norm_data[dst3_idx].norm_pt1, columns_norm_data[src_idx].norm_pt1, sizeof(f32)*colors_to_copy);
            //memcpy(columns_norm_data[dst4_idx].norm_pt1, columns_norm_data[src_idx].norm_pt1, sizeof(f32)*colors_to_copy);
            //memcpy(columns_norm_data[dst1_idx].norm_pt2, columns_norm_data[src_idx].norm_pt2, sizeof(f32)*colors_to_copy);
            //memcpy(columns_norm_data[dst2_idx].norm_pt2, columns_norm_data[src_idx].norm_pt2, sizeof(f32)*colors_to_copy);
            //memcpy(columns_norm_data[dst3_idx].norm_pt2, columns_norm_data[src_idx].norm_pt2, sizeof(f32)*colors_to_copy);
            //memcpy(columns_norm_data[dst4_idx].norm_pt2, columns_norm_data[src_idx].norm_pt2, sizeof(f32)*colors_to_copy);


            columns_bitmaps_data[dst1_idx] = columns_bitmaps_data[src_idx];
            columns_bitmaps_data[dst2_idx] = columns_bitmaps_data[src_idx];
            columns_bitmaps_data[dst3_idx] = columns_bitmaps_data[src_idx];
            columns_bitmaps_data[dst4_idx] = columns_bitmaps_data[src_idx];
        }
    }
}

int light_map_size;


u32 get_voxel_color(s32 x, s32 y, s32 z);

void init_chunks();

void load_map(char* map_filename) {

    if(!map_data_allocated) {
        allocate_map_data();
    }


    eye_height = 10;
    knee_height = 4;
    accel = .68f;
    vel = 0.0f;
    max_accel = 80.0f;
    
    move_forward_speed = .7f;
    strafe_speed = 1.8f;
    fly_speed = 2.2f;

    u32 clear_start = SDL_GetTicks();

    //memset(columns_bitmaps_data, 0, sizeof(column_bitmaps)*MAP_X_SIZE*MAP_Y_SIZE);
    //memset(columns_norm_data, 0, sizeof(column_normals)*MAP_X_SIZE*MAP_Y_SIZE);
    //memset(columns_colors_data, 0, sizeof(column_colors)*MAP_X_SIZE*MAP_Y_SIZE);
    for(int y = light_map_size; y < light_map_size; y++) {
        for(int x = light_map_size; x < light_map_size; x++) {
            u32 cols_to_clear = 0;
            u32 idx = get_voxelmap_idx(x,y);
            for(int j = 0; j < columns_header_data[idx].num_runs; j++) {
                cols_to_clear += count_colors_in_span(columns_runs_data[idx].runs_info[j]);
            }
            memset(columns_colors_data[idx].colors, 0, sizeof(u32)*cols_to_clear);
        }
    }
    //memset(columns_runs_data, 0, sizeof(column_runs)*MAP_X_SIZE*MAP_Y_SIZE);
    memset(columns_header_data, 0, sizeof(column_header)*MAP_X_SIZE*MAP_Y_SIZE);
    u32 clear_end = SDL_GetTicks();
    s32 clear_dt_ms = clear_end-clear_start;

    printf("CLEAR MAP TOOK %ums\n", clear_dt_ms);
    


    u32 load_start = SDL_GetTicks();
    int map_needs_doubling = 1;
    cur_map_max_height = 127;// 63;
    int err = load_voxlap_map(map_filename, 63, map_needs_doubling);
    
    if(err == INVALID_MAX_EXPECTED_HEIGHT) {
        printf("retrying load with 127\n");
        cur_map_max_height = 255; // 127
        err = load_voxlap_map(map_filename, 127, map_needs_doubling);

        if(err == INVALID_MAX_EXPECTED_HEIGHT) {
            //printf("Map is too tall!\n");
            //exit(1);
            //printf("retrying load with 255\n");
            cur_map_max_height = 255;
            map_needs_doubling = 0;
            
            err = load_voxlap_map(map_filename, 255, map_needs_doubling);
        }
    }
    u32 load_end = SDL_GetTicks();
    s32 load_dt_ms = load_end-load_start;
    printf("LOAD MAP TOOK %ums\n", load_dt_ms);
    
    if(map_needs_doubling == 0) {
        eye_height = 5;
        knee_height = 2;
        accel = .38f;
        max_accel = 40.0f;
        move_forward_speed = .35f;
        strafe_speed = .9f;
        fly_speed = 1.1f;
    }
    
    if(err != 0) {
        printf("Error loading map '%s': %s", map_filename, map_load_error_strings[err]);
        if(err == INVALID_MAX_EXPECTED_HEIGHT) { printf(" invalid value: %i", load_error_value); }
        printf("\n");
        exit(1);
    }
    //exit(1);


    u32 bitmap_start = SDL_GetTicks();
    // build bitmap cache (only 33MB, might as well keep it around)
    for(int y = 0; y < 512; y++) {
        for(int x = 0; x < 512; x++) {
            u32 idx = get_voxelmap_idx(x,y);
            col_to_solid_bitmap(idx, &columns_bitmaps_data[idx]);
        }
    }

    


    u32 bitmap_end = SDL_GetTicks();
    u32 bitmap_dt_ms = (bitmap_end - bitmap_start);
    printf("BITMAP CONSTRUCTION TOOK %ums\n", bitmap_dt_ms);

    if(map_needs_doubling) {
        light_map_size = 1024;
        u32 double_start = SDL_GetTicks();
        double_map();
        u32 double_end = SDL_GetTicks();
        u32 double_dt_ms = (double_end - double_start);
        printf("MAP DOUBLING TOOK %ums\n", double_dt_ms);


    } else {
        printf("Can't handle maps of 512 right now!\n");
        exit(1);
        light_map_size = 512;
    }

    light_map_size = 1024;

    u32 chunk_start = SDL_GetTicks();
    init_chunks();
    u32 chunk_end = SDL_GetTicks();
    u32 chunk_dt_ms = (chunk_end - chunk_start);
    printf("MAP CHUNK INIT TOOK %ums\n", chunk_dt_ms);

    

    return;      
}


#endif