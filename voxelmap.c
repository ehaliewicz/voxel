#ifndef VOXELMAP_H
#define VOXELMAP_H

#include "stdio.h"

#include <dirent.h> 
#include "map_table.c"

#include "types.h"


// 4 bytes per run entry

// 4x4 chunk
// currently it'll load half of an 8x8 tile i think?

// 4x4 chunks

// 32 entry cache line

typedef struct {
    u8 top_y;     // duplicate of top in the topmost span of this column
    u8 num_runs;
} column_header;



#define COLUMN_MAX_HEIGHT 128
#define COLUMN_HEIGHT_SHIFT 7
typedef struct {
    u32 colors[256];
} column_colors;

typedef struct {
    f32 norm_pt1[64];
    f32 norm_pt2[64];
} column_normals;

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
typedef struct {
    span runs_info[128];
} column_runs;

int cur_map_max_height; // usually 63 but not always

column_header* columns_header_data;//[1024*1024];
column_colors* columns_colors_data;//[1024*1024];

column_runs* columns_runs_data;//[1024*1024];

column_normals* columns_norm_data;//[1024*1024];


column_header* mip_columns_header_data;//[512*512];
column_colors* mip_columns_colors_data;//[512*512];
column_runs* mip_columns_runs_data;//[512*512];
column_normals* mip_columns_norm_data;//[512*512];

static int map_data_allocated = 0;

void allocate_map_data() {
    columns_header_data = malloc(sizeof(column_header)*1024*1024);
    columns_colors_data = malloc(sizeof(column_colors)*1024*1024);
    columns_runs_data = malloc(sizeof(column_runs)*1024*1024);
    columns_norm_data = malloc(sizeof(column_normals)*1024*1024);

    mip_columns_header_data = malloc(sizeof(column_header)*512*512);
    mip_columns_colors_data = malloc(sizeof(column_colors)*512*512);
    mip_columns_runs_data = malloc(sizeof(column_runs)*512*512);
    mip_columns_norm_data = malloc(sizeof(column_normals)*512*512);
    map_data_allocated = 1;
}

#include "libmorton/morton_BMI.h"

u32 get_voxelmap_idx(s32 x, s32 y) {
    //return m2D_e_BMI(x, y);
    y &= 1023;
    x &= 1023;

    // 2 bits each
    u32 tile_x = x >> 8; // 9 for tiles of 512x512, 8 for 2x2 tiles of 256x256, 7 for tiles of 128x128
    u32 tile_y = y >> 8;

    // 256x256 chunks
    y &= 255; // 255 // 511
    x &= 255; // 255 // 511
    //y &= 1023;
    //x &= 1023;

    // 8x8...
    u16 low_x = x&0b11; // 2 bits
    u16 low_y = y&0b11;  
    u16 high_x = x>>2;   // 6 bits
    u16 high_y = y>>2;
    // 20 bit result
    return (tile_y<<18)|(tile_x<<16)|(high_y<<10)|(high_x<<4)|(low_y<<2)|low_x;
}

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
    //__m256i wrapped_tile_xs = wrapped_xs & 255; // 255 // 511
    //__m256i wrapped_tile_ys = wrapped_ys & 255; // 255 // 511
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

u32 voxel_get_color(s32 x, s32 y, s32 z) {
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

int voxel_is_solid(s32 map_x, s32 map_y, s32 map_z) {
    // TODO: maybe binary search
    if(map_x < 0 || map_x > 511 || map_y < 0 || map_y > 511 || map_z < 0 || map_z >= (cur_map_max_height+1)) {
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

int find_first_solid_span_gte_z(s32 x, s32 y, s32 z) {    
    int res = -1;
    if(x < 0 || x > 511 || y < 0 || y > 511 || z < 0 || z >= (cur_map_max_height+1)) {
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
    if(map_x < 0 || map_x > 511 || map_y < 0 || map_y > 511 || map_z < 0 || map_z >= (cur_map_max_height+1)) {
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

void set_voxel_to_surface(s32 x, s32 y, s32 z, u32 color) {
    printf("unsupported!\n");
    assert(0);

#if 0
    // only applies if there is a voxel here :)
    if(x < 0 || x > 512 || y < 0 || y > 512 || z < 0 || z > 64) {
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

#if 0
void remove_voxel_at(s32 map_x, s32 map_y, s32 map_z) {
    // we either 
    //
    // case 0: top voxel of a top surface
    // case 1: bottom voxel of the bottom surface
    u32 idx = get_voxelmap_idx(map_x, map_y);
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        u8 top_of_top_span_pos = columns_runs_data[idx].runs_info->top;
        u8 bot_of_top_span_pos = columns_runs_data[idx].runs_info->bot;
        u8 bot_of_bot_span_pos = columns_runs_data[idx].runs_info->bottom_voxels_end;
        u8 top_of_bot_span_pos = columns_runs_data[idx].runs_info->bottom_voxels_start;
        if(map_z == top_of_top_span_pos && top_of_top_span_pos < bot_of_top_span_pos) {
            columns_runs_data[idx].runs_info->top++;
            return;
        }

        if(map_z == bot_of_bot_span_pos-1 && bot_of_bot_span_pos != bot_of_top_span_pos) {
            columns_runs_data[idx].runs_info->bottom_voxels_end--;
            return;
        }
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
#endif

u32 convert_voxlap_color_to_transparent_abgr(u32 argb) {
    u8 a = 0b11111101;//(argb>>26)<<2;// | 0b11);
    u8 r = (argb>>16)&0xFF;
    u8 g = (argb>>8)&0xFf;
    u8 b = argb&0xFF;
    return (a<<24)|(b<<16)|(g<<8)|r;
}

u32 convert_voxlap_color_to_abgr(u32 argb) {
    u8 a = 0b11111101;//0xFF;//(argb>>26)<<2;// | 0b11);
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

int load_voxlap_map(char* file, int expected_height) {
    FILE* f = fopen(file, "rb");
    if(f == NULL) {
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);  /* same as rewind(f); */

    char *bytes = malloc(len);
    fread(bytes, len, 1, f);
    fclose(f);

    u8 *v = (u8*)bytes;
    u8 *base = v;
    int x,y,z;
    for (y=511; y >= 0; --y) {
        for (x=511; x >= 0; --x) {
            
            z = 0;
            u32 idx = get_voxelmap_idx(x, y);
            column_header *header = &columns_header_data[idx];

            span *runs = &columns_runs_data[idx].runs_info[0];
            u32 *output_color_ptr = &columns_colors_data[idx].colors[0];
            int num_runs = 0;     

            for(;;) {
                u32 *color;
                int i;
                int number_4byte_chunks = v[0];
                int top_color_start = v[1];
                int top_color_end   = v[2]; // inclusive
                //assert(top_color_end >= top_color_start);
                if(top_color_start > expected_height) { load_error_value = top_color_start; return INVALID_MAX_EXPECTED_HEIGHT; }
                if(top_color_end > expected_height) { load_error_value = top_color_end; return INVALID_MAX_EXPECTED_HEIGHT; }
                int bottom_color_start;
                int bottom_color_end; // exclusive
                int len_top;
                int len_bottom;
                
                
                color = (u32 *) (v+4);
                int actual_top_color_end = (top_color_end < top_color_start ? top_color_start : top_color_end);
                for(z=top_color_start; z <= top_color_end; z++) {
                    *output_color_ptr++ = convert_voxlap_color_to_abgr(*color++);
                }

                len_top = top_color_end - top_color_start + 1;

                // check for end of data marker
                if (number_4byte_chunks == 0) {
                    // infer ACTUAL number of 4-byte chunks from the length of the color data
                    v += 4 * (len_top + 1);
                    runs[num_runs].top_surface_start = top_color_start; 
                    runs[num_runs].top_surface_end = expected_height; // top_color_end // NOTE: this here is important!
                    runs[num_runs].bot_surface_start = expected_height+1;
                    runs[num_runs++].bot_surface_end = expected_height+1;
                    // fill empty colors
                    u32 prev_color = *(output_color_ptr-1);
                    while(z++ <= expected_height) {
                        *output_color_ptr++ = prev_color;
                    }

                    break;
                }

                // infer the number of bottom colors in next span from chunk length
                len_bottom = (number_4byte_chunks-1) - len_top;

                // now skip the v pointer past the data to the beginning of the next span
                v += v[0]*4;

                bottom_color_end   = v[3]; // aka air start
                if(bottom_color_end > expected_height) { load_error_value = bottom_color_end; return INVALID_MAX_EXPECTED_HEIGHT; }
                bottom_color_start = bottom_color_end - len_bottom;

                for(z=bottom_color_start; z < bottom_color_end; ++z) {
                    *output_color_ptr++ = convert_voxlap_color_to_abgr(*color++);
                }

                runs[num_runs].top_surface_start = top_color_start;
                runs[num_runs].top_surface_end = top_color_end; //top_color_end; // NOTE: this here is important!
                runs[num_runs].bot_surface_start = bottom_color_start;
                runs[num_runs++].bot_surface_end = bottom_color_end;

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
        return BAD_DIMENSIONS;
    }

    return 0;

}

u64 column_to_bitmap(int num_runs, span* spans) {
    assert(0);
#if 0 
    u64 bmp = 0;
    for(int i = 0; i < num_runs; i++) {
        bmp |= ((1ull << (spans[i].bot)) - (1ull << spans[i].top));
        bmp |= (1ull << 63);
        //if(spans[i].bot == 64) { bmp |= (1ull<<63); }
        //for(int j = spans[i].top; j < spans[i].bot; j++) {
        //    bmp |= ((u64)1 << j);
        //}
    }
    return bmp;
#endif
}

#define AMBIENT_OCCLUSION_RADIUS 4

#define AMBIENT_OCCLUSION_DIAMETER (AMBIENT_OCCLUSION_RADIUS*2+1)

u64 surface_bitmap_array[1024*1024];
u64 solid_bitmap_array[1024*1024];

//#define NORMAL_RADIUS 6
void light_map(int min_x, int min_y, int max_x, int max_y) {

    for(int y = min_y; y <= max_y; y++) {
        for(int x = min_x; x <= max_x; x++) {
            u32 voxelmap_idx = get_voxelmap_idx(x, y);
            column_header* header = &columns_header_data[voxelmap_idx];
            span* runs = columns_runs_data[voxelmap_idx].runs_info;
           
        #if 1
            for(int i = 0; i < header->num_runs; i++) {
                
                int z_ranges[2][2] = {{runs[i].top_surface_start, runs[i].top_surface_end+1}, 
                                   {runs[i].bot_surface_start, runs[i].bot_surface_end}};
                for(int range_idx = 0; range_idx < 2; range_idx++) {
                    int min_z = z_ranges[range_idx][0];
                    int max_z_exclusive = z_ranges[range_idx][1];

                    for(int z = min_z; z < max_z_exclusive; z++) {
                        //if(x == 126 && y == 134 && z == 231) {
                        //    printf("break!\n");
                        //}
                        //if(z >= (runs[i].top_surface_end+1) && z < runs[i].bot_surface_start) {
                        //    continue;
                        //}


                        //if(!voxel_is_solid(x, y, z)) {
                        //    continue;
                        //}
                        //if(!voxel_is_surface(x, y, z)) {
                        //    continue;
                        //}

                        int solid_cells = 0;
                        int samples = 0;


                            
                        f32 norm_x = 0.0;
                        f32 norm_y = 0.0;
                        f32 norm_z = 0.0;



                    #if 1


                        // NOTE: THIS IS BEING REUSED IN THE FUNCTION POINTER VERSION ABOVE!
                        // faster incremental version, ~9.5 seconds for a radius of 4
                        for(int yy = -AMBIENT_OCCLUSION_RADIUS; yy <= AMBIENT_OCCLUSION_RADIUS; yy++) {
                            int ty = y+yy;
                            if(ty < 0 || ty > 511) { continue; }
                            for(int xx = -AMBIENT_OCCLUSION_RADIUS; xx <= AMBIENT_OCCLUSION_RADIUS; xx++) {
                                int tx = x+xx;
                                if(tx < 0 || tx > 511) { continue; }

                                // don't search below this voxel
                                //int cur_span_idx = find_first_solid_span_gte_z(x, y, z-AMBIENT_OCCLUSION_RADIUS);
                                u32 test_voxelmap_idx = get_voxelmap_idx(tx, ty);
                                span* cur_spans = columns_runs_data[test_voxelmap_idx].runs_info;
                                int test_span_num_runs = columns_header_data[test_voxelmap_idx].num_runs;
                                //if(test_span_num_runs > 2)  {
                                //    printf("whoa!\n"); 
                                //}
                                int cur_span_idx = 0;

                                // find first span that ends on the current test z ?
                                int start_z = (z-AMBIENT_OCCLUSION_RADIUS) < 0 ? 0 : (z-AMBIENT_OCCLUSION_RADIUS);
                                //for(cur_span_idx = 0; cur_span_idx < test_span_num_runs; cur_span_idx++) {
                                    //int tz = z-AMBIENT_OCCLUSION_RADIUS;
                                //    if(cur_spans[cur_span_idx].bot_surface_end > start_z) { break; }
                                    //if(start_z >= cur_spans[cur_span_idx].top_surface_start && start_z < cur_spans[cur_span_idx].bot_surface_end) { break; }
                                    //if(cur_spans[cur_span_idx].bot_surface_end > z-AMBIENT_OCCLUSION_RADIUS) { break; }
                                //}

                                for(int zz = -AMBIENT_OCCLUSION_RADIUS; cur_span_idx < test_span_num_runs && zz <= AMBIENT_OCCLUSION_RADIUS; zz++) { //AMBIENT_OCCLUSION_RADIUS; zz++) {

                                    int tz = z+zz;
                                    if(tz < 0 || tz >= (cur_map_max_height+1)) { continue; }

                                    u8 valid_ao_sample = (tx != x && ty != y && tz != z) && (tz <= z);
                                    
                                    //if(x == 126 && y == 134 && z == 231) {
                                    //    printf("%i,%i,%i valid sample?: %i\n", tx, ty, tz, valid_ao_sample);
                                    //}
                                    samples += (valid_ao_sample ? 1 : 0);
                                    u8 out_of_bounds = (tx < 0 || tx >= 512 || ty < 0 || ty >= 512 || tz < 0 || tz >= (cur_map_max_height+1));
                                    //int has_bot = (cur_spans[cur_span_idx].bot_surface_end > cur_spans[cur_span_idx].bot_surface_start);
                                    //int bot = (has_bot ? cur_spans[cur_span_idx].bot_surface_end : (cur_spans[cur_span_idx].top_surface_end+1));
                                    
                                    //u8 within_top_surface_span = (tz >= cur_spans[cur_span_idx].top_surface_start && tz < (cur_spans[cur_span_idx].top_surface_end+1));
                                    //u8 within_bot_surface_span = (tz >= cur_spans[cur_span_idx].bot_surface_start && tz < cur_spans[cur_span_idx].bot_surface_end);
                                    
                                    u8 cell_is_solid = voxel_is_solid(tx,ty,tz); //tz >= cur_spans[cur_span_idx].top_surface_start && tz < cur_spans[cur_span_idx].bot_surface_end;
                                    //u8 cell_is_solid = voxel_is_solid(tx,ty,tz);
                                    // && voxel_is_solid(tx, ty, tz);
                                    solid_cells += ((valid_ao_sample && cell_is_solid) ? 1 : 0);

                                    norm_x += cell_is_solid ? xx : 0.0;
                                    norm_y += cell_is_solid ? yy : 0.0;
                                    norm_z += cell_is_solid ? zz : 0.0;
                                    
                                    cur_span_idx += (tz >= cur_spans[cur_span_idx].bot_surface_end-1 ? 1 : 0); // is this right?? what was the -1 for -1);
                                }
                            }
                        }
                    #elif 0
                        // slow version, ~10 seconds for radius of 4
                        for(int yy = -AMBIENT_OCCLUSION_RADIUS; yy <= AMBIENT_OCCLUSION_RADIUS; yy++) {
                            int ty = y+yy;
                            for(int xx = -AMBIENT_OCCLUSION_RADIUS; xx <= AMBIENT_OCCLUSION_RADIUS; xx++) {
                                int tx = x+xx;

                                for(int zz = -AMBIENT_OCCLUSION_RADIUS; zz <= AMBIENT_OCCLUSION_RADIUS; zz++) {
                                    int tz = z+zz;
                                    
                                    u8 valid_ao_sample = (tx != x && ty != y && tz != z) && (tz <= z);
                                    samples += (valid_ao_sample ? 1 : 0);
                                    u8 out_of_bounds = (tx < 0 || tx >= 512 || ty < 0 || ty >= 512 || tz < 0 || tz >= 64);
                                    u8 cell_is_solid = (!out_of_bounds) && voxel_is_solid(tx, ty, tz);
                                    solid_cells += ((valid_ao_sample && cell_is_solid) ? 1 : 0);

                                    norm_x += cell_is_solid ? xx : 0.0;
                                    norm_y += cell_is_solid ? yy : 0.0;
                                    norm_z += cell_is_solid ? zz : 0.0;
                                
                                }
                            }
                        }

                    #endif 

                        if(norm_x == 0 && norm_y == 0 && norm_z == 0) {
                            norm_z = -1;
                        }

                        
                        // 0 -> no filled surrounding voxels, 1 -> all filled surrounding voxels
                        // but divide in 2 to reduce the effect, so the effect is more subtle
                        f32 zero_to_one; 
                        if(samples == 0) {
                            zero_to_one = 0;
                        } else {
                            zero_to_one = ((solid_cells*.75)/(f32)samples);
                        }

                        f32 one_to_zero = 1-zero_to_one; // 0-> all filled surrounding voxels, 0.5-> no filled surrounding voxels

                        // each albedo has 6 AO bits, so use up all 6 of them
                        f32 one_to_zero_scaled = 63.0 * one_to_zero; // scale from 0->.5 to 0-63


                        f32 len = magnitude_vector(norm_x, norm_y, norm_z);
                        f32 fnorm_x = norm_x / len;
                        f32 fnorm_y = norm_y / len;
                        f32 fnorm_z = norm_z / len;
                        float2 norm = encode_norm(fnorm_x, fnorm_y, fnorm_z);
                        voxel_set_normal(x, y, z, norm.x, norm.y);

                        //f32 i = ((((fnorm_y * .5) + fnorm_z) * 64.0 + 103.5)/256.0);


                        u8 one_to_zero_ao_bits = ((u8)floorf(one_to_zero_scaled));
                        u32 base_color = voxel_get_color(x, y, z);
                        u8 alpha_bits = (base_color>>24)&0b11;
                        //u8 ao_and_alpha_byte = (one_to_zero_ao_bits<<2) | alpha_bits;
                        long r, g, b;

                        r = min(((base_color & 0xFF)),255);
                        g = min((((base_color >> 8) & 0xFF)),255);
                        b = min((((base_color >> 16) & 0xFF)),255);

                        u8 ao_and_alpha_byte = (one_to_zero_ao_bits<<2) | alpha_bits;
                        voxel_set_color(x, y, z, ((ao_and_alpha_byte<<24)|(b<<16)|(g<<8)|r));

                        //z = (z == (runs[i].top_surface_end)) ? runs[i].bot_surface_start : z;
                    }
                }
            }
    #endif
        }
    }
}

typedef enum {
    AIR,
    TOP_SURF,
    SOLID,
    BOT_SURF
} col_state;


void mip_columns(int x1, int x2, int y1, int y2) {
    u32 colors[256] = {0};
    u32* color_ptr = colors;
    
    span mip_spans[128];
    int num_runs = 0;

    int cur_state = AIR;

    span cur_span = {.top_surface_start = 1, .top_surface_end = 0, .bot_surface_start = 0, .bot_surface_end = 0};
    if(x1 == 258 && y1 == 162) {
        printf("break!");
    }

    for(int z = 0; z < 256; z++) {
        u8 ul_solid = voxel_is_solid(x1,y1,z);
        u8 ur_solid = voxel_is_solid(x2,y1,z);
        u8 dl_solid = voxel_is_solid(x1,y2,z);
        u8 dr_solid = voxel_is_solid(x2,y2,z);
        int is_solid = (ul_solid || ur_solid || dl_solid || dr_solid);
        if(!is_solid && cur_state == AIR) { continue; }
        u8 ul_surface = voxel_is_surface(x1,y1,z);
        u8 ur_surface = voxel_is_surface(x2,y1,z);
        u8 dl_surface = voxel_is_surface(x1,y2,z);
        u8 dr_surface = voxel_is_surface(x2,y2,z);
        u8 is_surface = (ul_surface || ur_surface || dl_surface || dr_surface);

        u32 ul_col;
        u32 ur_col;
        u32 dl_col;
        u32 dr_col;
        int sum_r;
        int sum_g;
        int sum_b;
        u32 avg_col;

        if(is_surface) {
            ul_col = voxel_get_color(x1,y1,z);
            ur_col = voxel_get_color(x2,y1,z);
            dl_col = voxel_get_color(x1,y2,z);
            dr_col = voxel_get_color(x2,y2,z);

            u32 ul_r = ul_col & 0xFF;
            u32 ul_g = (ul_col >> 8) & 0xFF;
            u32 ul_b = (ul_col >> 16) & 0xFF;
            u32 ur_r = ur_col & 0xFF;
            u32 ur_g = (ur_col >> 8) & 0xFF;
            u32 ur_b = (ur_col >> 16) & 0xFF;
            u32 dl_r = dl_col & 0xFF;
            u32 dl_g = (dl_col >> 8) & 0xFF;
            u32 dl_b = (dl_col >> 16) & 0xFF;
            u32 dr_r = dr_col & 0xFF;
            u32 dr_g = (dr_col >> 8) & 0xFF;
            u32 dr_b = (dr_col >> 16) & 0xFF;

            sum_r = ul_r + ur_r + dl_r + dr_r;
            sum_g = ul_g + ur_g + dl_g + dr_g;
            sum_b = ul_b + ur_b + dl_b + dr_b;

            u8 num_cols = 0;
            num_cols += (ul_surface ? 1 : 0);
            num_cols += (ur_surface ? 1 : 0);
            num_cols += (dl_surface ? 1 : 0);
            num_cols += (dr_surface ? 1 : 0);

            if(is_surface) {
                sum_r = (int)(sum_r / (f32)num_cols);
                sum_g = (int)(sum_g / (f32)num_cols);
                sum_b = (int)(sum_b / (f32)num_cols);
            }
            u8 res_r = sum_r >= 255 ? 255 : sum_r;
            u8 res_g = sum_g >= 255 ? 255 : sum_g;
            u8 res_b = sum_b >= 255 ? 255 : sum_b;

            avg_col = (0b11111111<<24)|(res_b<<16)|(res_g<<8)|res_r;
        }




       
        switch(cur_state) {
            case AIR: do {
                if(is_solid) {
                    assert(is_surface);
                    cur_state = TOP_SURF;
                    cur_span.top_surface_start = z;
                    cur_span.top_surface_end = z;
                    *color_ptr++ = avg_col;
                }
            } while(0);
            break;
            case TOP_SURF: do {
                if(!is_surface) {
                    cur_state = SOLID;
                    cur_span.top_surface_end = z-1;
                } else {
                    *color_ptr++ = avg_col;
                    cur_span.top_surface_end = z;
                }
            } while(0);
            break;
            case SOLID: do {
                if(is_surface) {
                    cur_state = BOT_SURF;
                    cur_span.bot_surface_start = z;
                    *color_ptr++ = avg_col;
                } else if(is_solid) {
                    // continue solid portion, no color
                } else {
                    // end span with no bottom surface
                    cur_span.bot_surface_start = cur_span.top_surface_end;
                    cur_span.bot_surface_end = cur_span.top_surface_end;
                    mip_spans[num_runs++] = cur_span;
                    cur_span = ((span){.top_surface_start = 1, .top_surface_end = 0, .bot_surface_start = 0, .bot_surface_end = 0});
                    cur_state = AIR;
                }
            } while(0);
            break;
            case BOT_SURF: do {
                if(is_surface) {
                    // continue bottom surface
                    *color_ptr++ = avg_col;
                    cur_span.bot_surface_end = z+1;
                } else if (is_solid) {
                    // empty bottom surface, output span, and move bottom surface to new span's top surface
                    span next_span; next_span.top_surface_start = cur_span.bot_surface_start;
                    cur_span.bot_surface_start = cur_span.top_surface_end;
                    cur_span.bot_surface_end = cur_span.top_surface_end;
                    next_span.top_surface_end = z-1;
                    mip_spans[num_runs++] = cur_span;
                    cur_span = next_span;
                    cur_state = SOLID;

                } else {
                    // back to air
                    // end current span
                    cur_span.bot_surface_end = z-1;
                    mip_spans[num_runs++] = cur_span;
                    cur_span = ((span){.top_surface_start = 1, .top_surface_end = 0, .bot_surface_start = 0, .bot_surface_end = 0});
                    cur_state = AIR;
                }
            } while(0);
            break;
        }
    }

    int num_colors = color_ptr-colors;
    if(cur_state == TOP_SURF) {
        //cur_span.top_surface_end = 255;
        cur_span.bot_surface_start = 255;
        cur_span.bot_surface_end = 255;
    } else if (cur_state == SOLID) {
        cur_span.bot_surface_start = 255;
        cur_span.bot_surface_end = 255;
    } else if (cur_state == BOT_SURF) {
        int top_len = (cur_span.top_surface_end+1)-cur_span.top_surface_start;
        cur_span.bot_surface_end = cur_span.bot_surface_start + (num_colors-top_len);
    }
    mip_spans[num_runs++] = cur_span;

    //assert(num_runs != 0);
    assert(num_runs < 128);

    int mip_idx = get_voxelmap_idx(x1>>1, y1>>1);

    mip_columns_header_data[mip_idx].top_y = mip_spans[0].top_surface_start;
    mip_columns_header_data[mip_idx].num_runs = num_runs;

    assert(num_colors < 256);
    
    memcpy(&mip_columns_colors_data[mip_idx].colors[0],     colors, sizeof(u32)*num_colors);
    memcpy(&mip_columns_runs_data[mip_idx].runs_info[0], mip_spans,  sizeof(span)*num_runs);


}



thread_pool_function(light_map_wrapper, arg_var)
{
	thread_params* tp = (thread_params*)arg_var;
    light_map(tp->min_x, tp->min_y, tp->max_x, tp->max_y);
	InterlockedIncrement64(&tp->finished);
}



void load_map(s32 map_idx) {  
    if(!map_table_loaded) {
        load_map_table();
    }
    if(!map_data_allocated) {
        allocate_map_data();
    }
    if(num_maps == 0) { printf("Add maps to /maps\n"); exit(1); }
    while(map_idx >= num_maps) { map_idx -= num_maps; }
    char buf[32];
    sprintf(buf, "./maps/%s", &map_name_table[map_idxs[map_idx]]);

    cur_map_max_height = 63;
    int err = load_voxlap_map(buf, 63);
    if(err == INVALID_MAX_EXPECTED_HEIGHT) {
        printf("retrying load with 127\n");
        cur_map_max_height = 127;
        err = load_voxlap_map(buf, 127);
        if(err == INVALID_MAX_EXPECTED_HEIGHT) {
            printf("retrying load with 255\n");
            cur_map_max_height = 255;
            err = load_voxlap_map(buf, 255);
        }
    }
    
    
    if(err != 0) {
        printf("Error loading map '%s': %s", buf, map_load_error_strings[err]);
        if(err == INVALID_MAX_EXPECTED_HEIGHT) { printf(" invalid value: %i", load_error_value); }
        printf("\n");
        exit(1);
    }
    //exit(1);


    thread_params parms[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++) {
        parms[i].finished = 0;
        parms[i].min_x = (i == 0) ? 0 : parms[i-1].max_x; //min_x + (draw_dx*i/RAYCAST_THREADS);
        parms[i].max_x = (i == NUM_THREADS-1) ? 511 : (parms[i].min_x + (511/NUM_THREADS));
        parms[i].min_y = 0;
        parms[i].max_y = 511;
    }
    render_pool rp = {
        .num_jobs = NUM_THREADS,
        .func = light_map_wrapper,
        .raw_func = light_map,
        .parms = parms
    };
    start_pool(pool, &rp);
    wait_for_render_pool_to_finish(&rp);
    
    //profile_block light_map_block;
    //TimeBlock(light_map_block, "light map");
    //light_map(0, 0, 511, 511);
    //EndTimeBlock(light_map_block);

    //for(int x = 0; x < 512; x += 2) {
    //    for(int y = 0; y < 512; y += 2) {
    //        mip_columns(x, x+1, y, y+1);
    //    }
    //}

    return;      
}



#endif