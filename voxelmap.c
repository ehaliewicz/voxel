#ifndef VOXELMAP_H
#define VOXELMAP_H

#include "stdio.h"

#include <dirent.h> 
#include "map_table.c"


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
    u32 colors[64];
} column_colors;

// 4 bytes
typedef struct {
    u8 is_top:1;
    u8 top:7;
    u8 is_transparent;
    u8 bot:7;
} span;


typedef struct {
    span runs_info[128];
} column_runs;
// 256MB of run data... ouch

column_header columns_header_data[1024*1024];
// 256 megs of color data
column_colors columns_colors_data[1024*1024];
column_runs columns_runs_data[1024*1024];


u32 get_voxelmap_idx(s32 x, s32 y) {
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
    return (tile_y<<18)|(tile_x<<16)|(high_y<<10)|(high_x<<4)|(low_y<<2)|low_x;
}

u32 get_world_pos_for_color_slot(u32 map_x, u32 map_y, u32 voxel_slot) {
    // TODO: maybe binary search
    u32 idx = get_voxelmap_idx(map_x, map_y);
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        u8 top_of_top_span_pos = columns_runs_data[idx].runs_info[i].top;
        u8 bot_of_top_span_pos = columns_runs_data[idx].runs_info[i].bot;
        u8 top_surf_len = (bot_of_top_span_pos-top_of_top_span_pos);
        if(voxel_slot < top_surf_len) {
            return top_of_top_span_pos + voxel_slot;
        }
        voxel_slot -= top_surf_len;
    }
}

s32 get_color_slot_for_world_pos(s32 x, s32 y, s32 z) {
    s16 color_slot_base_offset = 0;
    u32 idx = get_voxelmap_idx(x, y);
    span* runs = &columns_runs_data[idx].runs_info[0];
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        if(z >= runs[i].top && z < runs[i].bot) {
            // yeah baby
            return color_slot_base_offset + (z-runs[i].top);
            break;
        }
        color_slot_base_offset += (runs[i].bot - runs[i].top);
    }
    return -1;
}

u32 voxel_get_color(s32 x, s32 y, s32 z) {
    s16 color_slot_base_offset = 0;
    u32 idx = get_voxelmap_idx(x, y);
    span* runs = &columns_runs_data[idx].runs_info[0];
    u32* color_ptr = &columns_colors_data[idx].colors[0];
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        if(z >= runs[i].top && z < runs[i].bot) {
            return color_ptr[color_slot_base_offset + (z-runs[i].top)];
        }
        color_slot_base_offset += (runs[i].bot - runs[i].top);
    }
    return 0;
}

void voxel_set_color(s32 x, s32 y, s32 z, u32 color) {
    s16 color_slot_base_offset = 0;
    u32 idx = get_voxelmap_idx(x, y);
    span* runs = &columns_runs_data[idx].runs_info[0];
    u32* color_ptr = &columns_colors_data[idx].colors[0];
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        if(z >= runs[i].top && z < runs[i].bot) {
            color_ptr[color_slot_base_offset + (z-runs[i].top)] = color;
            break;
        }
        color_slot_base_offset += (runs[i].bot - runs[i].top);
    }
}

int voxel_is_solid(u32 map_x, u32 map_y, u32 map_z) {
    // TODO: maybe binary search
    u32 idx = get_voxelmap_idx(map_x, map_y);
    span* runs = &columns_runs_data[idx].runs_info[0];
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        u8 top_span_pos = runs[i].top;
        u8 bot_span_pos = runs[i].bot;
        if(map_z >= top_span_pos && map_z < bot_span_pos) {
            return 1;
        }
    }
    return 0;
}

int voxel_is_surface(u32 map_x, u32 map_y, u32 map_z) {
    // TODO: maybe binary search
    u32 idx = get_voxelmap_idx(map_x, map_y);
    span* runs = &columns_runs_data[idx].runs_info[0];
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        u8 top_of_top_span_pos = runs[i].top;
        u8 bot_of_top_span_pos = runs[i].bot;
        if(map_z >= top_of_top_span_pos && map_z < bot_of_top_span_pos) {
            return 1;
        }
    }
    return 0;
}

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


// xxxxxxxxxx
// yyyyyyyyyy

//           yy xx yyyyy xxxxx yyy xxx
__m256i get_voxelmap_idx_256(__m256i xs, __m256i ys) {
    __m256i ten_twenty_three_vec = _mm256_set1_epi32(1023);
    __m256i two_fifty_five_vec = _mm256_set1_epi32(255);
    __m256i low_three_bits_vec = _mm256_set1_epi32(0b111);
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
    __m256i tile_low_xs = _mm256_and_si256(wrapped_tile_xs, low_three_bits_vec); //wrapped_tile_xs & 0b111;
    __m256i tile_low_ys = _mm256_and_si256(wrapped_tile_ys, low_three_bits_vec); //wrapped_tile_ys & 0b111;
    __m256i tile_high_xs = _mm256_srli_epi32(wrapped_tile_xs, 3);
    __m256i tile_high_ys = _mm256_srli_epi32(wrapped_tile_ys, 3);
    return (tile_ys<<18)|(tile_xs<<16)|(tile_high_ys<<11)|(tile_high_xs<<6)|(tile_low_ys<<3)|tile_low_xs;
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


typedef enum {
    AIR,
    SURFACE,
    INSIDE,
} col_state;

u32 convert_voxlap_color_to_transparent_abgr(u32 argb) {
    u8 a = (argb>>24);// | 0b11);
    u8 r = (argb>>16)&0xFF;
    u8 g = (argb>>8)&0xFf;
    u8 b = argb&0xFF;
    return (0b01<<24)|(a<<24)|(b<<16)|(g<<8)|r;
}

u32 convert_voxlap_color_to_abgr(u32 argb) {
    u8 a = (argb>>24);// | 0b11);
    u8 r = (argb>>16)&0xFF;
    u8 g = (argb>>8)&0xFf;
    u8 b = argb&0xFF;
    return (0b11<<24)|(a<<24)|(b<<16)|(g<<8)|r;
}

int load_voxlap_map(char* file) {
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
    for (y=0; y < 512; ++y) {
        for (x=0; x < 512; ++x) {
            
            z = 0;
            u32 idx = get_voxelmap_idx(x, 512-y);
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
                int bottom_color_start;
                int bottom_color_end; // exclusive
                int len_top;
                int len_bottom;
                
                
                color = (u32 *) (v+4);
                for(z=top_color_start; z <= top_color_end; z++) {
                    *output_color_ptr++ = convert_voxlap_color_to_abgr(*color++);
                }

                len_top = top_color_end - top_color_start + 1;

                // check for end of data marker
                if (number_4byte_chunks == 0) {
                    // infer ACTUAL number of 4-byte chunks from the length of the color data
                    v += 4 * (len_top + 1);
                    runs[num_runs].top = top_color_start;
                    runs[num_runs].bot = 64; // NOTE: this here is important!
                    runs[num_runs].is_transparent = 0;
                    runs[num_runs++].is_top = 1;
                    break;
                }

                // infer the number of bottom colors in next span from chunk length
                len_bottom = (number_4byte_chunks-1) - len_top;

                // now skip the v pointer past the data to the beginning of the next span
                v += v[0]*4;

                bottom_color_end   = v[3]; // aka air start
                bottom_color_start = bottom_color_end - len_bottom;

                for(z=bottom_color_start; z < bottom_color_end; ++z) {
                    *output_color_ptr++ = convert_voxlap_color_to_abgr(*color++);
                }

                runs[num_runs].top = top_color_start;
                runs[num_runs].bot = top_color_end+1; // NOTE: this here is important!
                runs[num_runs].is_transparent = 0;
                runs[num_runs++].is_top = 1;
                runs[num_runs].top = bottom_color_start;
                runs[num_runs].bot = bottom_color_end;
                runs[num_runs].is_transparent = 0;
                runs[num_runs++].is_top = 0;
            }
            assert(runs[num_runs-1].bot == 64);
            header->num_runs = num_runs;
            header->top_y = runs[0].top;
        }
   }
    assert(v-base == len);
    return 0;

}


#define AMBIENT_OCCLUSION_RADIUS 4
void light_map() {

    for(int y = 0; y < 512; y++) {
        for(int x = 0; x < 512; x++) {
            for(int z = 0; z < 64; z++) {
                if(!voxel_is_solid(x, y, z)) {
                    continue;
                }
                if(!voxel_is_surface(x, y, z)) {
                    continue;
                }

                
                int fy = y-1;
                int by = y+1;
                int lx = x-1;
                int rx = x+1;
                int uz = z-1;
                int dz = z+1;

                int solid_cells = 0;
                int samples = 0;



                // TODO: handle this incrementally per column
                // can be much much much more efficient
                for(int ty = y-AMBIENT_OCCLUSION_RADIUS; ty <= y+AMBIENT_OCCLUSION_RADIUS; ty++) {
                    for(int tx = x-AMBIENT_OCCLUSION_RADIUS; tx <= x+AMBIENT_OCCLUSION_RADIUS; tx++) {
                        // don't search below this voxel
                        for(int tz = z-AMBIENT_OCCLUSION_RADIUS; tz < z+1; tz++) {
                            if(tx == x && ty == y && tz == z) {
                                continue;
                            }
                            samples++;
                            if(tx < 0 || tx >= 512 || ty < 0 || ty >= 512 || tz < 0 || tz >= 64) {
                                continue;
                            }
                            solid_cells += voxel_is_solid(tx, ty, tz);
                        }
                    }
                }

                if(solid_cells>0 ) {
                    //printf("whoa!\n");
                }
                
                // 0 -> no filled surrounding voxels, 1 -> all filled surrounding voxels
                // but divide in 2 to reduce the effect, so the effect is more subtle
                f32 zero_to_one  = (solid_cells/(f32)samples)/2;

                f32 one_to_zero = 1-zero_to_one; // 0-> all filled surrounding voxels, 0.5-> no filled surrounding voxels

                // each albedo has 6 AO bits, so use up all 6 of them
                f32 one_to_zero_scaled = 2 * 63.0 * one_to_zero; // scale from 0->.5 to 0-63

                u8 one_to_zero_ao_bits = ((u8)floorf(one_to_zero_scaled));

                u32 base_color = voxel_get_color(x, y, z);
                u8 alpha_bits = (base_color>>24)&0b11;
                u8 ao_and_alpha_byte = (one_to_zero_ao_bits<<2) | alpha_bits;
                u8 r = (base_color & 0xFF);//*one_to_zero;
                u8 g = ((base_color >> 8) & 0xFF);//*one_to_zero;
                u8 b = ((base_color >> 16) & 0xFF);//*one_to_zero;
                voxel_set_color(x, y, z, ((ao_and_alpha_byte<<24)|(b<<16)|(g<<8)|r));
            }
        }
    }
}


static int map_table_loaded = 0;

void load_map(s32 map_idx) {  
    if(!map_table_loaded) {
        load_map_table();

        map_table_loaded = 1;
    }
    if(num_maps == 0) { printf("Add maps to /maps\n"); exit(1); }
    while(map_idx >= num_maps) { map_idx -= num_maps; }
    char buf[32];
    sprintf(buf, "./maps/%s", &map_name_table[map_idxs[map_idx]]);
    if(load_voxlap_map(buf) != 0) {
        printf("Error loading map '%s'\n", buf);
        exit(1);
    }
    light_map();
    return;      
}



#endif