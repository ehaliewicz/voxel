#ifndef VOXELMAP_H
#define VOXELMAP_H



// 4 bytes per run entry
typedef struct {
    u8 max_y;    // duplicate of top_voxels_start in the topmost span of this column
    u8 num_runs; 
} column_header;



// 2MB of header data
//column_header columns_header_data[1024*1024]

//u8 columns_max_y[1024*1024];


#define COLUMN_MAX_HEIGHT 128
#define COLUMN_HEIGHT_SHIFT 7
typedef struct {
    u32 colors[64];
} column_colors;

// 4 bytes
typedef struct {
    // we already have the top defined by the max_y in the header
    // so we can just say start of surface voxels
    // and air/skip voxels length are defined by surface_voxels_start - max_y;
    u8 top_voxels_start;
    u8 top_voxels_end;
    // surface voxels length is implicit by the distance between surface_voxels_end and max_y

    u8 bottom_voxels_start; // ignore center voxels for rendering
    u8 bottom_voxels_end;
} span;


typedef struct {
    span runs_info[64];
} column_runs;
// 256MB of run data... ouch

f32 normal_pt1_data[1024*1024];
f32 normal_pt2_data[1024*1024];
column_header columns_header_data[1024*1024];
// 256 megs of color data
column_colors columns_colors_data[1024*1024];
column_runs columns_runs_data[1024*1024];


#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize2.h"

u8 scratch_colormap[1024*1024*4];
u32 scratch_depthmap[1024*1024*4];

void load_colormap(char *color_filename) {
    int x,y,n;
    unsigned char* data = stbi_load(color_filename, &x, &y, &n, 0);
    for(int y = 0; y < 1024; y++) {
        int y_offset = y*1024*4;
        int iy_offset = y*1024*3;
        for(int x = 0; x < 1024; x++) {
            int ix_offset = x * 3;
            int x_offset = x * 4;
            // we got R G B A, but we want A G B R?
            scratch_colormap[y_offset+x_offset] = data[iy_offset+ix_offset];
            scratch_colormap[y_offset+x_offset+1] = data[iy_offset+ix_offset+1];
            scratch_colormap[y_offset+x_offset+2] = data[iy_offset+ix_offset+2];
            scratch_colormap[y_offset+x_offset+3] = 0xFF;
        }
    }
    stbi_image_free(data);
}

void load_depthmap(char* depth_filename) {
    int x,y,n;
    unsigned char* data = stbi_load(depth_filename, &x, &y, &n, 1);
    for(int y = 0; y < 1024; y++) {
        for(int x = 0; x < 1024; x++) {
            scratch_depthmap[y*1024+x] = data[y*1024+x];
        }
    }
}

void load_and_interpolate_depthmap(char* depth_filename) {
    int x,y,n;
    printf("Filtering heightmap: %i\n", filter_heightmap);
    unsigned char* data = stbi_load(depth_filename, &x, &y, &n, 1);
    for(int iy = 0; iy < 512; iy++) {
        int uiy = (iy-1)&511;
        int diy = (iy+1)&511;
        int oy = iy*2;
        for(int ix = 0; ix < 512; ix++) {
            int lix = (ix-1)&511;
            int rix = (ix+1)&511;
            int ox = ix*2;

            u8 sample = data[iy*512+ix];
            if(1) { //filter_heightmap) {
                u8 up_left_sample = data[uiy*512+lix];
                u8 up_sample = data[uiy*512+ix];
                u8 up_right_sample = data[uiy*512+rix];

                u8 left_sample = data[iy*512+lix];
                u8 right_sample = data[iy*512+rix];

                u8 down_left_sample = data[diy*512+lix];
                u8 down_sample = data[diy*512+ix];
                u8 down_right_sample = data[diy*512+rix];

                u8 ul_out = (int)((up_left_sample+up_sample+left_sample+sample)/4.0);
                u8 ur_out = (int)((up_sample+up_right_sample+sample+right_sample)/4.0);
                u8 dl_out = (int)((left_sample+sample+down_left_sample+down_sample)/4.0);
                u8 dr_out = (int)((sample+right_sample+down_sample+down_right_sample)/4.0);

                scratch_depthmap[oy*1024+ox] = ul_out;
                scratch_depthmap[oy*1024+ox+1] = ur_out;
                scratch_depthmap[(oy+1)*1024+ox] = dl_out;
                scratch_depthmap[(oy+1)*1024+ox+1] = dr_out;
            } else {
                scratch_depthmap[oy*1024+ox] = sample;
                scratch_depthmap[oy*1024+ox+1] = sample;
                scratch_depthmap[(oy+1)*1024+ox] = sample;
                scratch_depthmap[(oy+1)*1024+ox+1] = sample;
            }

        }
    }

}



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
    u16 low_x = x&0b111;
    u16 low_y = y&0b111;
    u16 high_x = x>>3;
    u16 high_y = y>>3;
    return (tile_y<<18)|(tile_x<<16)|(high_y<<11)|(high_x<<6)|(low_y<<3)|low_x;
}

u32 get_world_pos_for_color_slot(u32 map_x, u32 map_y, u32 voxel_slot) {
    // TODO: maybe binary search
    u32 idx = get_voxelmap_idx(map_x, map_y);
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        u8 top_of_top_span_pos = columns_runs_data[idx].runs_info->top_voxels_start;
        u8 bot_of_top_span_pos = columns_runs_data[idx].runs_info->top_voxels_end;
        u8 top_surf_len = (bot_of_top_span_pos-top_of_top_span_pos)+1;
        if(voxel_slot < top_surf_len) {
            return top_of_top_span_pos + voxel_slot;
        }
        voxel_slot -= top_surf_len;

        u8 top_of_bot_span_pos = columns_runs_data[idx].runs_info->bottom_voxels_start;
        u8 bot_of_bot_span_pos = columns_runs_data[idx].runs_info->bottom_voxels_end;
        u8 bot_surf_len = (bot_of_top_span_pos-top_of_top_span_pos);
        if(voxel_slot < bot_surf_len) {
            return top_of_bot_span_pos + voxel_slot;
        }
        voxel_slot -= bot_surf_len;
    }
}

s32 get_color_slot_for_world_pos(s32 x, s32 y, s32 z) {
    s16 color_slot_base_offset = 0;
    u32 idx = get_voxelmap_idx(x, y);
    span* runs = &columns_runs_data[idx].runs_info[0];
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        if(z >= runs[i].top_voxels_start && z < runs[i].top_voxels_end) {
            // yeah baby
            return color_slot_base_offset + (z-runs[i].top_voxels_start);
            break;
        }
        color_slot_base_offset += (runs[i].top_voxels_end - runs[i].top_voxels_start);
        if(z >= runs[i].bottom_voxels_start && z < runs[i].bottom_voxels_end) {
            return color_slot_base_offset + (z-runs[i].bottom_voxels_start);
            break;
        }
        color_slot_base_offset += (runs[i].top_voxels_end - runs[i].top_voxels_start);
    }
    return -1;
}

u32 voxel_get_color(s32 x, s32 y, s32 z) {
    s16 color_slot_base_offset = 0;
    u32 idx = get_voxelmap_idx(x, y);
    span* runs = &columns_runs_data[idx].runs_info[0];
    u32* color_ptr = &columns_colors_data[idx].colors[0];
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        if(z >= runs[i].top_voxels_start && z < runs[i].top_voxels_end) {
            return color_ptr[color_slot_base_offset + (z-runs[i].top_voxels_start)];
        }
        color_slot_base_offset += (runs[i].top_voxels_end - runs[i].top_voxels_start);
        if(z >= runs[i].bottom_voxels_start && z < runs[i].bottom_voxels_end) {
            return color_ptr[color_slot_base_offset + (z-runs[i].bottom_voxels_start)];
        }
        color_slot_base_offset += (runs[i].top_voxels_end - runs[i].top_voxels_start);
    }
    return 0;
}

void voxel_set_color(s32 x, s32 y, s32 z, u32 color) {
    s16 color_slot_base_offset = 0;
    u32 idx = get_voxelmap_idx(x, y);
    span* runs = &columns_runs_data[idx].runs_info[0];
    u32* color_ptr = &columns_colors_data[idx].colors[0];
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        if(z >= runs[i].top_voxels_start && z < runs[i].top_voxels_end) {
            color_ptr[color_slot_base_offset + (z-runs[i].top_voxels_start)] = color;
            break;
        }
        color_slot_base_offset += (runs[i].top_voxels_end - runs[i].top_voxels_start);
        if(z >= runs[i].bottom_voxels_start && z < runs[i].bottom_voxels_end) {
            color_ptr[color_slot_base_offset + (z-runs[i].bottom_voxels_start)] = color;
            break;
        }
        color_slot_base_offset += (runs[i].top_voxels_end - runs[i].top_voxels_start);
    }
}

int voxel_is_solid(u32 map_x, u32 map_y, u32 map_z) {
    // TODO: maybe binary search
    u32 idx = get_voxelmap_idx(map_x, map_y);
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        u8 top_of_top_span_pos = columns_runs_data[idx].runs_info->top_voxels_start;
        u8 bot_of_bot_span_pos = columns_runs_data[idx].runs_info->bottom_voxels_end;
        if(map_z >= top_of_top_span_pos && map_z < bot_of_bot_span_pos) {
            return 1;
        }
    }
    return 0;
}

int voxel_is_surface(u32 map_x, u32 map_y, u32 map_z) {
    // TODO: maybe binary search
    u32 idx = get_voxelmap_idx(map_x, map_y);
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        u8 top_of_top_span_pos = columns_runs_data[idx].runs_info->top_voxels_start;
        u8 bot_of_top_span_pos = columns_runs_data[idx].runs_info->top_voxels_end;
        u8 top_of_bot_span_pos = columns_runs_data[idx].runs_info->bottom_voxels_start;
        u8 bot_of_bot_span_pos = columns_runs_data[idx].runs_info->bottom_voxels_end;
        if(map_z >= top_of_top_span_pos && map_z < bot_of_top_span_pos) {
            return 1;
        }
        if(map_z >= top_of_bot_span_pos && map_z < bot_of_bot_span_pos) {
            return 1;
        }
    }
    return 0;
}

void remove_voxel_at(s32 map_x, s32 map_y, s32 map_z) {
    // we either 
    //
    // case 0: top voxel of a top surface
    // case 1: bottom voxel of the bottom surface
    u32 idx = get_voxelmap_idx(map_x, map_y);
    for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
        u8 top_of_top_span_pos = columns_runs_data[idx].runs_info->top_voxels_start;
        u8 bot_of_top_span_pos = columns_runs_data[idx].runs_info->top_voxels_end;
        u8 bot_of_bot_span_pos = columns_runs_data[idx].runs_info->bottom_voxels_end;
        u8 top_of_bot_span_pos = columns_runs_data[idx].runs_info->bottom_voxels_start;
        if(map_z == top_of_top_span_pos && top_of_top_span_pos < bot_of_top_span_pos) {
            columns_runs_data[idx].runs_info->top_voxels_start++;
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

int is_surface(u32 x, u32 y, u32 z) {
    s32 uy = (y-1);
    s32 dy = (y+1);
    s32 lx = (x-1);
    s32 rx = (x+1);
    
    return (//uy < 0 || dy > 1023 || lx < 0 || rx > 1023 ||
            scratch_depthmap[get_swizzled_map_idx(lx,uy)] < z ||
            scratch_depthmap[get_swizzled_map_idx(x,uy)] < z ||
            scratch_depthmap[get_swizzled_map_idx(rx,uy)] < z ||
            scratch_depthmap[get_swizzled_map_idx(lx,y)] < z ||
            scratch_depthmap[get_swizzled_map_idx(rx,y)] < z ||
            scratch_depthmap[get_swizzled_map_idx(lx,dy)] < z ||
            scratch_depthmap[get_swizzled_map_idx(x,dy)] < z ||
            scratch_depthmap[get_swizzled_map_idx(rx,dy)] < z);
}

void initialize_voxelmap_run_entry(u32 x, u32 y, u32 single_height_val, u32 color) {
    u32 internal_idx = get_voxelmap_idx(x, y);
    if(single_height_val == 0) { single_height_val = 1; }

    column_header* header = &columns_header_data[internal_idx];
    //u16* cptr = columns_colors_data[internal_idx].colors;
    u32* cptr = columns_colors_data[internal_idx].colors;
    //u16 color = u32abgr_to_u16(color_val);

    column_runs* column_run_data = &columns_runs_data[internal_idx];

    u8 top_of_column = 255 - single_height_val;

    header->max_y = top_of_column;
    header->num_runs = 1;
    
    u16 bot_of_top_surface_voxels = top_of_column;
    if(x == 0 || x == 1023 || y == 0 || y == 1023) {
        bot_of_top_surface_voxels = 254;
    } else {
        while(bot_of_top_surface_voxels<254 && is_surface(x, y, 255-bot_of_top_surface_voxels)) {
            bot_of_top_surface_voxels++;
        }

        bot_of_top_surface_voxels--;
        if(bot_of_top_surface_voxels < top_of_column) {
            bot_of_top_surface_voxels++;
        }
        // bot_of_top_surface_voxels is now the one past the bot surface, so decrement it
    }

    column_run_data->runs_info[0].top_voxels_start = top_of_column;
    column_run_data->runs_info[0].top_voxels_end = bot_of_top_surface_voxels;


    for(int i = top_of_column; i <= bot_of_top_surface_voxels; i++) {
        *cptr++ = color;
    }

    //if(x != 0 && y != 0) {
    //    printf("whoa!\n");
    //}
    if(bot_of_top_surface_voxels < 254) {
        column_run_data->runs_info[0].bottom_voxels_start = 254;
        column_run_data->runs_info[0].bottom_voxels_end = 255;

    //u16* cptr = column_color_data->colors;

        *cptr++ = color;
    } else {
        column_run_data->runs_info[0].bottom_voxels_start = 255;
        column_run_data->runs_info[0].bottom_voxels_end = 255;
    }
    //for(int i = bottom_voxels_start+1; i <= bottom_voxels_end; i++) {
    //    *cptr++ = color;
    //}
}   


typedef struct {
    f32 x,y,z;
} norm;

norm get_norm_for_point(int x, int y) {
    u32 center_depth = scratch_depthmap[get_swizzled_map_idx(x,y)];
    f32 min_angle_for_point_right = 0;
    f32 max_angle_for_point_left = PI; // 180 degrees in radians

    u32 left_depth = scratch_depthmap[get_swizzled_map_idx(x-1,y)];
    u32 right_depth = scratch_depthmap[get_swizzled_map_idx(x+1,y)];
    u32 up_depth = scratch_depthmap[get_swizzled_map_idx(x,y-1)];
    u32 down_depth = scratch_depthmap[get_swizzled_map_idx(x,y+1)];

    s32 SurfaceVectorX = 2*(right_depth-left_depth);
    s32 SurfaceVectorY = -4;
    s32 SurfaceVectorZ = 2*(down_depth-up_depth);
    f32 magnitude_surf_vector = magnitude_vector(SurfaceVectorX, SurfaceVectorY, SurfaceVectorZ);
    f32 SurfaceVectorXF = SurfaceVectorX / magnitude_surf_vector;
    f32 SurfaceVectorYF = SurfaceVectorY / magnitude_surf_vector;
    f32 SurfaceVectorZF = SurfaceVectorZ / magnitude_surf_vector;
    return (norm){.x = SurfaceVectorXF, .y = SurfaceVectorYF, .z = SurfaceVectorZF};
}

typedef struct {
    f32 x, y;
} float2;

typedef struct {
    f32 x, y, z;
} float3;

float2 encode_norm(f32 x, f32 y, f32 z) {
    f32 scale = 1.0 / (fabs(x)+fabs(y)+fabs(z));
    f32 tx = x * scale;
    f32 ty = y * scale;
    return (float2){.x = tx+ty, .y=tx-ty};

}

float3 decode_norm(float2 xy) {
    f32 tx = xy.x + xy.y;
    f32 ty = xy.x - xy.y;
    
    f32 rx = tx;
    f32 ry = ty;
    f32 rz = 2.0 - fabs(tx) - fabs(ty);            
    f32 mag = magnitude_vector(rx, ry, rz);
    rx /= mag;
    ry /= mag;
    rz /= mag;
    return (float3){.x=rx, .y=ry, .z=rz};
}


static void calculate_normals(f32* normal_x_tbl, f32* normal_y_tbl) {
    for(int y = 0; y < 1024; y++) {
        for(int x = 0; x < 1024; x++) {
            norm n = get_norm_for_point(x, y);
            //norm up = get_norm_for_point(x, y-1);
            //norm left = get_norm_for_point(x-1, y);
            //norm right = get_norm_for_point(x+1, y);
            //norm down = get_norm_for_point(x, y+1);

            
            //f32 nx = (n.x/2) + (up.x/8) + (left.x/8) + (right.x/8) + (down.x/8);
            //f32 ny = (n.y/2) + (up.y/8) + (left.y/8) + (right.y/8) + (down.y/8);
            //f32 nz = (n.z/2) + (up.z/8) + (left.z/8) + (right.z/8) + (down.z/8);
            
            //f32 nx = (n.x/5) + (up.x/5) + (left.x/5) + (right.x/5) + (down.x/5);
            //f32 ny = (n.y/5) + (up.y/5) + (left.y/5) + (right.y/5) + (down.y/5);
            //f32 nz = (n.z/5) + (up.z/5) + (left.z/5) + (right.z/5) + (down.z/5);
            
            f32 nx = 0;
            f32 ny = 0;
            f32 nz = 0;
            for(int xx = x-2; xx < x+2; xx++) {
                for(int yy = y-2; yy < y+2; yy++) {
                    norm sn = get_norm_for_point(xx, yy);
                    nx += sn.x/16.0;
                    ny += sn.y/16.0;
                    nz += sn.z/16.0;
                }
            }

            f32 magnitude_surf_vector = magnitude_vector(nx, ny, nz);
            nx /= magnitude_surf_vector;
            ny /= magnitude_surf_vector;
            nz /= magnitude_surf_vector;

            //nx /= fabs(nx)+fabs(ny)+fabs(nz);
            //ny /= fabs(nx)+fabs(ny)+fabs(nz);
            //nz /= fabs(nx)+fabs(ny)+fabs(nz);

            //nx = (nz > 0.0) ? x : octwrap_x(nx,ny,nz);
            //ny = (nz > 0.0) ? y : octwrap_y(nx,ny,nz);
            //nx = nx * 0.5 + 0.5;
            //ny = ny * 0.5 + 0.5;





            u32 map_idx = get_swizzled_map_idx(x,y);
            if(1) { //blend_normals) {
                float2 encoded = encode_norm(nx, ny, nz);

                //depthmap_normal_xs[get_swizzled_map_idx(x,y)] = nx;//n.x;
                //depthmap_normal_ys[get_swizzled_map_idx(x,y)] = ny;//n.y;
                //depthmap_normal_zs[get_swizzled_map_idx(x,y)] = nz;//n.z;
                normal_x_tbl[map_idx] = encoded.x;
                normal_y_tbl[map_idx] = encoded.y;
            } else {
                float2 encoded = encode_norm(n.x, n.y, n.z);
                normal_x_tbl[map_idx] = encoded.x;
                normal_y_tbl[map_idx] = encoded.y;
            //    depthmap_normal_xs[get_swizzled_map_idx(x,y)] = n.x;
            //    depthmap_normal_ys[get_swizzled_map_idx(x,y)] = n.y;
            //    depthmap_normal_zs[get_swizzled_map_idx(x,y)] = n.z;
            }
        }
    }
}




/*
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
*/

#include "libvxl.h"
#include "libvxl.c"
#include "stdio.h"

typedef enum {
    AIR,
    SURFACE,
    INSIDE,
} col_state;

u32 ARGB_to_ABGR(u32 argb) {
    u8 a = argb>>24;
    u8 r = (argb>>16)&0xFF;
    u8 g = (argb>>8)&0xFf;
    u8 b = argb&0xFF;
    return (a<<24)|(b<<16)|(g<<8)|r;
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

                //for(i=z; i < top_color_start; i++)
                    //setgeom(x,y,i,0);

                
                
                color = (u32 *) (v+4);
                for(z=top_color_start; z <= top_color_end; z++) {
                    *output_color_ptr++ = ARGB_to_ABGR(*color++);
                }

                len_top = top_color_end - top_color_start + 1;

                // check for end of data marker
                if (number_4byte_chunks == 0) {
                    // infer ACTUAL number of 4-byte chunks from the length of the color data
                    v += 4 * (len_top + 1);
                    runs[num_runs].top_voxels_start = top_color_start;
                    runs[num_runs].top_voxels_end = top_color_end+1; // NOTE: this here is important!
                    runs[num_runs].bottom_voxels_start = top_color_end+1;
                    runs[num_runs++].bottom_voxels_end = 63;
                    break;
                }

                // infer the number of bottom colors in next span from chunk length
                len_bottom = (number_4byte_chunks-1) - len_top;

                // now skip the v pointer past the data to the beginning of the next span
                v += v[0]*4;

                bottom_color_end   = v[3]; // aka air start
                bottom_color_start = bottom_color_end - len_bottom;

                for(z=bottom_color_start; z < bottom_color_end; ++z) {
                    *output_color_ptr++ = ARGB_to_ABGR(*color++);
                }

                runs[num_runs].top_voxels_start = top_color_start;
                runs[num_runs].top_voxels_end = top_color_end+1; // NOTE: this here is important!
                runs[num_runs].bottom_voxels_start = bottom_color_start;
                runs[num_runs++].bottom_voxels_end = bottom_color_end;
            }
            //if(z < 63) {
            //    runs[num_runs].top_voxels_start = 63;
            //    runs[num_runs].top_voxels_end = 63;
            //    runs[num_runs].bottom_voxels_start = 63;
            //    runs[num_runs++].bottom_voxels_end = 63;
            //}
            assert(runs[num_runs-1].bottom_voxels_end == 63);
            header->num_runs = num_runs;
            header->max_y = runs[0].top_voxels_start;
        }
   }
    assert(v-base == len);
    return 0;

}


#define MAX_LIGHT_RAY_STEPS 20
void light_map() {

    for(int y = 0; y < 512; y++) {
        for(int x = 0; x < 512; x++) {
            for(int z = 0; z < 64; z++) {
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


                
                for(int tz = z-3; tz < z; tz++) {
                    int tz_diff = z-tz;
                    for(int ty = y-tz_diff; ty < y+tz_diff+1; ty++) {
                        for(int tx = x-tz_diff; tx < x+tz_diff+1; tx++) {
                            samples++;
                            if(tx == x && ty == y && tz == z) {
                                continue;
                            }
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
                
                f32 zero_to_one  = (solid_cells/(f32)samples)/2;
                f32 one_to_zero = 1-zero_to_one;//(samples-solid_cells)/(f32)samples;
                f32 one_to_zero_scaled = 255.0 * one_to_zero;
                u8 one_to_zero_alpha = (u8)floorf(one_to_zero_scaled);
                u32 base_color = voxel_get_color(x, y, z);
                u8 r = (base_color & 0xFF);
                u8 g = ((base_color >> 8) & 0xFF);
                u8 b = ((base_color >> 16) & 0xFF);
                voxel_set_color(x, y, z, ((one_to_zero_alpha<<24)|(b<<16)|(g<<8)|r));
            }
        }
    }
}

#include <dirent.h> 
#include "map_table.c"


static int map_table_loaded = 0;

void load_map(s32 map_idx) {  
    if(!map_table_loaded) {
        load_map_table();

        map_table_loaded = 1;
    }

    while(map_idx >= num_maps) { map_idx -= num_maps; }
    char buf[32];
    sprintf(buf, "./maps/%s", &map_name_table[map_idxs[map_idx]]);
    if(load_voxlap_map(buf) != 0) {
        printf("Error loading map '%s'\n", buf);
        exit(1);
    }
    light_map();
    return;      

#if 0
    map_info map_info = map_table[map_idx];
    printf("Loading colormap %i...\n", map_idx);
    char color_filename[32];
    sprintf(color_filename, "./res/C%i%s.png", map_info.color_idx, map_info.water_version ? "W" : "");
    char depth_filename[32];
    sprintf(depth_filename, "./res/D%i.png", map_info.depth_idx);

    int x,y,n,ok;
    printf("Loading colormap %s...\n", color_filename);
    ok = stbi_info(color_filename, &x, &y, &n);
    if(!ok) {
        printf("Error loading colormap!\n");
    } 
    if(x != 1024 || y != 1024) {
        printf("Colormap dimensions not correct: %i,%i!\n", x, y);
        exit(1);
    } else {
        load_colormap(color_filename);
    }
    //int x,y,n,ok;
    printf("Loading depthmap %s...\n", depth_filename);
    ok = stbi_info(depth_filename, &x, &y, &n);
    if(!ok) {
        printf("Error loading depthmap!\n");
    } 
    if(x == 512 && y == 512) {
        load_and_interpolate_depthmap(depth_filename);
    } else if (x == 1024 && y == 1024) {
        load_depthmap(depth_filename);
    } else {
        printf("Depthmap dimensions not correct: %i,%i!\n", x, y);
        exit(1);
    }

    swizzle_array(scratch_depthmap);
    swizzle_array((u32*)scratch_colormap);
    calculate_normals(normal_pt1_data, normal_pt2_data);

    for(int y = 0; y < 1024; y++) {
        for(int x = 0; x < 1024; x++) {
            initialize_voxelmap_run_entry(x, y, scratch_depthmap[get_swizzled_map_idx(x,y)], ((u32*)scratch_colormap)[get_swizzled_map_idx(x,y)]);
        }
    }
    if(map_idx == 30) {
        column_header* header = &columns_header_data[get_voxelmap_idx(512,512)];
        column_colors* colors = &columns_colors_data[get_voxelmap_idx(512,512)];
        column_runs* runs = &columns_runs_data[get_voxelmap_idx(512,512)];
        printf("%p\n", header);
    }
    //vmap->loaded = 1;
    //vmap->loaded_map_idx = map_idx;
#endif
}



#endif