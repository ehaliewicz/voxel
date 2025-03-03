#include "config.h"
#include "types.h"
#include "utils.h"

#include "voxelmap.h"



/* 
    mip levels of chunk
    0 128x128x256
    1 64x64x128
    2 32x32x64
*/

typedef struct {
    volatile u64 dirty_voxels:3; // a bit N is set if mip N has stale mip and lighting info
    volatile u64 mip_loaded:3;
    int toplevel_chunk_tweaked:1;
    int global_x;
    int global_y;
} chunk_info;

int allocated_chunk_data = 0;

chunk_info* chunks;



u8 tweak_color_rand_table[256*256];
int rand_table_idx = 0;
int rand_table_init = 0;

void init_rand_table() {
    for(int i = 0; i < 256; i++) {
        for(int co = 0; co < 256; co++) {
            f32 percent_diff = lerp(0.96f, rand()/((float)RAND_MAX), 1.04f);
            f32 mod_col = (co/256.0f) * percent_diff;
            f32 clamp_col = (mod_col > 1.0f) ? 1.0f : ((mod_col < 0.0f) ? 0.0f : mod_col);
            tweak_color_rand_table[i*256+co] = (u8)(clamp_col*255.0f);
        }
    }
}

void prepare_chunk(int global_x, int global_y, int mip_level, int dummy_arg);

typedef struct {
    s32 el_sz;
    s32 cap;
    s32 len;
    u8* els;
} darray;

darray resize_darray(darray darr) {
    while(darr.len >= darr.cap) {
        u32 new_cap = darr.cap * 2;
        if(new_cap < 8) { new_cap = 8; }
        darr.els = realloc(darr.els, sizeof(darray)+darr.el_sz*new_cap);
        darr.cap = new_cap;
    }

    return darr;
}

u64 calc_chunk_statistics(chunk_info c) {
    int global_x = c.global_x;
    int global_y = c.global_y;
    darray cols = {.cap = 0, .len = 0, .el_sz = sizeof(u32), .els = NULL };
    darray col_cnts = {.cap = 0, .len = 0, .el_sz = sizeof(u32), .els = NULL };

    f64 tot_vox_per_column = 0;
    f64 tot_run_per_column = 0;

    for(int y = global_y; y < global_y+CHUNK_SIZE; y++) {
        //printf("row %i/%i\n", y-global_y, CHUNK_SIZE);
        for(int x = global_x; x < global_x+CHUNK_SIZE; x++) {

            int idx = get_voxelmap_idx(x, y);
            int num_runs = columns_header_data[idx].num_runs;
            tot_run_per_column += num_runs;

            span* runs = columns_runs_data[idx].runs_info;
            u32* col_ptr = columns_colors_data[idx].colors;
            for(int r = 0; r < num_runs; r++) {
                int top_top = runs[r].top_surface_start;
                int top_exclusive_bot = runs[r].top_surface_end+1;
                int bot_top = runs[r].bot_surface_start;
                int bot_exclusive_bot = runs[r].bot_surface_end;

                int num_cols_top = top_exclusive_bot - top_top;
                int num_cols_bot = (bot_exclusive_bot <= bot_top) ? 0 : (bot_exclusive_bot - bot_top);
                tot_vox_per_column += num_cols_top;
                tot_vox_per_column += num_cols_bot;


                for(int c = 0; c < num_cols_top+num_cols_bot; c++) {
                    u32 color = col_ptr[c];

                    // calculating number of colors per block
                    int matched_idx = cols.len;
                    for(int i = 0; i < cols.len; i++) {
                        if(((u32*)(cols.els))[i] == color) {
                            matched_idx = i;
                            ((u32*)(col_cnts.els))[i]++;
                            goto dont_add_new_color;
                        }
                    }
                    // add new color and count
                    // realloc if necessary
                    cols = resize_darray(cols);
                    col_cnts = resize_darray(col_cnts);

                    ((u32*)(cols.els))[matched_idx] = color;
                    ((u32*)(col_cnts.els))[matched_idx] = 1;
                    cols.len++;
                    col_cnts.len++;
                dont_add_new_color:;
                }
            }



        }
    }
    //avg_vox_per_column /= (CHUNK_SIZE*CHUNK_SIZE);
    printf("%i Total unique colors for chunk\n", cols.len);
    printf("Avg visible voxels per column %f\n", tot_vox_per_column/(CHUNK_SIZE*CHUNK_SIZE));
    printf("Avg runs per column %f\n", tot_run_per_column/(CHUNK_SIZE*CHUNK_SIZE));
    u64 sum_all_colors = 0;
    for(int i = 0; i < col_cnts.len; i++) {
        sum_all_colors += ((u32*)(col_cnts.els))[i];
    }
    //printf("%i Total voxels for chunk\n", sum_all_colors);
    return cols.len;
}

u64 calc_global_statistics() {
    darray cols = {.cap = 0, .len = 0, .el_sz = sizeof(u32), .els = NULL };
    darray col_cnts = {.cap = 0, .len = 0, .el_sz = sizeof(u32), .els = NULL };

    f64 tot_vox_per_column = 0;

    for(int y = 0; y < MAP_Y_SIZE; y++) {
        //printf("row %i/%i\n", y-global_y, CHUNK_SIZE);
        for(int x = 0; x < MAP_X_SIZE; x++) {

            int idx = get_voxelmap_idx(x, y);
            int num_runs = columns_header_data[idx].num_runs;
            span* runs = columns_runs_data[idx].runs_info;
            u32* col_ptr = columns_colors_data[idx].colors;
            for(int r = 0; r < num_runs; r++) {
                int top_top = runs[r].top_surface_start;
                int top_exclusive_bot = runs[r].top_surface_end+1;
                int bot_top = runs[r].bot_surface_start;
                int bot_exclusive_bot = runs[r].bot_surface_end;

                int num_cols_top = top_exclusive_bot - top_top;
                int num_cols_bot = (bot_exclusive_bot <= bot_top) ? 0 : (bot_exclusive_bot - bot_top);
                tot_vox_per_column += num_cols_top;
                tot_vox_per_column += num_cols_bot;

                for(int c = 0; c < num_cols_top+num_cols_bot; c++) {
                    u32 color = col_ptr[c];

                    // calculating number of colors per block
                    int matched_idx = cols.len;
                    for(int i = 0; i < cols.len; i++) {
                        if(((u32*)(cols.els))[i] == color) {
                            matched_idx = i;
                            ((u32*)(col_cnts.els))[i]++;
                            goto dont_add_new_color;
                        }
                    }
                    // add new color and count
                    // realloc if necessary
                    cols = resize_darray(cols);
                    col_cnts = resize_darray(col_cnts);

                    ((u32*)(cols.els))[matched_idx] = color;
                    ((u32*)(col_cnts.els))[matched_idx] = 1;
                    cols.len++;
                    col_cnts.len++;
                dont_add_new_color:;
                }
            }



        }
    }
    //avg_vox_per_column /= (CHUNK_SIZE*CHUNK_SIZE);
    printf("%i Total unique colors for chunk\n", cols.len);
    printf("Avg visible voxels per column %f\n", tot_vox_per_column/(MAP_X_SIZE*MAP_Y_SIZE));
    u64 sum_all_colors = 0;
    for(int i = 0; i < col_cnts.len; i++) {
        sum_all_colors += ((u32*)(col_cnts.els))[i];
    }
    //printf("%i Total voxels for chunk\n", sum_all_colors);
    return cols.len;
}


void prepare_chunk(int global_x, int global_y, int mip_level, int dummy_arg);

thread_pool_function(prepare_chunk_wrapper, arg_var)
{
	thread_params* tp = (thread_params*)arg_var;
    prepare_chunk(tp->min_x, tp->min_y, tp->max_x, tp->max_y);
	InterlockedIncrement64(&tp->finished);
}

void prepare_chunks_from_pos(f32 pos_x, f32 pos_y, int async);

void init_chunks() {

    if(!rand_table_init) {
        rand_table_init = 1;
        init_rand_table();
    }


    if(!allocated_chunk_data) {
        chunks = malloc_wrapper(sizeof(chunk_info)*CHUNKS_PER_AXIS*CHUNKS_PER_AXIS, "chunk info");
    }

    printf("resetting all chunks");

    {
        u64 biggest_color_count = 0;
        for(int y = 0; y < CHUNKS_PER_AXIS; y++) {
            for(int x = 0; x < CHUNKS_PER_AXIS; x++) {
                chunks[y*CHUNKS_PER_AXIS+x] = ((chunk_info){
                    .dirty_voxels = 0b111,
                    .mip_loaded = 0b000,
                    .toplevel_chunk_tweaked = 1,
                    .global_x = x*CHUNK_SIZE,
                    .global_y = y*CHUNK_SIZE
                });


                //biggest_color_count = max(biggest_color_count, calc_chunk_statistics(chunks[y*CHUNKS_PER_AXIS+x]));
            }
        }
        //printf("BIGGEST COLOR COUNT PER CHUNK %llu\n", biggest_color_count);
        //calc_global_statistics();
    }


    // assume raw map data is loaded
    // this assumption may change later on

    // but all voxels are dirty at this point

    // generate tasks to MIP, TWEAK COLORS, and LIGHT all chunks, at lowest mip level

    // lighting only done for lowest mip level

    thread_params init_chunk_params[CHUNKS_PER_AXIS*CHUNKS_PER_AXIS];
    
    for(int y = 0; y < CHUNKS_PER_AXIS; y++) {
        for(int x = 0; x < CHUNKS_PER_AXIS; x++) {
            int global_x = x*CHUNK_SIZE;
            int global_y = y*CHUNK_SIZE;
            init_chunk_params[y*CHUNKS_PER_AXIS+x] = ((thread_params) {
                .finished=0, .min_x = global_x, .min_y = global_y, .max_x = 1, .max_y = -1 // last arg is dummy
            });
        }
    }

    job_pool init_chunk_jobs = {
        .num_jobs = CHUNKS_PER_AXIS*CHUNKS_PER_AXIS,
        .parms = init_chunk_params,
        .func = prepare_chunk_wrapper,
        .raw_func = prepare_chunk
    };
    start_pool(map_thread_pool, &init_chunk_jobs);

    wait_for_job_pool_to_finish(&init_chunk_jobs);

    for(int y = 0; y < CHUNKS_PER_AXIS; y++) {
        for(int x = 0; x < CHUNKS_PER_AXIS; x++) {
            int global_x = x*CHUNK_SIZE;
            int global_y = y*CHUNK_SIZE;
            init_chunk_params[y*CHUNKS_PER_AXIS+x] = ((thread_params) {
                .finished=0, .min_x = global_x, .min_y = global_y, .max_x = 0, .max_y = -1 // last arg is dummy
            });
        }
    }
    start_pool(map_thread_pool, &init_chunk_jobs);
    wait_for_job_pool_to_finish(&init_chunk_jobs);

    for(int y = 0; y < CHUNKS_PER_AXIS; y++) {
        for(int x = 0; x < CHUNKS_PER_AXIS; x++) {

            // this forces mip level 1 and 2 to be generated
            //prepare_chunk(x*CHUNK_SIZE, y*CHUNK_SIZE, 2, -1);


        }
    }
    

    //prepare_chunks_from_pos(pos_x, pos_y, 0);
}


u32 tweak_color(u32 abgr) {
    u8 a = (abgr>>24)&0xFF;
    u8 b = (abgr>>16)&0xFF;
    u8 g = (abgr>>8)&0xFF;
    u8 r = abgr&0xFF;


    u8 bc = tweak_color_rand_table[(((rand_table_idx++)&0xFF)<<8)+b];//g;
    u8 gc = tweak_color_rand_table[(((rand_table_idx++)&0xFF)<<8)+g];//g;
    u8 rc = tweak_color_rand_table[(((rand_table_idx++)&0xFF)<<8)+r];//r;
 
    return (a<<24)|(bc<<16)|(gc<<8)|rc;
}


void tweak_column_colors(int min_x, int min_y, int max_x, int max_y) {

    for(int y = min_y; y < max_y; y++) {
        for(int x = min_x; x < max_x; x++) {
            u32 idx = get_voxelmap_idx(x,y);
            column_runs runs = columns_runs_data[idx];
            int col_idx = 0;
            for(int i = 0; i < columns_header_data[idx].num_runs; i++) {
                for(int z = runs.runs_info[i].top_surface_start; z <= runs.runs_info[i].top_surface_end; z++) {
                    columns_colors_data[idx].colors[col_idx] = tweak_color(columns_colors_data[idx].colors[col_idx]);
                    col_idx++;

                }
                for(int z = runs.runs_info[i].bot_surface_start; z < runs.runs_info[i].bot_surface_end; z++) {
                    columns_colors_data[idx].colors[col_idx] = tweak_color(columns_colors_data[idx].colors[col_idx]);
                    col_idx++;
                }
            }
        }
    }
}

void tweak_mip_column_colors(int min_x, int min_y, int max_x, int max_y) {

    for(int y = min_y; y < max_y; y += 2) {
        for(int x = min_x; x < max_x; x += 2) {
            u32 idx = get_mip_voxelmap_idx(x,y);
            mip_column_runs runs = mip_columns_runs_data[idx];
            int col_idx = 0;
            for(int i = 0; i < mip_columns_header_data[idx].num_runs; i++) {
                for(int z = runs.runs_info[i].top_surface_start; z <= runs.runs_info[i].top_surface_end; z++) {
                    mip_columns_colors_data[idx].colors[col_idx] = tweak_color(mip_columns_colors_data[idx].colors[col_idx]);
                    col_idx++;

                }
                for(int z = runs.runs_info[i].bot_surface_start; z < runs.runs_info[i].bot_surface_end; z++) {
                    mip_columns_colors_data[idx].colors[col_idx] = tweak_color(mip_columns_colors_data[idx].colors[col_idx]);
                    col_idx++;
                }
            }
        }
    }
}

void tweak_mip2_column_colors(int min_x, int min_y, int max_x, int max_y) {

    for(int y = min_y; y < max_y; y += 4) {
        for(int x = min_x; x < max_x; x += 4) {
            u32 idx = get_mip2_voxelmap_idx(x,y);
            mip_column_runs runs = mip2_columns_runs_data[idx];
            int col_idx = 0;
            for(int i = 0; i < mip2_columns_header_data[idx].num_runs; i++) {
                for(int z = runs.runs_info[i].top_surface_start; z <= runs.runs_info[i].top_surface_end; z++) {
                    mip2_columns_colors_data[idx].colors[col_idx] = tweak_color(mip2_columns_colors_data[idx].colors[col_idx]);
                    col_idx++;

                }
                for(int z = runs.runs_info[i].bot_surface_start; z < runs.runs_info[i].bot_surface_end; z++) {
                    mip2_columns_colors_data[idx].colors[col_idx] = tweak_color(mip2_columns_colors_data[idx].colors[col_idx]);
                    col_idx++;
                }
            }
        }
    }
}


#define AMBIENT_OCCLUSION_RADIUS 2
#define NORMAL_RADIUS 2

#define AMBIENT_OCCLUSION_DIAMETER (AMBIENT_OCCLUSION_RADIUS*2+1)


void light_map_mip0(int min_x, int min_y, int max_x, int max_y) {
    for(int y = min_y; y < max_y; y++) {
        for(int x = min_x; x < max_x; x++) {
            u32 voxelmap_idx = get_voxelmap_idx(x, y);
            column_header* header = &columns_header_data[voxelmap_idx];
            span* runs = columns_runs_data[voxelmap_idx].runs_info;

            u32* color_ptr = columns_colors_data[voxelmap_idx].colors;
            f32* normal_pt1_ptr = columns_norm_data[voxelmap_idx].norm_pt1;
            f32* normal_pt2_ptr = columns_norm_data[voxelmap_idx].norm_pt2;

            for(int i = 0; i < header->num_runs; i++) {
                
                int top_z = runs[i].top_surface_start;
                int end_top_z = runs[i].top_surface_end;
                int end_top_z_exclusive = end_top_z+1;
                int jump_z = runs[i].bot_surface_start;
                int bot_z = runs[i].bot_surface_end;

                int skipped = 0;
                for(int z = top_z; z < max(end_top_z_exclusive, bot_z); z++) {
                    if(z > end_top_z && !skipped) { 
                        z = jump_z-1;
                        skipped = 1;
                        continue; // make sure there are actual bottom voxels before iterating
                    }
                    int solid_cells = 0;
                    int samples = 0;

                    f32 norm_x = 0.0;
                    f32 norm_y = 0.0;
                    f32 norm_z = 0.0;

                    for(int yy = -AMBIENT_OCCLUSION_RADIUS; yy <= AMBIENT_OCCLUSION_RADIUS; yy++) {
                        int ty = y+yy;
                        if(ty < 0 || ty >= MAP_Y_SIZE) { continue; }
                        for(int xx = -AMBIENT_OCCLUSION_RADIUS; xx <= AMBIENT_OCCLUSION_RADIUS; xx++) {
                            int tx = x+xx;
                            if(tx < 0 || tx > MAP_X_SIZE) { continue; }

                            u32 test_voxelmap_idx = get_voxelmap_idx(tx, ty);

                            int test_span_num_runs = columns_header_data[test_voxelmap_idx].num_runs;
                            int test_span_top_y = columns_header_data[test_voxelmap_idx].top_y;
                            if(test_span_num_runs == 0) { continue; }

                            for(int zz = -AMBIENT_OCCLUSION_RADIUS; zz <= 0; zz++) { 
                                
                                int tz = z+zz;
                                if(tz < 0) { continue; }
                                u8 valid_ao_sample = (tz <= z);

                                samples += (valid_ao_sample ? 1 : 0);

                                //u8 cell_is_solid = voxel_is_solid(tx,ty,tz); 
                                u8 cell_is_solid = get_bit_in_bitmap(tz, &columns_bitmaps_data[test_voxelmap_idx]) ? 1 : 0;
                                
                                //norm_x += cell_is_solid ? -xx : 0.0;
                                //norm_y += cell_is_solid ? -yy : 0.0;
                                //norm_z += cell_is_solid ? -zz : 0.0;

                                solid_cells += ((valid_ao_sample && cell_is_solid) ? 1 : 0);
                            }
                        }
                    }

                    if(norm_x == 0 && norm_y == 0 && norm_z == 0) {
                        norm_z = -1;
                    }

                    
                    // 0 -> no filled surrounding voxels, 1 -> all filled surrounding voxels
                    // but divide in 2 to reduce the effect, so the effect is more subtle
                    f32 zero_to_one; 
                    samples--; // subtract the center sample
                    solid_cells--;
                    if(samples == 0) {
                        zero_to_one = 0;
                    } else {
                        zero_to_one = ((solid_cells*.25)/(f32)samples);
                    }


                    f32 one_to_zero = 1-zero_to_one; // 0-> all filled surrounding voxels, 0.5-> no filled surrounding voxels

                    // each albedo has 6 AO bits, so use up all 6 of them
                    f32 one_to_zero_scaled = 63.0 * one_to_zero; //63.0 * one_to_zero; // scale from 0->.5 to 0-63


                    //f32 len = magnitude_vector(norm_x, norm_y, norm_z);
                    //f32 fnorm_x = norm_x / len;
                    //f32 fnorm_y = norm_y / len;
                    //f32 fnorm_z = norm_z / len;
                    //float2 norm = encode_norm(fnorm_x, fnorm_y, fnorm_z);
                    //*normal_pt1_ptr++ = norm.x;
                    //*normal_pt2_ptr++ = norm.y;

                    //voxel_set_normal(x, y, z, norm.x, norm.y);

                    u8 one_to_zero_ao_bits = ((u8)floorf(one_to_zero_scaled));
                    u32 base_color = get_voxel_color(x, y, z);
                    u8 alpha_bits = (base_color>>24)&0b11;
                    long r, g, b;

                    r = min(((base_color & 0xFF)),255);
                    g = min((((base_color >> 8) & 0xFF)),255);
                    b = min((((base_color >> 16) & 0xFF)),255);

                    u8 ao_and_alpha_byte = (one_to_zero_ao_bits<<2) | alpha_bits;
                    u32 col = ((ao_and_alpha_byte<<24)|(b<<16)|(g<<8)|r);
                    *color_ptr++ = col;
                    //voxel_set_color(x, y, z, ((ao_and_alpha_byte<<24)|(b<<16)|(g<<8)|r));

                }
            }
        }
    }
}


void light_map_mip1(int min_x, int min_y, int max_x, int max_y) {
    for(int y = min_y; y < max_y; y++) {
        for(int x = min_x; x < max_x; x++) {
            u32 voxelmap_idx = get_mip_voxelmap_idx(x, y);
            column_header* header = &mip_columns_header_data[voxelmap_idx];
            span* runs = mip_columns_runs_data[voxelmap_idx].runs_info;

            for(int i = 0; i < header->num_runs; i++) {
                
                int top_z = runs[i].top_surface_start;
                int end_top_z = runs[i].top_surface_end;
                int end_top_z_exclusive = end_top_z+1;
                int jump_z = runs[i].bot_surface_start;
                int bot_z = runs[i].bot_surface_end;

                int skipped = 0;
                for(int z = top_z; z < max(end_top_z_exclusive, bot_z); z++) {
                        if(z > end_top_z && !skipped) { 
                            z = jump_z-1;
                            skipped = 1;
                            continue; // make sure there are actual bottom voxels before iterating
                        }
                        int solid_cells = 0;
                        int samples = 0;

                        f32 norm_x = 0.0;
                        f32 norm_y = 0.0;
                        f32 norm_z = 0.0;

                        for(int yy = -AMBIENT_OCCLUSION_RADIUS; yy <= AMBIENT_OCCLUSION_RADIUS; yy++) {
                            int ty = y+yy;
                            if(ty < 0 || ty >= MIP_MAP_Y_SIZE) { continue; }
                            for(int xx = -AMBIENT_OCCLUSION_RADIUS; xx <= AMBIENT_OCCLUSION_RADIUS; xx++) {
                                int tx = x+xx;
                                if(tx < 0 || tx > MIP_MAP_X_SIZE) { continue; }

                                u32 test_voxelmap_idx = get_mip_voxelmap_idx(tx, ty);

                                int test_span_num_runs = mip_columns_header_data[test_voxelmap_idx].num_runs;
                                int test_span_top_y = mip_columns_header_data[test_voxelmap_idx].top_y;
                                if(test_span_num_runs == 0) { continue; }

                                //for(int zz = -NORMAL_RADIUS; zz <= NORMAL_RADIUS; zz++) { 
                                    
                                for(int zz = -AMBIENT_OCCLUSION_RADIUS; zz <= 0; zz++) {

                                    int tz = z+zz;
                                    if(tz < 0) { continue; }
                                    u8 valid_ao_sample = (tz <= z);

                                    samples += (valid_ao_sample ? 1 : 0);

                                    //u8 cell_is_solid = voxel_is_solid(tx,ty,tz); 
                                    u8 cell_is_solid = get_bit_in_mip1_bitmap(tz, &mip_columns_bitmaps_data[test_voxelmap_idx]) ? 1 : 0;
                                    
                                    //norm_x += cell_is_solid ? -xx : 0.0;
                                    //norm_y += cell_is_solid ? -yy : 0.0;
                                    //norm_z += cell_is_solid ? -zz : 0.0;

                                    solid_cells += ((valid_ao_sample && cell_is_solid) ? 1 : 0);
                                }
                            }
                        }

                        if(norm_x == 0 && norm_y == 0 && norm_z == 0) {
                            norm_z = -1;
                        }

                        
                        // 0 -> no filled surrounding voxels, 1 -> all filled surrounding voxels
                        // but divide in 2 to reduce the effect, so the effect is more subtle
                        f32 zero_to_one; 
                        samples--; // subtract the center sample
                        solid_cells--;
                        if(samples == 0) {
                            zero_to_one = 0;
                        } else {
                            zero_to_one = ((solid_cells*.25)/(f32)samples);
                        }


                        f32 one_to_zero = 1-zero_to_one; // 0-> all filled surrounding voxels, 0.5-> no filled surrounding voxels

                        // each albedo has 6 AO bits, so use up all 6 of them
                        f32 one_to_zero_scaled = 63.0 * one_to_zero; //63.0 * one_to_zero; // scale from 0->.5 to 0-63


                        //f32 len = magnitude_vector(norm_x, norm_y, norm_z);
                        //f32 fnorm_x = norm_x / len;
                        //f32 fnorm_y = norm_y / len;
                        //f32 fnorm_z = norm_z / len;
                        //float2 norm = encode_norm(fnorm_x, fnorm_y, fnorm_z);
                        //voxel_mip1_set_normal(x, y, z, norm.x, norm.y);

                        u8 one_to_zero_ao_bits = ((u8)floorf(one_to_zero_scaled));
                        u32 base_color = get_mip1_voxel_color(x, y, z);
                        u8 alpha_bits = (base_color>>24)&0b11;
                        long r, g, b;

                        r = min(((base_color & 0xFF)),255);
                        g = min((((base_color >> 8) & 0xFF)),255);
                        b = min((((base_color >> 16) & 0xFF)),255);

                        u8 ao_and_alpha_byte = (one_to_zero_ao_bits<<2) | alpha_bits;
                        voxel_mip1_set_color(x, y, z, ((ao_and_alpha_byte<<24)|(b<<16)|(g<<8)|r));

                    }
            }
        }
    }
}



void light_map_mip2(int min_x, int min_y, int max_x, int max_y) {
    for(int y = min_y; y < max_y; y++) {
        for(int x = min_x; x < max_x; x++) {
            u32 voxelmap_idx = get_mip2_voxelmap_idx(x, y);
            column_header* header = &mip2_columns_header_data[voxelmap_idx];
            span* runs = mip2_columns_runs_data[voxelmap_idx].runs_info;

            for(int i = 0; i < header->num_runs; i++) {
                
                int top_z = runs[i].top_surface_start;
                int end_top_z = runs[i].top_surface_end;
                int end_top_z_exclusive = end_top_z+1;
                int jump_z = runs[i].bot_surface_start;
                int bot_z = runs[i].bot_surface_end;

                int skipped = 0;
                for(int z = top_z; z < max(end_top_z_exclusive, bot_z); z++) {
                        if(z > end_top_z && !skipped) { 
                            z = jump_z-1;
                            skipped = 1;
                            continue; // make sure there are actual bottom voxels before iterating
                        }
                        int solid_cells = 0;
                        int samples = 0;

                        f32 norm_x = 0.0;
                        f32 norm_y = 0.0;
                        f32 norm_z = 0.0;

                        for(int yy = -AMBIENT_OCCLUSION_RADIUS; yy <= AMBIENT_OCCLUSION_RADIUS; yy++) {
                            int ty = y+yy;
                            if(ty < 0 || ty >= MIP_MAP_Y_SIZE) { continue; }
                            for(int xx = -AMBIENT_OCCLUSION_RADIUS; xx <= AMBIENT_OCCLUSION_RADIUS; xx++) {
                                int tx = x+xx;
                                if(tx < 0 || tx > MIP_MAP_X_SIZE) { continue; }

                                u32 test_voxelmap_idx = get_mip2_voxelmap_idx(tx, ty);

                                int test_span_num_runs = mip2_columns_header_data[test_voxelmap_idx].num_runs;
                                int test_span_top_y = mip2_columns_header_data[test_voxelmap_idx].top_y;
                                if(test_span_num_runs == 0) { continue; }

                                //for(int zz = -NORMAL_RADIUS; zz <= NORMAL_RADIUS; zz++) { 
                                for(int zz = -AMBIENT_OCCLUSION_RADIUS; zz <= 0; zz++) {
                                    
                                    int tz = z+zz;
                                    if(tz < 0) { continue; }
                                    u8 valid_ao_sample = (tz <= z);

                                    samples += (valid_ao_sample ? 1 : 0);

                                    //u8 cell_is_solid = voxel_is_solid(tx,ty,tz); 
                                    u8 cell_is_solid = get_bit_in_mip1_bitmap(tz, &mip2_columns_bitmaps_data[test_voxelmap_idx]) ? 1 : 0;
                                    
                                    //norm_x += cell_is_solid ? -xx : 0.0;
                                    //norm_y += cell_is_solid ? -yy : 0.0;
                                    //norm_z += cell_is_solid ? -zz : 0.0;

                                    solid_cells += ((valid_ao_sample && cell_is_solid) ? 1 : 0);
                                }
                            }
                        }

                        if(norm_x == 0 && norm_y == 0 && norm_z == 0) {
                            norm_z = -1;
                        }

                        
                        // 0 -> no filled surrounding voxels, 1 -> all filled surrounding voxels
                        // but divide in 2 to reduce the effect, so the effect is more subtle
                        f32 zero_to_one; 
                        samples--; // subtract the center sample
                        solid_cells--;
                        if(samples == 0) {
                            zero_to_one = 0;
                        } else {
                            zero_to_one = ((solid_cells*.25)/(f32)samples);
                        }


                        f32 one_to_zero = 1-zero_to_one; // 0-> all filled surrounding voxels, 0.5-> no filled surrounding voxels

                        // each albedo has 6 AO bits, so use up all 6 of them
                        f32 one_to_zero_scaled = 63.0 * one_to_zero; //63.0 * one_to_zero; // scale from 0->.5 to 0-63


                        //f32 len = magnitude_vector(norm_x, norm_y, norm_z);
                        //f32 fnorm_x = norm_x / len;
                        //f32 fnorm_y = norm_y / len;
                        //f32 fnorm_z = norm_z / len;
                        //float2 norm = encode_norm(fnorm_x, fnorm_y, fnorm_z);
                        //voxel_mip2_set_normal(x, y, z, norm.x, norm.y);

                        u8 one_to_zero_ao_bits = ((u8)floorf(one_to_zero_scaled));
                        
                        u32 base_color = get_mip2_voxel_color(x, y, z);
                        u8 alpha_bits = (base_color>>24)&0b11;
                        long r, g, b;

                        r = min(((base_color & 0xFF)),255);
                        g = min((((base_color >> 8) & 0xFF)),255);
                        b = min((((base_color >> 16) & 0xFF)),255);

                        u8 ao_and_alpha_byte = (one_to_zero_ao_bits<<2) | alpha_bits;
                        voxel_mip2_set_color(x, y, z, ((ao_and_alpha_byte<<24)|(b<<16)|(g<<8)|r));
                    }
            }
        }
    }
}



u32 blend_mip_colors(u32 colors[8], int num_colors) {
    f32 ao = 0.0f;
    f32 t = 0.0f;
    f32 r = 0.0f;
    f32 g = 0.0f;
    f32 b = 0.0f;
    for(int i = 0; i < num_colors; i++) {
        r += ((colors[i]>>16)&0xFF)/255.0f;
        g += ((colors[i]>>8)&0xFF)/255.0f;
        b += ((colors[i]>>0)&0xFF)/255.0f;
        ao += ((colors[i]>>26)&0b111111)/63.0f;
        t += ((colors[i]>>24)&0b11)/3.0f;
    }
    r /= num_colors;
    g /= num_colors;
    b /= num_colors;
    ao /= num_colors;
    //ao /= 2;
    t /= num_colors;
    u8 ur = (r * 255.0f);
    u8 ug = (g * 255.0f);
    u8 ub = (b * 255.0f);
    u8 uao = (ao * 63.0f) >= 63.0f ? 63.0 : ao * 63.0f;
    u8 ut = (t * 3.0f);

    return (uao<<26) | (ut<<24) | (ur<<16) | (ug<<8) | ub;
}

void mip_columns(
    int non_mip_map_x,
    int non_mip_map_y,
    //column_bitmaps bitmaps[4],
    column_header* output_header,
    mip_column_colors* output_colors,
    mip_column_runs* output_runs,

    //column_normals* output_normals
    mip_column_bitmaps* output_bitmaps
    ) {


    // clear bitmap
    output_bitmaps->bits[0] = 0;
    output_bitmaps->bits[1] = 0;
    
    int color_output_idx = 0;
    int cur_run_output_idx = 0;
    run_state cur_state = OUT_OF_RUN;

    u8 cur_run_top_z;
    u8 cur_run_bot_z;

    u8 top_top_z = 0;
    u8 got_top_top_z = 0;

    //memset(output_colors->colors, 0xFF, sizeof(mip_column_colors));
    int z = 0;
    int lowest_min_z = cur_map_max_height;
    for(int y = 0; y < 2; y++) {
        for(int x = 0; x < 2; x++) {
            s32 gmap_x = non_mip_map_x+x;
            s32 gmap_y = non_mip_map_y+y;
            s32 gmap_idx = get_voxelmap_idx(gmap_x, gmap_y);
            lowest_min_z = min(columns_header_data[gmap_idx].top_y, lowest_min_z);
        }
    }


    for(z = lowest_min_z; z <= cur_map_max_height; z += 2) {
        
        // test if cur_y is in a span
        // this is kinda shitty
        u8 voxel_bitmap = 0;
        u8 got_voxel = 0;

        u32 mip_colors[8] = {0,0,0,0,0,0,0,0};
        u8 got_colors = 0;

        for(int y = 0; y < 2; y++) {
            for(int x = 0; x < 2; x++) {
                s32 gmap_x = non_mip_map_x+x;
                s32 gmap_y = non_mip_map_y+y;
                s32 gmap_idx = get_voxelmap_idx(gmap_x, gmap_y);

                u8 num_runs = columns_header_data[gmap_idx].num_runs;
                span* runs = columns_runs_data[gmap_idx].runs_info;
                u32* color_ptr = columns_colors_data[gmap_idx].colors;

                for(int tz = z; tz < z+2; tz++) {
                    int i = y*2+x;
                    int cumulative_skipped_voxels = 0;
                    for(int j = 0; j < num_runs; j++) {
                        int top_top = runs[j].top_surface_start;
                        int top_exclusive_bot = runs[j].top_surface_end+1;
                        
                        if ((tz >= top_top && tz < top_exclusive_bot)) { 
                            u32 col = color_ptr[cumulative_skipped_voxels + (tz - top_top)];
                            //if(col != 0) {
                            mip_colors[got_colors++] = col;
                            //}
                            break;
                        }
                        cumulative_skipped_voxels += (top_exclusive_bot - top_top);


                        int bot_top = runs[j].bot_surface_start;
                        int bot_exclusive_bot = runs[j].bot_surface_end;

                        if ((tz >= bot_top && tz < bot_exclusive_bot)) {
                            u32 col = color_ptr[cumulative_skipped_voxels + (tz - bot_top)];
                            mip_colors[got_colors++] = col;
                            break;
                        }
                        if(bot_exclusive_bot > bot_top) {
                            cumulative_skipped_voxels += (bot_exclusive_bot - bot_top);
                        }
                    }
                }
            }
        }

        // bitmap for these four voxels
        // ready to go
        if(got_colors) {
            int output_z = z/2;
            int output_bitmap_word = output_z >> 6;
            int output_bitmap_bit = output_z & 0b111111;

            output_bitmaps->bits[output_bitmap_word] |= (1 << output_bitmap_bit);
            if(!got_top_top_z) {
                top_top_z = output_z;
                got_top_top_z = 1;
            }
            cur_run_bot_z = output_z;
            //output_bitmaps[i].bits[y>>1] = (voxel_bitmap ? 1 : 0);
            
            output_colors->colors[color_output_idx++] = blend_mip_colors(mip_colors, got_colors);// 0xFFFFFFFF;
            if(cur_state == OUT_OF_RUN) {
                // start a run
                cur_state = IN_RUN;
                cur_run_top_z = output_z;

            } else {
                // continue a run
            }
        } else {
            if(cur_state == OUT_OF_RUN) {
                // just air
            } else {
                // end a run
                output_runs->runs_info[cur_run_output_idx].top_surface_start = cur_run_top_z;
                output_runs->runs_info[cur_run_output_idx].top_surface_end = cur_run_bot_z; //z/2-1;
                output_runs->runs_info[cur_run_output_idx].bot_surface_start = cur_run_bot_z; //z/2-1;
                output_runs->runs_info[cur_run_output_idx++].bot_surface_end = cur_run_bot_z; //z/2-1;
                cur_state = OUT_OF_RUN;

            }
        }
    }

    

    if(cur_state == IN_RUN) {
        // end a run
        output_runs->runs_info[cur_run_output_idx].top_surface_start = cur_run_top_z;
        output_runs->runs_info[cur_run_output_idx].top_surface_end = cur_run_bot_z; //z/2;
        output_runs->runs_info[cur_run_output_idx].bot_surface_start = cur_run_bot_z; //z/2;
        output_runs->runs_info[cur_run_output_idx++].bot_surface_end = cur_run_bot_z; //z/2;

    }

    output_header->num_runs = cur_run_output_idx;
    output_header->top_y = top_top_z;

}



void mip2_columns(
    int non_mip_map_x,
    int non_mip_map_y,
    //column_bitmaps bitmaps[4],
    column_header* output_header,
    mip_column_colors* output_colors,
    mip_column_runs* output_runs,

    //column_normals* output_normals
    mip_column_bitmaps* output_bitmaps
    ) {

    output_bitmaps->bits[0] = 0;
    
    int color_output_idx = 0;
    int cur_run_output_idx = 0;
    run_state cur_state = OUT_OF_RUN;

    u8 cur_run_top_z;
    u8 cur_run_bot_z;

    u8 top_top_z = 0;
    u8 got_top_top_z = 0;

    int z = 0;
    for(; z <= cur_map_max_height/2; z += 2) {
        
        // test if cur_y is in a span
        // this is kinda shitty
        u8 voxel_bitmap = 0;

        u32 mip_colors[8] = {0,0,0,0,0,0,0,0};
        u8 got_colors = 0;

        for(int y = 0; y < 2; y++) {
            for(int x = 0; x < 2; x++) {
                s32 gmap_x = non_mip_map_x+(x<<1);
                s32 gmap_y = non_mip_map_y+(y<<1);
                s32 mip1_idx = get_mip_voxelmap_idx(gmap_x, gmap_y);

                u8 num_runs = mip_columns_header_data[mip1_idx].num_runs;
                span* runs = mip_columns_runs_data[mip1_idx].runs_info;
                u32* color_ptr = mip_columns_colors_data[mip1_idx].colors;

                for(int tz = z; tz < z+2; tz++) {
                    int i = y*2+x;
                    int cumulative_skipped_voxels = 0;
                    for(int j = 0; j < num_runs; j++) {
                        int top_top = runs[j].top_surface_start;
                        int top_exclusive_bot = runs[j].top_surface_end+1;

                                                
                        if (tz >= top_top && tz < top_exclusive_bot) {
                            u32 col = color_ptr[cumulative_skipped_voxels + (tz - top_top)];
                            mip_colors[got_colors++] = col;
                            break;
                        }
                        cumulative_skipped_voxels += (top_exclusive_bot - top_top);

                        int bot_top = runs[j].bot_surface_start;
                        int bot_exclusive_bot = runs[j].bot_surface_end;
                        if(tz >= bot_top && tz < bot_exclusive_bot) {
                            u32 col = color_ptr[cumulative_skipped_voxels + (tz - top_top)];
                            mip_colors[got_colors++] = col;
                            break;
                        }

                        if(bot_exclusive_bot > bot_top) {
                            cumulative_skipped_voxels += (bot_exclusive_bot - bot_top);
                        }
                    }
                }
            }
        }

        // bitmap for these four voxels
        // ready to go
        if(got_colors) {
            int output_z = z/2;
            int output_bitmap_word = output_z >> 6;
            int output_bitmap_bit = output_z & 0b111111;
            output_bitmaps->bits[output_bitmap_word] |= (1 << output_bitmap_bit);

            if(!got_top_top_z) {
                top_top_z = output_z;
                got_top_top_z = 1;
            }
            cur_run_bot_z = output_z;
            
            output_colors->colors[color_output_idx++] = blend_mip_colors(mip_colors, got_colors);// 0xFFFFFFFF;
            if(cur_state == OUT_OF_RUN) {
                // start a run
                cur_state = IN_RUN;
                cur_run_top_z = output_z;

            } else {
                // continue a run
            }
        } else {
            if(cur_state == OUT_OF_RUN) {
                // just air
            } else {
                // end a run
                output_runs->runs_info[cur_run_output_idx].top_surface_start = cur_run_top_z;
                output_runs->runs_info[cur_run_output_idx].top_surface_end = cur_run_bot_z; 
                output_runs->runs_info[cur_run_output_idx].bot_surface_start = cur_run_bot_z; 
                output_runs->runs_info[cur_run_output_idx++].bot_surface_end = cur_run_bot_z; 
                cur_state = OUT_OF_RUN;

            }
        }
    }

    

    if(cur_state == IN_RUN) {
        // end a run
        output_runs->runs_info[cur_run_output_idx].top_surface_start = cur_run_top_z;
        output_runs->runs_info[cur_run_output_idx].top_surface_end = cur_run_bot_z; //z/2;
        output_runs->runs_info[cur_run_output_idx].bot_surface_start = cur_run_bot_z; //z/2;
        output_runs->runs_info[cur_run_output_idx++].bot_surface_end = cur_run_bot_z; //z/2;

    }

    output_header->num_runs = cur_run_output_idx;
    output_header->top_y = top_top_z;

}

// TODO: improve mip columns code
// and thread it
// but its ok for now
void mip_map1(int min_x, int min_y, int max_x, int max_y) {
    
    for(int y = min_y; y < max_y; y += 2) {
        for(int x = min_x; x < max_x; x += 2) {
            u32 output_idx = get_mip_voxelmap_idx(x, y);

            mip_columns(
                x, y,
                &mip_columns_header_data[output_idx],
                &mip_columns_colors_data[output_idx],
                &mip_columns_runs_data[output_idx],
                &mip_columns_bitmaps_data[output_idx]
            );
        }
    }
}

void mip_map2(int min_x, int min_y, int max_x, int max_y) {  
    for(int y = min_y; y < max_y; y += 4) {
        for(int x = min_x; x < max_x; x += 4) {
            u32 output_idx = get_mip2_voxelmap_idx(x, y);
            mip2_columns(
                x, y,
                &mip2_columns_header_data[output_idx],
                &mip2_columns_colors_data[output_idx],
                &mip2_columns_runs_data[output_idx],
                &mip2_columns_bitmaps_data[output_idx]
            );
        }
    }
}

void mark_chunk_dirty(int global_x, int global_y) {
    int chunk_x = global_x / CHUNK_SIZE;
    int chunk_y = global_y / CHUNK_SIZE;
    chunks[chunk_y*CHUNKS_PER_AXIS+chunk_x].dirty_voxels = 0b111;
}

void prepare_chunk(int global_x, int global_y, int mip_level, int dummy_arg) {
    // DONT BOTHER WITH MIP LEVEL 2
    if(mip_level == 2) { return; }

    int chunk_x = global_x / CHUNK_SIZE;
    int chunk_y = global_y / CHUNK_SIZE;
    chunk_info* chunk = &chunks[chunk_y*CHUNKS_PER_AXIS+chunk_x];
    if (!(chunk->dirty_voxels & (1<<mip_level))) {
        // already prepared
        return;
    }

    if(mip_level == 0) {
        // tweak and light full level
        if(chunk->toplevel_chunk_tweaked == 0) {
            //tweak_column_colors(global_x, global_y, global_x+CHUNK_SIZE, global_y+CHUNK_SIZE);
            chunk->toplevel_chunk_tweaked = 1;
        }
        light_map_mip0(global_x, global_y, global_x+CHUNK_SIZE, global_y+CHUNK_SIZE);

        // mip1 and mip2 should be regenerated
        // TODO: write lighting functions for the lower level mip maps 
        chunk->dirty_voxels = 0b110;
        chunk->mip_loaded |= 0b001;
        //prepare_chunk(global_x, global_y, 1);
        //prepare_chunk(global_x, global_y, 2);

    } else if (mip_level == 1) {
        mip_map1(global_x, global_y, global_x+CHUNK_SIZE, global_y+CHUNK_SIZE);
        //tweak_mip_column_colors(global_x, global_y, global_x+CHUNK_SIZE, global_y+CHUNK_SIZE);
        light_map_mip1(global_x, global_y, global_x+CHUNK_SIZE, global_y+CHUNK_SIZE);
        chunk->dirty_voxels &= 0b101;
        chunk->mip_loaded |= 0b010;

    } else {
        // if mip level 1 isn't generated
        // it needs to be :(
        if(chunk->dirty_voxels & 0b010) {
            // TODO: is this right?  we CANNOT generate mip2 chunks without the mip1 chunks, and higher level logic should handle it
            return;
            // wew lad
            // printf("Bailing out :(\n");
            // prepare_chunk(global_x, global_y, 1, -1);
            // we can't generate at this point
            // just bail out until level 1 is done
            //return;
        }
        mip_map2(global_x, global_y, global_x+CHUNK_SIZE, global_y+CHUNK_SIZE);
        //tweak_mip2_column_colors(global_x, global_y, global_x+CHUNK_SIZE, global_y+CHUNK_SIZE);
        light_map_mip2(global_x, global_y, global_x+CHUNK_SIZE, global_y+CHUNK_SIZE);
        chunk->dirty_voxels &= 0b011;
        chunk->mip_loaded |= 0b100;
    }
}

// global x,y coordinates of a 128x128x256 chunk

int required_nearest_lod_for_chunk(int pos_x, int pos_y, int global_x, int global_y) {
    int x1 = global_x;
    int x2 = global_x+CHUNK_SIZE;
    int y1 = global_y;
    int y2 = global_y+CHUNK_SIZE;
    
    int dx1 = x1-pos_x;
    int dx2 = x2-pos_x;
    int dy1 = y1-pos_y;
    int dy2 = y1-pos_y;

    f32 dist1 = sqrtf(dx1*dx1+dy1*dy1);
    f32 dist2 = sqrtf(dx2*dx2+dy1*dy1);
    f32 dist3 = sqrtf(dx1*dx1+dy2*dy2);
    f32 dist4 = sqrtf(dx1*dx1+dy2*dy2);

    f32 min_dist = min(min(dist1, dist2), min(dist3, dist4));

    int in_mip2_range = (min_dist >= MAX_MIP1_DIST);
    int in_mip1_range = (min_dist >= MAX_MIP0_DIST);
    if(in_mip2_range) {
        return 1;
    } else if (in_mip1_range) {
        return 0;
    } else {
        return 0;
    }
}

int required_furthest_lod_for_chunk(int pos_x, int pos_y, int global_x, int global_y) {
    int x1 = global_x;
    int x2 = global_x+CHUNK_SIZE;
    int y1 = global_y;
    int y2 = global_y+CHUNK_SIZE;
    
    int dx1 = x1-pos_x;
    int dx2 = x2-pos_x;
    int dy1 = y1-pos_y;
    int dy2 = y1-pos_y;

    f32 dist1 = sqrtf(dx1*dx1+dy1*dy1);
    f32 dist2 = sqrtf(dx2*dx2+dy1*dy1);
    f32 dist3 = sqrtf(dx1*dx1+dy2*dy2);
    f32 dist4 = sqrtf(dx1*dx1+dy2*dy2);

    f32 max_dist = max(max(dist1, dist2), max(dist3, dist4));

    int in_mip2_range = (max_dist >= MAX_MIP1_DIST);
    int in_mip1_range = (max_dist >= MAX_MIP0_DIST);
    if(in_mip2_range) {
        return 2;
    } else if (in_mip1_range) {
        return 2;
    } else {
        return 1;
    }
}


thread_params prepare_chunk_params[NUM_CHUNK_THREADS] = {
    {.finished=1}, {.finished=1}, {.finished=1}, {.finished=1}, 
    {.finished=1}, {.finished=1}, {.finished=1}, {.finished=1}, 
    //{.finished=1}, {.finished=1}, {.finished=1}, {.finished=1}, 
    //{.finished=1}, {.finished=1}, {.finished=1}, {.finished=1}
};

job_pool prepare_chunk_jobs = {
    .num_jobs = NUM_CHUNK_THREADS,
    .parms = prepare_chunk_params,
    .func = prepare_chunk_wrapper,
    .raw_func = prepare_chunk
};


int launch_job_on_free_thread(thread_pool* tp, job_pool* rp, thread_params params) {
    for(int i = 0; i < NUM_CHUNK_THREADS; i++) {
        if(prepare_chunk_params[i].finished) { 
            prepare_chunk_params[i] = params;
            prepare_chunk_params[i].finished = 0;
            thread_pool_add_work(tp, rp->func, (void*)&prepare_chunk_params[i]);
            return i;
        }
    }
    return -1;
}

// finished 1, working 0
// finished 0, working 1 -> working
// finished 1, working 0 -> done

void prepare_chunks_from_pos(f32 pos_x, f32 pos_y, int async) {
    // up to 384 from player is mip0
    // up

    int gathered_jobs = 0;
    int no_free_threads_for_jobs = 0;

    int currently_working_jobs[NUM_CHUNK_THREADS][3];
    int num_working_jobs = 0;
    for(int i = 0; i < NUM_CHUNK_THREADS; i++) {

        if(prepare_chunk_params->finished) { continue; }

        int global_x = prepare_chunk_params[i].min_x;
        int global_y = prepare_chunk_params[i].min_y;
        int lod_generating = prepare_chunk_params[i].max_x;
        currently_working_jobs[num_working_jobs][0] = global_x;
        currently_working_jobs[num_working_jobs][1] = global_y;
        currently_working_jobs[num_working_jobs][2] = lod_generating;
    }

    for(int y = 0; y < CHUNKS_PER_AXIS; y++) {
        for(int x = 0; x < CHUNKS_PER_AXIS; x++) {
            int global_x = x*CHUNK_SIZE;
            int global_y = y*CHUNK_SIZE;
            int nearest_lod = required_nearest_lod_for_chunk(pos_x, pos_y, global_x, global_y);

            // if 0 generate 0 1 2
            // if 1 generate 0 1 2 
            // if 2 generate 1 2
            chunk_info* chk = &chunks[y*CHUNKS_PER_AXIS+x];
            int mip0_dirty = chk->dirty_voxels&0b001;
            int mip1_dirty = chk->dirty_voxels&0b010;
            int mip2_dirty = chk->dirty_voxels&0b100;
            
            int tbl[3][4] = {
                {3, 0, 1, 2}, // lod 0 -> generate 0,1
                {3, 1, 0, 2}, // lod 1 -> generate 1,0,2 (2 requires 1)
                {2, 1, 1, -1} // lod 2 -> generate 2,1 (2 requires 1)
            };

            int num_lods = tbl[nearest_lod][0];
            for(int i = 0; i < num_lods; i++) {
                int required_lod = tbl[nearest_lod][i+1];
                if((chk->dirty_voxels & (1<<required_lod)) == 0) {
                    continue;
                }
                if(required_lod == 2 && (chk->dirty_voxels & 0b010)) {
                    // cannot generate LOD2 with LOD1 not generated
                }
                // check if this chunk is already being process for this lod
                int already_processing = 0;
                for(int j = 0; j < num_working_jobs; j++) {
                    if ((currently_working_jobs[j][0] == global_x) &&
                        (currently_working_jobs[j][1] == global_y) &&
                        (currently_working_jobs[j][2] == required_lod)) {
                        already_processing = 1;
                    }
                }
                if(already_processing) {
                    continue;
                }

                thread_params param = {
                    .finished=0, .min_x = global_x, .min_y = global_y, .max_x = required_lod, .max_y = -1 // last arg is dummy
                };

                int launched = launch_job_on_free_thread(
                        map_thread_pool, &prepare_chunk_jobs, param
                );
                if(launched != -1) { 
                    gathered_jobs++;
                    //printf("Launched on thread %i\n", launched);
                } else {
                    no_free_threads_for_jobs++;
                }
            }

        }
    }
    //if(gathered_jobs) {
    //    printf("Launched %i jobs\n", gathered_jobs);
    //}
    //if(no_free_threads_for_jobs) {
    //    printf("Couldn't launch %i jobs\n", no_free_threads_for_jobs);
    //}
    if(!async) {
        //while(1) {
        //   top_of_wait_loop:;
        //    for(int i = 0; i < NUM_CHUNK_THREADS; i++) {
        //        if(prepare_chunk_params[i].finished == 0) { goto top_of_wait_loop; }
        //    }
        //    break;
        //}
       
        wait_for_job_pool_to_finish(&prepare_chunk_jobs);
    }
    //printf("prepare done!\n");
}