#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "types.h"

#define DAY_BACKGROUND_COLOR 0xFFBBAE67
#define NIGHT_BACKGROUND_COLOR 0xFF000000//0xFF632D0F
#define DAY_AMB_LIGHT_FACTOR .7
#define NIGHT_AMB_LIGHT_FACTOR .05
//2047
//2047
#define DAY_MAX_Z 512
#define NIGHT_MAX_Z 512
#define NO_FOG_MAX_Z 2048
//2048

#define PI 3.14159265359

#define PROFILER 0
#define THREADING 1
#include "selectable_profiler.c"
#include <SDL2/SDL.h>


#include "depth.c"
#include "color.c"

#include "utils.h"

u32 background_color = DAY_BACKGROUND_COLOR;
f32 amb_light_factor = .4;


u32 render_size = 0;



#define OUTPUT_WIDTH  1920
//1920
#define OUTPUT_HEIGHT 1080
//1080



static u32 get_swizzled_map_idx(s32 x, s32 y) {
    // yyyyyyyy xxxxxxxx yy xx

    // 4x4 tiles for color
    //return m2D_e_magicbits(x & 1023, y & 1023);
    
    //
    y &= 1023;
    x &= 1023;
    u32 low_y = (y & 0b11)<<2;
    u32 low_x = x & 0b11;
    u32 high_y = ((y>>2)<<12);
    u32 high_x = ((x>>2)<<4);
    return low_y | low_x | high_y | high_x;

    //return (((y&1023)<<10) | (x&1023));
}



static void swizzle_array(u32* arr) {
    u32* tmp = malloc(1024*1024*4);
    for(int y = 0; y < 1024; y++) {
        for(int x = 0; x < 1024; x++) {
            int src_idx = (y<<10) | x;
            tmp[get_swizzled_map_idx(x, y)] = arr[src_idx];
        }
    }
    memcpy(arr, tmp, 1024*1024*4);
    free(tmp);
}


f32 magnitude_vector(f32 x, f32 y, f32 z) {
    return sqrtf((x*x)+(y*y)+(z*z));
}

static int filter_heightmap = 0;
static int blend_columns = 0;
static int blend_normals = 1;

f32 lerp(f32 a, f32 t, f32 b);

#include "voxelmap.h"

static __m256i get_swizzled_map_idx_256(__m256i x, __m256i y) {


    // swizzled
    __m256i wrap_mask = _mm256_set1_epi32(1023);
    
    __m256i two_bit_mask = _mm256_set1_epi32(0b11);

    __m256i wrapped_x = _mm256_and_si256(x, wrap_mask);
    __m256i wrapped_y = _mm256_and_si256(y, wrap_mask);


    //return _mm256_add_epi32(_mm256_slli_epi32((__m256i)wrapped_y, 10), (__m256i)wrapped_x);
    // swizzled stuff here
    __m256i low_x = _mm256_and_si256(wrapped_x, two_bit_mask);
    __m256i low_y = _mm256_slli_epi32(_mm256_and_si256(wrapped_y, two_bit_mask), 2);
    __m256i high_x = _mm256_slli_epi32(_mm256_srli_epi32(wrapped_x, 2), 4);
    __m256i high_y = _mm256_slli_epi32(_mm256_srli_epi32(wrapped_y, 2), 12);

    return _mm256_add_epi32(_mm256_add_epi32(low_x, low_y), _mm256_add_epi32(high_x, high_y));

}

f32 pos_x = 0.0;
f32 pos_y = 0.0;
f32 dir_x = 1;
f32 dir_y = 0.0;
f32 plane_x = 0.0;
f32 plane_y = -1.00;
static f32 height = 200.0;

double fabs(double x);
f32 sqrtf(f32 x);
f32 atan2f(f32 y, f32 x);
f32 sinf(f32 x);
f32 cosf(f32 x);
double asin(double x);

f32 dt;
f32 pitch_ang = 0;
f32 sun_ang = 0;

f32 roll = 0;//-1.57;



f32 forwardbackward = 0;
f32 leftright = 0;
f32 updown = 0;
f32 lookupdown = 0;
f32 rollleftright = 0;

typedef enum {
    VIEW_STANDARD = 0,
    VIEW_DEPTH = 1,
    VIEW_NORMALS = 2
} view_modes;

char* view_mode_strs[] = {
    "standard",
    "depth",
    "normals"
};

typedef enum {
    NO_LIGHTING = 0,
    FANCY_LIGHTING = 1,
    SIDE_LIGHTING = 2,
} lighting_modes;

char* lighting_mode_strs[] = {
    "none",
    "fancy",
    "side"
};

typedef enum {
    NO_FOG = 0,
    FANCY_FOG = 1,
    CONSTANT_DEPTH_FOG = 2
} fog_modes;

f32 total_time;
static int frame = 0;
static int vector = 0;//1;
static int fogmode = FANCY_FOG;
static int lighting = FANCY_LIGHTING;
static int view = VIEW_STANDARD;
static int gravmode = 0;

static int double_pixels = 1;

int render_width = OUTPUT_WIDTH/2;   
int render_height = OUTPUT_HEIGHT/2;
//const int render_width = OUTPUT_WIDTH;
//const int render_height = OUTPUT_HEIGHT;
//#endif 

static int swizzled = 0;
static int setup_render_size = 0;


f32 max_z = DAY_MAX_Z;

int cur_map = 0;
static int map_loaded = 0;

void next_map() {
    load_map(cur_map++);
}




void handle_keyup(SDL_KeyboardEvent key) {
    switch(key.keysym.scancode) {
        case SDL_SCANCODE_Q:
            rollleftright = 0;
            break;
        case SDL_SCANCODE_E:
            rollleftright = 0;
            break;

        case SDL_SCANCODE_UP:
            forwardbackward = 0;
            break;
        case SDL_SCANCODE_DOWN:
            forwardbackward = 0;
            break;
        case SDL_SCANCODE_LEFT: 
            leftright = 0;
            break;
        case SDL_SCANCODE_RIGHT: 
            leftright = 0;
            break;
        case SDL_SCANCODE_Z:
            updown = 0;
            break;
        case SDL_SCANCODE_X:
            updown = 0;
            break;
        case SDL_SCANCODE_A:
            lookupdown = 0;
            break;
        case SDL_SCANCODE_S:
            lookupdown = 0;
            break;
        case SDL_SCANCODE_L:
            lighting++;
            if(lighting > 2) { lighting = 0; }
            //lighting = !lighting;
            break;
        case SDL_SCANCODE_V:
            view++;
            if(view > 2) { view = 0;}
            break;
        case SDL_SCANCODE_F:
            fogmode++;
            if(fogmode > 2) {
                fogmode = 0;
             }
            break;
        case SDL_SCANCODE_G:
            gravmode = !gravmode;
            break;
        case SDL_SCANCODE_M:
            //filter_heightmap = !filter_heightmap;
            //load_cur_map();
            break;
        case SDL_SCANCODE_B:
            //blend_columns = !blend_columns;
            //blend_normals = !blend_normals;
            //load_cur_map();
            break;
        case SDL_SCANCODE_N:
            next_map();
            break;
        case SDL_SCANCODE_D: do {
            if(double_pixels == 1) {
                render_width = OUTPUT_WIDTH/4;
                render_height = OUTPUT_HEIGHT/4;
                double_pixels = 2;
            } else if (double_pixels == 2) {
                render_width = OUTPUT_WIDTH;
                render_height = OUTPUT_HEIGHT;
                double_pixels = 0;
            } else {
                render_width = OUTPUT_WIDTH/2;
                render_height = OUTPUT_HEIGHT/2;
                double_pixels = 1;
            }
            setup_render_size = 0;
        } while(0);
        break;
          
    }
}

void handle_keydown(SDL_KeyboardEvent key) {

    switch(key.keysym.scancode) {
        case SDL_SCANCODE_Q:
            rollleftright = +1.0;
            break;
        case SDL_SCANCODE_E:
            rollleftright = -1.0;
            break;

        case SDL_SCANCODE_UP:
            forwardbackward = .4;
            break;
        case SDL_SCANCODE_DOWN:
            forwardbackward = -.4;
            break;
        case SDL_SCANCODE_LEFT: 
            leftright = +.7;
            break;
        case SDL_SCANCODE_RIGHT: 
            leftright = -.7;
            break;
        case SDL_SCANCODE_Z:
            updown = -1.0;
            break;
        case SDL_SCANCODE_X:
            updown = +1.0;
            break;
        case SDL_SCANCODE_A:
            lookupdown = +1.0;
            break;
        case SDL_SCANCODE_S:
            lookupdown = -1.0;
            break;
    }
}


int falling = 1;

void handle_input(f32 dt) {

    int head_margin = 1;
    int knee_height = 2;
    int eye_height = 8;

    if(gravmode && falling) {
        int collide_z = 0;
        f32 contact_point = (255.0-height)+eye_height;  
        f32 new_world_pos_z = contact_point+.05*dt;
           
        for(int x = pos_x-1; x < pos_x+2; x++) {
            for(int y = pos_y-1; y < pos_y+2; y++) {
                for(int h = new_world_pos_z; h < new_world_pos_z+1; h++) {

                    if(voxel_is_solid(x, y, h) || h >= 63) {
                        falling = 0;
                        break;
                    } else {
                        height = 255.0-(h-eye_height);
                    }

                }
            }
        }
    }

    if(leftright) {
        f32 rot_speed = dt * leftright * .9;

        f32 old_dir_x = dir_x;
        dir_x = dir_x * cos(rot_speed) - dir_y * sin(rot_speed);
        dir_y = old_dir_x * sin(rot_speed) + dir_y * cos(rot_speed);
        f32 old_plane_x = plane_x;
        plane_x = plane_x * cos(rot_speed) - plane_y * sin(rot_speed);
        plane_y = old_plane_x * sin(rot_speed) + plane_y * cos(rot_speed);
    }

    //if(lookupdown) {
        pitch_ang += dt*0.017*5 * lookupdown;// -= dt*400;
    //}
    //if(lookdown) {
    //    pitch_ang -= dt*0.017*5;
    //    //pitch += dt*400;
    //}

    if(forwardbackward) {
        f32 new_pos_x = pos_x + dir_x * forwardbackward * dt * 100;
        f32 new_pos_y = pos_y + dir_y * forwardbackward * dt * 100;
        pos_x = new_pos_x;
        pos_y = new_pos_y;
        //check if we collide based on height (aabb basically)
        int collide_x = 0;
        if ((s32)new_pos_x != pos_x) {
            int world_height = (s32)(255.0-height) + eye_height;

            int collide_ground_highest_point = 64;


            for(int x = new_pos_x-1; x < new_pos_x+2; x++) {
                for(int y = pos_y-1; y < pos_y+2; y++) {
                    if(voxel_is_solid(x, y, world_height)) {
                        collide_ground_highest_point = min(collide_ground_highest_point, world_height);
                        if(voxel_is_solid(x, y, world_height-1)) {
                            // check if everything up to eye height-head_margin is clear
                            collide_x = 1;
                            break;
                        }
                        for(int h = world_height-knee_height; h >= world_height-(eye_height+head_margin); h--) {
                            if(voxel_is_solid(x, y, h)) {
                                collide_x = 1;
                                break;
                            }
                        }
                    }
                }
            }
            
            if(!collide_x) {
                pos_x = new_pos_x;
                if(collide_ground_highest_point != -1) {
                    //s32 valid_z_pos = (collide_ground_highest_point-1)-eye_height;
                    //height = 255.0-valid_z_pos;
                }
            } else {
                printf("X COLLISION!\n");
            }
        }

        int collide_y = 0;
        if ((s32)new_pos_y != pos_y) {
            int world_height = (s32)(255.0-height) + eye_height;

            int collide_ground_highest_point = 64;


            for(int x = pos_x-1; x < pos_x+2; x++) {
                for(int y = pos_y-1; y < new_pos_y+2; y++) {
                    if(voxel_is_solid(x, y, world_height)) {
                        collide_ground_highest_point = min(collide_ground_highest_point, world_height);
                        if(voxel_is_solid(x, y, world_height-1)) {
                            // check if everything up to eye height-head_margin is clear
                            collide_y = 1;
                            break;
                        }
                        for(int h = world_height-knee_height; h >= world_height-(eye_height+head_margin); h--) {
                            if(voxel_is_solid(x, y, h)) {
                                collide_y = 1;
                                break;
                            }
                        }
                    }
                }
            }
            
            if(!collide_y) {
                pos_y = new_pos_y;
                if(collide_ground_highest_point != -1) {
                    //s32 valid_z_pos = (collide_ground_highest_point-1)-eye_height;
                    //height = 255.0-valid_z_pos;
                }
            } else {
                printf("X COLLISION!\n");
            }
        }

        
    }

    if(updown) {
        height += dt*updown*50;
    }
    if(rollleftright) {
        roll -= rollleftright*dt*1.2;
    } else {
        // rollback towards zero
        if(roll > 0) {
            roll -= dt*.3;
        } else if (roll < 0) {
            roll += dt*.3;
        }
        if(fabs(roll) <= dt*.3) {
            roll = 0;
        }
    }
    //height += dt*50;
    //height -= dt*50;
    
    //u32 cell_height = depthmap_u32s[get_swizzled_map_idx((s32)pos_x, (s32)pos_y)]+10;
    //if(height < cell_height) { height = cell_height; }

}






int mouse_is_down = 0;

//void handle_mouse_down(int mouse_x, int mouse_y) {
//    down_mouse_x = mouse_x;
//    down_mouse_y = mouse_y;
//    forwardbackward = 3;
//}


void handle_mouse_up() {
    if(mouse_is_down) {
        leftright = 0;
        updown = 0;
        mouse_is_down = 0;
        forwardbackward = 0;
        rollleftright = 0;
    }
}

static u32* inter_buffer = NULL;
static u32* base_inter_buffer = NULL;
//static f32* z_buffer = NULL;
static u8* occlusion_buffer = NULL;
static u8* base_occlusion_buffer = NULL;



int right_mouse_down = 0;
int cur_mouse_x, cur_mouse_y;

void update_mouse_pos(int mouse_x, int mouse_y) {
    cur_mouse_x = mouse_x;
    cur_mouse_y = mouse_y;
}

void handle_right_mouse() {
    right_mouse_down = 1;
}

void handle_right_mouse_up() {
    if(right_mouse_down) {
        right_mouse_down = 0;

    }
}
int middle_mouse_down = 0;
void handle_middle_mouse() {
    middle_mouse_down = 1;
}

void handle_middle_mouse_up() {
    if(middle_mouse_down) {
        middle_mouse_down = 0;

    }
}

f32 pitch_ang_to_pitch(f32 pitchang) {
    return (sin(pitchang)*render_size)+(render_size/2);
}

void handle_mouse_click() {
    mouse_is_down = 1;
    forwardbackward = .8;
    s32 centerX = OUTPUT_WIDTH/2;
    s32 centerY = OUTPUT_HEIGHT/2;
    s32 dx = centerX - cur_mouse_x;
    s32 dy = centerY - cur_mouse_y;

    leftright = (f32)dx / OUTPUT_WIDTH * 2;
    roll = -(f32)dx / OUTPUT_WIDTH * 2;
    f32 desired_pitch = ((render_size/2)-200) + (f32)dy / (render_size/2) * (render_size/2);
    pitch_ang = asin((desired_pitch-(render_size/2))/render_size);
    updown   = ((f32)dy / OUTPUT_HEIGHT) * 10;

}

#include "vc.c"

double fmod(double x, double y);


//#define DIRECTIONAL_LIGHTING


static u32* pixels = NULL; //[OUTPUT_WIDTH * OUTPUT_HEIGHT];



__m256 magnitude_vector_256(__m256 x, __m256 y, __m256 z) {
    return _mm256_sqrt_ps(
        _mm256_add_ps(
            _mm256_mul_ps(x, x), 
            _mm256_add_ps(
                _mm256_mul_ps(y, y),
                _mm256_mul_ps(z, z)
            )
        )
    );
}
__m256 reciprocal_magnitude_vector_256(__m256 x, __m256 y, __m256 z) {
    return _mm256_rsqrt_ps(
        _mm256_add_ps(
            _mm256_mul_ps(x, x), 
            _mm256_add_ps(
                _mm256_mul_ps(y, y),
                _mm256_mul_ps(z, z)
            )
        )
    );
}


f32 lerp(f32 a, f32 t, f32 b) {
    return ((1.0-t)*a)+(t*b);
}

f32 lerp_color(u32 a, f32 t, u32 b) {
    u8 ra  = (a&0xFF); 
    u8 ga = ((a>>8)&0xFF);
    u8 ba = ((a>>16)&0xFF);
    u8 rb  = (b&0xFF); 
    u8 gb = ((b>>8)&0xFF);
    u8 bb = ((b>>16)&0xFF);
    u8 lr = lerp(ra, t, rb);
    u8 lg = lerp(ga, t, gb);
    u8 lb = lerp(ba, t, bb);
    return (0xFF<<24)|(lb<<16)|(lg<<8)|lr;
}

__m256 lerp256(__m256 a, __m256 t, __m256 b) {
    const __m256 one_vec = _mm256_set1_ps(1.0);

    return _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(one_vec,t), a),
                         _mm256_mul_ps(t,b));
}


s32 horizontal_min_256_epi32(__v8si v) {
    __v4si i = (__v4si)_mm256_extractf128_si256( (__m256i)v, 1 );\
    // compare lower and upper halves, get min(0,4), min(1,5), min(2,6), min(3,7)
    i = (__v4si)_mm_min_epi32( (__m128i)i, _mm256_castsi256_si128( (__m256i)v ) ); 
     // compare lower and upper 64-bit halves, get min(min(0,4), min(2,6)), min(min(1,5), min(3,7))
    i = (__v4si)_mm_min_epi32( (__m128i)i, _mm_shuffle_epi32( (__m128i)i, 0b00001110 ) ); 
    return min(i[0], i[1]);
}

s32 horizontal_max_256_epi32(__v8si v) {
    __v4si i = (__v4si)_mm256_extractf128_si256( (__m256i)v, 1 );\
    // compare lower and upper halves, get min(0,4), min(1,5), min(2,6), min(3,7)
    i = (__v4si)_mm_max_epi32( (__m128i)i, _mm256_castsi256_si128( (__m256i)v ) ); 
     // compare lower and upper 64-bit halves, get min(min(0,4), min(2,6)), min(min(1,5), min(3,7))
    i = (__v4si)_mm_max_epi32( (__m128i)i, _mm_shuffle_epi32( (__m128i)i, 0b00001110 ) ); 
    return max(i[0], i[1]);
}


__m256 dot_256(__m256 x1, __m256 y1, __m256 z1,
                __m256 x2, __m256 y2, __m256 z2) {
    return _mm256_add_ps(
        _mm256_mul_ps(x1, x2), 
        _mm256_add_ps(
            _mm256_mul_ps(y1, y2),
            _mm256_mul_ps(z1, z2)
        )
    );
}


__m256 abs_ps(__m256 x) {
    __m256 sign_mask = _mm256_set1_ps(-0.0); // -0.f = 1 << 31
    return _mm256_andnot_ps(sign_mask, x);
}



f32 rotate_x(f32 angle, s32 input_x, s32 input_y) {
    
    f32 sinroll = sinf(roll);
    f32 cosroll = cosf(roll);            
    f32 yy = (s32)(input_y-render_height/2);
    f32 xx = (s32)(input_x-render_width/2);
    f32 temp_xx = xx * cosroll - yy * sinroll;
    f32 temp_yy = xx * sinroll + yy*cosroll; 
    return temp_xx + render_width/2;
}

f32 rotate_y(f32 angle, s32 input_x, s32 input_y) {
    
    f32 sinroll = sinf(roll);
    f32 cosroll = cosf(roll);            
    f32 yy = (s32)(input_y-render_height/2);
    f32 xx = (s32)(input_x-render_width/2);
    f32 temp_xx = xx * cosroll - yy * sinroll;
    f32 temp_yy = xx * sinroll + yy*cosroll; 
    xx = temp_xx + render_width/2;
    return temp_yy + render_height/2;
}




#define LOG2(X) ((unsigned) (8*sizeof (unsigned long long) - __builtin_clzll((X)) - 1))


int log2_fast(u32 x) {
    return 31 - __builtin_clz(x);
}

static u32 *world_pos_buffer = NULL;
static u32 *base_world_pos_buffer = NULL;

static f32* norm_buffer = NULL;
static f32* base_norm_buffer = NULL;

static u32* albedo_buffer = NULL;
static u32* base_albedo_buffer = NULL;
u32 num_render_size_bits; 

__m256i fb_swizzle_256(__m256i xs, __m256i ys) {
    __m256i low_x_mask = _mm256_set1_epi32(0b111);
    __m256i low_xs = _mm256_and_si256(xs, low_x_mask);
    __m256i high_xs = _mm256_srli_epi32(xs, 3);
    __m256i high_x_shift_vec = _mm256_set1_epi32(num_render_size_bits+3);
    __m256i high_xs_shifted = _mm256_sllv_epi32(high_xs, high_x_shift_vec);
    __m256i ys_shifted = _mm256_slli_epi32(ys, 3);
    return _mm256_or_si256(high_xs_shifted, _mm256_or_si256(ys_shifted, low_xs));
}

s32 fb_swizzle(s32 x, s32 y) {
    
    // we want 3 lower bits of x to be contiguous
    u32 low_x = x & 0b111;
    u32 high_x = x >> 3;

    return (high_x<<(num_render_size_bits+3))|(y<<3)|low_x;
    
    //const int row_size = 8;
    //const int row_size_shift = 3;
    //const int row_mask = 0b111;
    //s32 in_tile_x = x & row_mask;
    //s32 out_tile_x = x>>row_size_shift;
    //s32 in_tile_row_offset = (y*row_size)+in_tile_x;
    //s32 out_tile_offset = out_tile_x*(render_size*row_size);
    //return out_tile_offset+in_tile_row_offset+in_tile_x;
}


void set_occlusion_bit(u32 x, u32 y, u8 bit) {
    // bit logic
    u32 bit_pos_x = x & 0b111;
    u32 byte_x = x >> 3;
    u32 byte_offset = (byte_x<<num_render_size_bits)|y;
    occlusion_buffer[byte_offset] |= (bit << bit_pos_x);
    u32 fb_idx = fb_swizzle(x, y);
    //z_buffer[fb_idx] = inv_z;
}


u8 get_occlusion_bit(u32 x, u32 y) {
    // bit logic
    u32 bit_pos_x = x & 0b111;
    u32 byte_x = x >> 3;
    u32 byte_offset = (byte_x<<num_render_size_bits)|y;

    u8 byte = occlusion_buffer[byte_offset];
    return byte & (1<<bit_pos_x);
}

u8 get_occlusion_byte(u32 x, u32 y) {
    u32 byte_x = x >> 3;
    u32 byte_offset = (byte_x<<num_render_size_bits)|y;
    return occlusion_buffer[byte_offset];
}

#define VECTOR

#include "thread_pool.h"


void render_vector(s32 min_x, s32 min_y, s32 max_x, s32 max_y);
void raycast_scalar(s32 min_x, s32 min_y, s32 max_x, s32 max_y);
void fill_empty_entries(s32 min_x, s32 min_y, s32 max_x, s32 max_y);
void rotate_light_and_blend(s32 min_x, s32 min_y, s32 max_x, s32 max_y);

typedef struct {
    s32 min_x, min_y;
    s32 max_x, max_y;
    volatile uint64_t finished;
} thread_params;

thread_pool_function(raycast_scalar_wrapper, arg_var)
{
	thread_params* tp = (thread_params*)arg_var;
    raycast_scalar(tp->min_x, tp->min_y, tp->max_x, tp->max_y);

	InterlockedIncrement64(&tp->finished);
}

thread_pool_function(fill_empty_entries_wrapper, arg_var)
{
	thread_params* tp = (thread_params*)arg_var;
    fill_empty_entries(tp->min_x, tp->min_y, tp->max_x, tp->max_y);

	InterlockedIncrement64(&tp->finished);
}

thread_pool_function(rotate_light_and_blend_wrapper, arg_var) 
{
	thread_params* tp = (thread_params*)arg_var;
    rotate_light_and_blend(tp->min_x, tp->min_y, tp->max_x, tp->max_y);

	InterlockedIncrement64(&tp->finished);

}


f32 scale_height, inv_scale_height;

thread_pool* pool;

u16 mix_15bpp_samples(u16 sample_a, u16 sample_b) {
    u16 ar = (sample_a&0b11111)<<3;
    u16 ag = ((sample_a>>5)&0b11111)<<3;
    u16 ab = ((sample_a>>10)&0b11111)<<3;
    u16 br = (sample_b&0b11111)<<3;
    u16 bg = ((sample_b>>5)&0b11111)<<3;
    u16 bb = ((sample_b>>10)&0b11111)<<3;
    u16 mr = ((ar+br)>>1)&0xFF;
    u16 mb = ((ag+bg)>>1)&0xFF;
    u16 mg = ((ab+bb)>>1)&0xFF;
    return ((mg>>3)<<10)|((mb>>3)<<5)|(mr>>3);
}   

u16 random_adjust_15bpp_sample(u16 sample_a) {
    u8 r = (sample_a&0b11111)<<3;
    u8 g = ((sample_a>>5)&0b11111)<<3;
    u8 b = ((sample_a>>10)&0b11111)<<3;
    s8 rr =(rand()&3)-1; // 0, 1, 2
    s8 rg = (rand()&3)-1;
    s8 rb = (rand()&3)-1;
    r = (r == 0 && rr == -1) ? r : (r == 0xFF && rr == 1) ? r : r+rr;
    g = (g == 0 && rg == -1) ? g : (g == 0xFF && rg == 1) ? g : g+rg;
    b = (b == 0 && rb == -1) ? b : (b == 0xFF && rb == 1) ? b : b+rb;
    return ((b>>3)<<10)|((g>>3)<<5)|(r>>3);
}   
u32 candidate_idxs[1024*1024];
    //u16 candidate_tall_neighbor_sample[1024*1024];
u16 candidate_short_neighbor_sample[1024*1024];
u16 candidate_color_sample[1024*1024];

#if 0
void blend_tall_columns() {

    int num_candidates = 0;

    //printf("Got %i blend candidates\n", num_candidates);
    //for(int i = 0; i < num_candidates; i++) {
    for(int y = 0; y < 1024; y++) {
        for(int x = 0; x < 1024; x++) {
        u32 idx = get_voxelmap_idx(x, y);//candidate_idxs[i];
        // split in half
        // top is normal
        // bot is blended with the short sample
        column_header* header = &columns_header_data[idx];
        u8 prev_height = header->first_three_runs[1];
        if(prev_height < 20) { continue; }
        if(header->first_three_run_colors[0] == 0) { continue; }
        u8 half_height = prev_height/2;
        u8 top_height = (prev_height-half_height);
        //u16 top_color = candidate_color_sample[i];
        //u16 bot_color = 0b11111<<10; //candidate_short_neighbor_sample[i];//mix_15bpp_samples(top_color, candidate_short_neighbor_sample[i]);
        header->first_three_runs[1] = half_height;
        header->first_three_runs[2] = 0;
        header->first_three_runs[3] = top_height;
        //header->first_three_run_colors[0] = top_color;
        header->first_three_run_colors[1] = random_adjust_15bpp_sample(header->first_three_run_colors[0]);
        header->num_runs = 2;

        }
    }
}
#endif

static int setup_internal_buffer = 0;

void setup_internal_render_buffers() {        
    u32 min_size = (s32) ceilf(sqrtf((render_width*render_width)+(render_width*render_width)));
    render_size = 2;
    while(render_size < min_size) {
        // if render size isn't big enough based on the render width
        // we need to reallocate the internal buffer
        render_size *= 2;
    }

    // if internal buffers have already been allocated
    if(base_inter_buffer != NULL) {
        free(base_inter_buffer);
    }
    if(base_occlusion_buffer != NULL) {
        free(base_occlusion_buffer);
    }

    if(pixels == NULL) {
        pixels = malloc(sizeof(u32)*OUTPUT_WIDTH*OUTPUT_HEIGHT+32);
        while(((intptr_t)pixels)&0b11111) {
            pixels++;
        }
    }

    num_render_size_bits = LOG2(render_size);

    scale_height = ((((16/9)*(.5))/(4/3))*render_size);
    inv_scale_height = 1/scale_height;

    base_inter_buffer = malloc((sizeof(u32)*render_size*render_size)+32);
    base_world_pos_buffer = malloc((sizeof(u32)*render_size*render_size)+32);
    base_norm_buffer = malloc((sizeof(f32)*2*render_size*render_size)+32);
    base_occlusion_buffer = malloc((sizeof(u8)*render_size*render_size/8)+32);
    base_albedo_buffer = malloc((sizeof(u32)*render_size*render_size)+32);
    
    inter_buffer = base_inter_buffer;
    world_pos_buffer = base_world_pos_buffer;
    norm_buffer = base_norm_buffer;
    occlusion_buffer = base_occlusion_buffer;
    albedo_buffer = base_albedo_buffer;

    while(((intptr_t)inter_buffer)&0b11111) {
        inter_buffer++;
    }
    while(((intptr_t)occlusion_buffer)&0b11111) {
        occlusion_buffer++;
    }
    while(((intptr_t)norm_buffer)&0b11111) {
        norm_buffer++;
    }
    while(((intptr_t)world_pos_buffer)&0b11111) {
        world_pos_buffer++;
    }

    while(((intptr_t)albedo_buffer)&0b11111) {
        albedo_buffer++;
    }
    setup_render_size = 1;

    //pitch = render_size/2;
    //pitch_ang = 0;

}

typedef void (*raw_render_func)(s32 min_x, s32 min_y, s32 max_x, s32 max_y);

typedef struct {
    int num_jobs;
    PTP_WORK_CALLBACK func;
    raw_render_func raw_func;
    thread_params *parms;
} render_pool;

void start_render_pool(thread_pool* tp, render_pool* job_pool) {
    for(int i = 0; i < job_pool->num_jobs; i++) {
#if THREADING 
        thread_pool_add_work(tp, job_pool->func, (void*)&job_pool->parms[i]);
#else
        job_pool->raw_func(
            job_pool->parms[i].min_x, 
            job_pool->parms[i].min_y,
            job_pool->parms[i].max_x,
            job_pool->parms[i].max_y
        );
#endif
    }
}
void wait_for_render_pool_to_finish(render_pool* p) {
#if THREADING 
    while(1) {
        top_of_wait_loop:;
        for(int i = 0; i < p->num_jobs; i++) {
            if(p->parms[i].finished == 0) { goto top_of_wait_loop; }
        }
        break;
    }
#else
    return;
#endif
}



__m256 srgb255_to_linear_256(__m256 cols) {

    __m256 one_over_255_vec = _mm256_set1_ps(1.0f / 255.0f);
    __m256 div_255 = _mm256_mul_ps(cols, one_over_255_vec);
    return _mm256_mul_ps(div_255, div_255);
}

__m256i linear_to_srgb_255_256(__m256 cols) {

    return _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_set1_ps(255.0f), _mm256_sqrt_ps(cols)));
}


Olivec_Canvas vc_render(f32 dt) {
    handle_input(dt);   

        
    static int setup_thread_pool = 0;
    if(!setup_thread_pool) {
        pool = thread_pool_create(8);

        setup_thread_pool = 1;
    }



    if(!setup_render_size) {
        setup_internal_render_buffers();
    }

    if(!map_loaded) {
        //get_and_load_vmap_entry_if_necessary(0, 0);
        load_map(cur_map);
        map_loaded = 1;
    }




    Olivec_Canvas oc = olivec_canvas(pixels, OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH);

    //memset(occlusion_buffer, 0, render_size*render_size);
    memset(occlusion_buffer, 0, render_size*render_size/8);


    
    s32 y_buffer = ((render_size-render_height)/2);
    s32 x_buffer = ((render_size-render_width)/2);

    // find bounding box of what needs to be drawn
    // technically we could figure this out per column
    // so that we never fill more than absolutely necessary, but i don't feel like working that out right now

    s32 x1 = rotate_x(roll, 0, 0);
    s32 x2 = rotate_x(roll, render_width-1, 0);
    s32 x3 = rotate_x(roll, 0, render_height-1);
    s32 x4 = rotate_x(roll, render_width-1, render_height-1);
    s32 y1 = rotate_y(roll, 0, 0);
    s32 y2 = rotate_y(roll, render_width-1, 0);
    s32 y3 = rotate_y(roll, 0, render_height-1);
    s32 y4 = rotate_y(roll, render_width-1, render_height-1);

    s32 min_x = min(x1, min(x2, min(x3, x4)))+x_buffer;
    s32 max_x = max(x1, max(x2, max(x3, x4)))+x_buffer+1;
    s32 min_y = min(y1, min(y2, min(y3, y4)))+y_buffer;
    s32 max_y = max(y1, max(y2, max(y3, y4)))+y_buffer; 


    int draw_dx = (max_x - min_x);
    int half_x = ((max_x-min_x)/2);
    int quarter_x = half_x/2;

    // FILL BEFORE RENDER
    {
        #if 1
        u32 undrawn_world_pos = 0b10000000;
        int min_x_aligned = min_x & ~0b11111;
        __m256i undrawn_vec = _mm256_set1_epi32(undrawn_world_pos);
        __m256 undrawn_norm_pt1_vec = _mm256_set1_ps(0);
        __m256i undrawn_albedo_vec = _mm256_set1_epi32(0x00000000);
        //__m256 undrawn_norm_pt2_vec = _mm256_set1_ps(0);
        for(int x = min_x_aligned; x < max_x; x += 8) {
            u32 base_fb_idx = fb_swizzle(x, min_y);
            u32* world_pos_buf_ptr = &world_pos_buffer[base_fb_idx];
            f32* norm_ptr = &norm_buffer[base_fb_idx*2];
            u32* albedo_ptr = &albedo_buffer[base_fb_idx];

            profile_block coverage_fill_empty_entries;
            TimeBlock(coverage_fill_empty_entries, "fill framebuffer");
            for(int y = min_y; y <= max_y; y++) {     
                _mm256_store_si256((__m256i*)world_pos_buf_ptr, undrawn_vec);
                world_pos_buf_ptr += 8;
                _mm256_store_si256((__m256i*)albedo_ptr, undrawn_albedo_vec);
                albedo_ptr += 8;
                _mm256_store_ps((f32*)norm_ptr, undrawn_norm_pt1_vec);
                _mm256_store_ps((f32*)(norm_ptr+8), undrawn_norm_pt1_vec);
                norm_ptr += 16;
            }
            EndCountedTimeBlock(coverage_fill_empty_entries, max_y-min_y);
        }
        #endif
    }

    // RAYCAST: 4 threads 
    {
        thread_params parms[8];
        for(int i = 0; i < 8; i++) {
            parms[i].finished = 0;
            parms[i].min_x = min_x + (draw_dx*i/8);
            parms[i].max_x = parms[i].min_x + (draw_dx/8)+1;
            parms[i].min_y = min_y;
            parms[i].max_y = max_y;
        }
        render_pool rp = {
            .num_jobs = 8,
            .func = raycast_scalar_wrapper,
            .raw_func = raycast_scalar,
            .parms = parms
        };
        start_render_pool(pool, &rp);
        wait_for_render_pool_to_finish(&rp);
    }




    {
    #if 0
        thread_params parms[8];
        for(int i = 0; i < 8; i++) {
            parms[i].finished = 0;
            parms[i].min_x = min_x + (quarter_x*i/2);
            parms[i].max_x = parms[i].min_x+(quarter_x/2);
            parms[i].min_y = min_y;
            parms[i].max_y = max_y;
        }
        render_pool rp = {
            .num_jobs = 8,
            .func = fill_empty_entries_wrapper,
            .raw_func = fill_empty_entries,
            .parms = parms
        };
        start_render_pool(pool, &rp);
        wait_for_render_pool_to_finish(&rp);
    #endif
    }


    int mouse_off_x = (cur_mouse_x - (OUTPUT_WIDTH/2))/(1<<double_pixels);
    int mouse_off_y = (cur_mouse_y - (OUTPUT_HEIGHT/2))/(1<<double_pixels);



    u32 center_fb_idx = fb_swizzle((render_size/2)+mouse_off_x, ((render_size/2)+mouse_off_y));
    u32 combined_world_idx = world_pos_buffer[center_fb_idx];



    u32 screen_center_map_y = (combined_world_idx>>8)&0b11111111111;
    u32 screen_center_map_x = (combined_world_idx>>19)&0b11111111111;
    
    u32 color_slot_in_column = (combined_world_idx & 0b11111111);
    if(color_slot_in_column == 0b10000000) {
        goto pixel_is_undrawn;
    }
    //s32 color_slot_in_column = -1;

    u32 vox_col_idx = get_voxelmap_idx(screen_center_map_x, screen_center_map_y);

    column_header* header = &columns_header_data[vox_col_idx];
    span* runs = &columns_runs_data[vox_col_idx].runs_info[0];
    u32* color_ptr = &columns_colors_data[vox_col_idx].colors[0];
    s32 world_height = 255.0 - height;

    //u16* color_ptr = colors.colors;


    static int last_frame_middle_mouse_down = 0;
    if(middle_mouse_down) {
        if(last_frame_middle_mouse_down == 0) {
            
  

        #if 1
            //s32 color_slot_in_column = get_color_slot_for_world_pos(screen_center_map_x, screen_center_map_y, world_height);
            //*(color_ptr+color_slot_in_column) = 0xFF<<24 | (B469FF);  
            u32 old_col = color_ptr[color_slot_in_column];
            old_col = (0xFFBB9F78);
            u8 old_ao_term = old_col & (0b11111100<<24);
            u8 old_alpha_term = (old_col>>24)&0b11;
            if(old_alpha_term == 3) {
                u32 old_col_without_alpha = 0b00000000111111101111111011111110 & old_col;
                old_col_without_alpha >>= 1;
                
                u32 new_col = old_col_without_alpha | (0b01<<24) | old_ao_term;
                color_ptr[color_slot_in_column] = new_col;
            }
        #endif
        }
        last_frame_middle_mouse_down = 1;
    } else {
        last_frame_middle_mouse_down = 0;
    }
    if (right_mouse_down) {
        
        *(color_ptr+color_slot_in_column) = 0xFFB469FF; 
    #if 0
        int screen_center_map_z = world_height;//get_world_pos_for_color_slot(screen_center_map_x, screen_center_map_y, color_slot_in_column);

        if(voxel_is_solid(screen_center_map_x, screen_center_map_y,  screen_center_map_z)) {
            // it should be a surface
            remove_voxel_at(screen_center_map_x, screen_center_map_y, screen_center_map_z);
        }
    #endif

    }
    pixel_is_undrawn:;
    {


        thread_params parms[8];
        if(double_pixels == 2) {
            // for some reason, the left edge of the screen breaks in this pass with 8 jobs
            // honestly, it runs so much faster that I sort of don't care about using 8 jobs/threads here
            for(int i = 0; i < 4; i++) {
                parms[i].finished = 0;
                parms[i].min_x = (render_width*i/4);
                parms[i].max_x = parms[i].min_x + (render_width/4);
                parms[i].min_y = 0;
                parms[i].max_y = render_height;
            }
            render_pool rp = {
                .num_jobs = 4,
                .func = rotate_light_and_blend_wrapper,
                .raw_func = rotate_light_and_blend,
                .parms = parms
            };
            start_render_pool(pool, &rp);
            wait_for_render_pool_to_finish(&rp);
        } else {
            for(int i = 0; i < 8; i++) {
                parms[i].finished = 0;
                parms[i].min_x = (render_width*i/8);
                parms[i].max_x = parms[i].min_x + (render_width/8);
                parms[i].min_y = 0;
                parms[i].max_y = render_height;
            }
            render_pool rp = {
                .num_jobs = 8,
                .func = rotate_light_and_blend_wrapper,
                .raw_func = rotate_light_and_blend,
                .parms = parms
            };
            start_render_pool(pool, &rp);
            wait_for_render_pool_to_finish(&rp);
        }   
    }
    
    f32 ms = dt*1000;
    f32 fps = 1000/ms;
    //char strbuf[64];
    //sprintf(strbuf, "fps: %i, view: %s, lighting: %s", 
    //        (int)fps, view_mode_strs[view], lighting_mode_strs[lighting]);
    //olivec_text(oc, strbuf, 0, 0, olivec_default_font, 2, 0xFFFFFFFF);
    
    printf("fps; %f\n", fps);
    
    //printf("pitch: %f, render_size: %i, percent pitch: %f\n", pitch, render_size, pitch/render_size);
    total_time += ms;
    frame++;

    sun_ang += dt*0.13;
    if(lighting == SIDE_LIGHTING) {
        sun_ang = 5.5;
    }
    f32 sun_degrees = sun_ang * 57.2958;


    //sun_degrees += 90;
    sun_degrees -= 90;
    while(sun_degrees > 360) {
        sun_degrees -= 360;
    }
    while(sun_degrees < 0) {
        sun_degrees += 360;
    }



    if(!lighting) {
        background_color = DAY_BACKGROUND_COLOR;
        amb_light_factor = 1.0;
    } else if (lighting) { //}== SIDE_LIGHTING) {
        background_color = DAY_BACKGROUND_COLOR;
        amb_light_factor = 0.8;
    } else {

        
        if(sun_degrees >= 0 && sun_degrees < 180) {
            f32 zero_to_one = sun_degrees / 180;
            background_color = lerp_color(NIGHT_BACKGROUND_COLOR, zero_to_one, DAY_BACKGROUND_COLOR);
            max_z = lerp(NIGHT_MAX_Z, sun_degrees/180, DAY_MAX_Z);
            amb_light_factor = lerp(NIGHT_AMB_LIGHT_FACTOR, zero_to_one, DAY_AMB_LIGHT_FACTOR);
        } else {
            f32 zero_to_one = (sun_degrees-180)/180;
            background_color = lerp_color(DAY_BACKGROUND_COLOR, zero_to_one*zero_to_one, NIGHT_BACKGROUND_COLOR);
            f32 sun_degrees_to_one = (sun_degrees-180)/180;
            max_z = lerp(DAY_MAX_Z, sun_degrees_to_one*sun_degrees_to_one, NIGHT_MAX_Z);
            amb_light_factor = lerp(DAY_AMB_LIGHT_FACTOR, zero_to_one*zero_to_one, NIGHT_AMB_LIGHT_FACTOR);
        }
        
    }
    
    if(!fogmode) {
        max_z = NO_FOG_MAX_Z;
    } else {
        max_z = DAY_MAX_Z;
    }
    //printf("LIGHTING MODE %s\n", lighting_mode_strs[lighting]);
    //printf("VIEW MODE: %s\n", view_mode_strs[view]);

    //printf("sun degrees: %f\n", sun_degrees);
    //printf("sun y vector component: %f\n", sinf(sun_ang));
    //printf("amb light factor: %f\n", amb_light_factor);


    //printf("sun angle %f\n", sun_degrees);
    return oc;

}


void rotate_light_and_blend(s32 min_x, s32 min_y, s32 max_x, s32 max_y) {
    // post-process rotation, lighting, fog blending

    profile_block rotate_light_and_blend_block;
    TimeBlock(rotate_light_and_blend_block, "rotate light and blend");
    f32 dx_per_x = rotate_x(roll, 1, 0) - rotate_x(roll, 0, 0);
    f32 dy_per_x = rotate_y(roll, 1, 0) - rotate_y(roll, 0, 0);
    f32 dx_per_y = rotate_x(roll, 0, 1) - rotate_x(roll, 0, 0);
    f32 dy_per_y = rotate_y(roll, 0, 1) - rotate_y(roll, 0, 0);
    __m256 dx_per_x_vec = _mm256_set1_ps(dx_per_x*8);
    __m256 dy_per_x_vec = _mm256_set1_ps(dy_per_x*8);
    __m256 dx_per_y_vec = _mm256_set1_ps(dx_per_y);//*8);
    __m256 dy_per_y_vec = _mm256_set1_ps(dy_per_y);//*8);


    //f32 row_y = rotate_y(roll, min_x, min_y);
    //f32 row_x = rotate_x(roll, min_x, min_y);
    f32 col_y = rotate_y(roll, min_x, min_y);
    f32 col_x = rotate_x(roll, min_x, min_y);

    s32 y_buffer = ((render_size-render_height)/2);
    s32 x_buffer = ((render_size-render_width)/2);
    __m256i y_buffer_vec = _mm256_set1_epi32(y_buffer);
    __m256i x_buffer_vec = _mm256_set1_epi32(x_buffer);

    __m256i undrawn_bit_mask_vec = _mm256_set1_epi32(0b10000000);
    __m256i depth_mask_vec = _mm256_set1_epi32(0xFFFF);
    __m256i low_seven_bits_mask = _mm256_set1_epi32(0b1111111);
    __m256i low_eight_bits_mask = _mm256_set1_epi32(0xFF);
    __m256i low_ten_bits_mask = _mm256_set1_epi32(0b1111111111);
    __m256i low_eleven_bits_mask = _mm256_set1_epi32(0b11111111111);
    __m256i tenth_bit_mask = _mm256_set1_epi32(0b1000000000);
    __m256i eighth_bit_mask = _mm256_set1_epi32(0b10000000);
    __m256i alpha_vec = _mm256_set1_epi32(0xFF<<24);

    __m256 zero_ps_vec = _mm256_set1_ps(0);
    __m256 one_half_ps_vec = _mm256_set1_ps(0.5);
    __m256 one_quarter_ps_vec = _mm256_set1_ps(0.25);
    __m256 one_eighth_ps_vec = _mm256_set1_ps(1/8.0);
    __m256 one_sixteenth_ps_vec = _mm256_set1_ps(1/16.0);
    __m256 one_ps_vec = _mm256_set1_ps(1);
    __m256 two_ps_vec = _mm256_set1_ps(2);

    __m256 render_size_ps_vec = _mm256_set1_ps(render_size);

    __m256 height_vec = _mm256_set1_ps(height);

    f32 fog_r = (background_color&0xFF);
    f32 fog_g = ((background_color>>8)&0xFF);
    f32 fog_b = ((background_color>>16)&0xFF);
    __m256 fog_rs = srgb255_to_linear_256(_mm256_set1_ps(fog_r));
    __m256 fog_gs = srgb255_to_linear_256(_mm256_set1_ps(fog_g));
    __m256 fog_bs = srgb255_to_linear_256(_mm256_set1_ps(fog_b));

    //f32 one_over_max_z_scale = 1/(max_z*32.0);
    //__m256 scale_depth_vec = _mm256_set1_ps(65536.0);
    __m256 one_over_fix_scale_vec = _mm256_set1_ps(1/32.0);
    __m256 one_over_max_z_vec = _mm256_set1_ps(1/max_z);
    __m256 one_over_scale_height_vec = _mm256_set1_ps(1/scale_height);
    __m256 scale_height_vec = _mm256_set1_ps(scale_height);
    
    f32 pitch = pitch_ang_to_pitch(pitch_ang);
    __m256 pitch_vec = _mm256_set1_ps(pitch);

    __m256 dir_x_vec = _mm256_set1_ps(dir_x);
    __m256 dir_y_vec = _mm256_set1_ps(dir_y);
    __m256 plane_x_vec = _mm256_set1_ps(plane_x);
    __m256 plane_y_vec = _mm256_set1_ps(plane_y);
    
    f32 wrap_pos_x = pos_x;
    //while(wrap_pos_x < 0) {
    //    wrap_pos_x += 1024;
    //}
    //while(wrap_pos_x > 1024) {
    //    wrap_pos_x -= 1024;
    //}        
    f32 wrap_pos_y = pos_y;//fmod(pos_y, 1024);
    //while(wrap_pos_y < 0) {
    //    wrap_pos_y += 1024;
    //}
    //while(wrap_pos_y > 1024) {
    //    wrap_pos_y -= 1024;
    //}
    __m256 pos_x_vec = _mm256_set1_ps(wrap_pos_x);
    __m256 pos_y_vec = _mm256_set1_ps(wrap_pos_y);
    __m256 inverse_height_vec = _mm256_set1_ps(255-height);
    __m256 ten_twenty_four_vec = _mm256_set1_ps(1024.0);

    f32 sun_vec_x = 0;
    f32 sun_vec_y = sinf(sun_ang);
    f32 sun_vec_z = cosf(sun_ang);
    __m256 sun_vec_x_vec = _mm256_set1_ps(sun_vec_x);
    __m256 sun_vec_y_vec = _mm256_set1_ps(sun_vec_y);
    __m256 sun_vec_z_vec = _mm256_set1_ps(sun_vec_z);

    __m256i one_vec = _mm256_set1_epi32(1);

    __m256 amb_light_factor_vec = _mm256_set1_ps(amb_light_factor);
    __m256i background_color_vec = _mm256_set1_epi32(background_color);

    for(int base_ox = min_x; base_ox < max_x; base_ox += 8) {
        //f32 yy = row_y;
        //f32 xx = row_x;
        f32 yy = col_y;
        f32 xx = col_x;
        //if(oy > (render_height*.75)) { 
        //    printf("wtf\n");
        //}
    #if 1
        __m256 yy_vec = _mm256_setr_ps(
            yy,            yy+dy_per_x,   yy+dy_per_x*2, yy+dy_per_x*3,
            yy+dy_per_x*4, yy+dy_per_x*5, yy+dy_per_x*6, yy+dy_per_x*7
        );
        __m256 xx_vec = _mm256_setr_ps(
            xx,            xx+dx_per_x,   xx+dx_per_x*2, xx+dx_per_x*3,
            xx+dx_per_x*4, xx+dx_per_x*5, xx+dx_per_x*6, xx+dx_per_x*7
        ); 
        // __m256 yy_vec = _mm256_setr_ps(
        //    yy,            yy+dy_per_y,   yy+dy_per_y*2, yy+dy_per_y*3,
        //    yy+dy_per_y*4, yy+dy_per_y*5, yy+dy_per_y*6, yy+dy_per_y*7
        //);
        //__m256 xx_vec = _mm256_setr_ps(
        //    xx,            xx+dx_per_y,   xx+dx_per_y*2, xx+dx_per_y*3,
        //    xx+dx_per_y*4, xx+dx_per_y*5, xx+dx_per_y*6, xx+dx_per_y*7
        //); 

            
        // ~14 cycles per pixel (14*8 per iteration ~109-112 cycles) w/ fancy lighting and fancy fog
        // ~13 cycles with no lighting and fog
        // ~12 cycles with no lighting and no fog
        
        //profile_block rotate_light_and_blend_per_row;
        //TimeBlock(rotate_light_and_blend_per_row, "rotate light and blend per-pixel");
        for(int oy = min_y; oy < max_y; oy++) { 
            __m256i iyys = _mm256_add_epi32(y_buffer_vec, _mm256_cvtps_epi32(yy_vec)); 
            __m256i ixxs = _mm256_add_epi32(x_buffer_vec, _mm256_cvtps_epi32(xx_vec));
            
            //yy_vec = _mm256_add_ps(yy_vec, dy_per_x_vec);
            //xx_vec = _mm256_add_ps(xx_vec, dx_per_x_vec);
            yy_vec = _mm256_add_ps(yy_vec, dy_per_y_vec);
            xx_vec = _mm256_add_ps(xx_vec, dx_per_y_vec);
            //yy += dy_per_x;
            //xx += dx_per_x;
            __m256i buf_idxs = fb_swizzle_256(ixxs, iyys);
            __m256i world_space_poses = _mm256_i32gather_epi32((u32*)world_pos_buffer, buf_idxs, 4);
            __m256i xs = _mm256_srli_epi32(world_space_poses, 19);
            __m256i ys = _mm256_and_si256(_mm256_srli_epi32(world_space_poses, 8), low_eleven_bits_mask);
            __m256 dxs = _mm256_sub_ps(pos_x_vec, _mm256_cvtepi32_ps(xs));
            __m256 dys = _mm256_sub_ps(pos_y_vec, _mm256_cvtepi32_ps(ys));
            __m256 float_depths = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(dxs, dxs), _mm256_mul_ps(dys, dys)));

            __m256i color_idxs = _mm256_and_si256(world_space_poses, low_seven_bits_mask);
            __m256i undrawn = _mm256_cmpeq_epi32(undrawn_bit_mask_vec, _mm256_and_si256(undrawn_bit_mask_vec, world_space_poses));
            __m256i depths_eq_max = undrawn;
            
            
            //__m256i voxelmap_idxs = get_voxelmap_idx_256(xs, ys);

            // each column is 128 rgbas
            // which is 512 
            //__m256i shifted_voxelmap_idxs = _mm256_slli_epi32(voxelmap_idxs, 6); // for 7 bits of color_idx
            //__m256i color_voxelmap_idxs = _mm256_add_epi32(shifted_voxelmap_idxs, color_idxs);

            //__m256i albedos = _mm256_i32gather_epi32((u32*)columns_colors_data, color_voxelmap_idxs, 4);
            //albedos = _mm256_blendv_epi8(albedos, background_color_vec, undrawn);



            __m256i albedos = _mm256_i32gather_epi32((s32*)albedo_buffer, buf_idxs, 4);

            // aaaaabbbbbgggggrrrrr
            //             rrrrr000
            //             ggggg000
            __m256i albedo_rs = _mm256_and_si256(albedos, low_eight_bits_mask);
            __m256i albedo_gs = _mm256_and_si256(_mm256_srli_epi32(albedos, 8), low_eight_bits_mask);
            __m256i albedo_bs = _mm256_and_si256(_mm256_srli_epi32(albedos, 16), low_eight_bits_mask);

        


            __m256 float_rs = srgb255_to_linear_256(_mm256_cvtepi32_ps(albedo_rs));
            __m256 float_gs  = srgb255_to_linear_256(_mm256_cvtepi32_ps(albedo_gs));
            __m256 float_bs = srgb255_to_linear_256(_mm256_cvtepi32_ps(albedo_bs));

            __m256i int_rs, int_gs, int_bs;

            //__m256i world_zs_ints = _mm256_and_si256(low_ten_bits_mask, world_poses);
            //__m256 world_zs = _mm256_cvtepi32_ps(world_zs_ints);
            //__m256 world_ys = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(world_poses, 10), low_ten_bits_mask));
            //__m256 world_xs = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(world_poses, 20), low_ten_bits_mask));
            //__m256 world_dys = _mm256_min_ps(
            //                        abs_ps(_mm256_sub_ps(world_ys, pos_y_vec)),
            //                        abs_ps(_mm256_sub_ps(world_ys, _mm256_add_ps(pos_y_vec, ten_twenty_four_vec))));
            //__m256 world_dxs = _mm256_min_ps(
            //                        abs_ps(_mm256_sub_ps(world_xs, pos_x_vec)),
            //                        abs_ps(_mm256_sub_ps(world_xs, _mm256_add_ps(pos_x_vec, ten_twenty_four_vec))));
            //__m256 world_dzs = _mm256_sub_ps(world_zs, inverse_height_vec);
            //__m256i depths_eq_max = _mm256_cmpeq_epi32(world_zs_ints, tenth_bit_mask);

            //__m256 float_depths = _mm256_sqrt_ps(
            //                        _mm256_add_ps(_mm256_mul_ps(world_dys, world_dys), 
            //                                         _mm256_mul_ps(world_dxs, world_dxs)));
            
            //float_depths = _mm256_blendv_ps(float_depths, _mm256_set1_ps(max_z), (__m256)depths_eq_max);

            //__m256i depths = _mm256_and_si256(gbuf_entries, depth_mask_vec);
            //__m256i depths_eq_max = _mm256_cmpeq_epi32(depths, _mm256_set1_epi32(65535));
            //__m256 float_depths = _mm256_mul_ps(_mm256_cvtepi32_ps(depths), one_over_fix_scale_vec);
         
            // 32-bit norm_pt1, norm_pt2
            // 32-bit, 27-bit world position (plus top 5-bits of color?)
            // 19-bit color
            //
            
            __m256 encoded_normal_xs = _mm256_i32gather_ps((f32*)norm_buffer, buf_idxs, 8);
            __m256 encoded_normal_ys = _mm256_i32gather_ps((f32*)(norm_buffer+1), buf_idxs, 8);



            __m256 normal_xs = _mm256_add_ps(encoded_normal_xs, encoded_normal_ys);
            __m256 normal_ys = _mm256_sub_ps(encoded_normal_xs, encoded_normal_ys);
            __m256 abs_normal_xs = abs_ps(normal_xs);
            __m256 abs_normal_ys = abs_ps(normal_ys);
            __m256 normal_zs = _mm256_sub_ps(_mm256_sub_ps(two_ps_vec, abs_normal_xs), abs_normal_ys);
            __m256 recip_mag_norm = reciprocal_magnitude_vector_256(normal_xs, normal_ys, normal_zs);
            normal_xs = _mm256_mul_ps(normal_xs, recip_mag_norm);
            normal_ys = _mm256_mul_ps(normal_ys, recip_mag_norm);
            normal_zs = _mm256_mul_ps(normal_zs, recip_mag_norm);

            // y,x,z looks pretty good
            // x,y,z looks vaporwave


            // z,y,x looks ok
            if(view == VIEW_NORMALS) {
                float_rs = (normal_ys + 1.0)/2.0;
                float_gs = (normal_xs + 1.0)/2.0;
                float_bs = (normal_zs + 1.0)/2.0;

            } else if (view == VIEW_DEPTH) {
                // this is now 0.0 to 1.0
                
                __m256 depth_zero_to_one = _mm256_div_ps(float_depths, _mm256_set1_ps(2048));
                __m256 depth_brightness = lerp256(one_ps_vec, depth_zero_to_one, zero_ps_vec);
                depth_brightness = _mm256_blendv_ps(depth_brightness, zero_ps_vec, (__m256)depths_eq_max);
                float_rs = depth_brightness;
                float_gs = depth_brightness;
                float_bs = depth_brightness;
            
            } else if (view == VIEW_STANDARD) {

            
                if(lighting == FANCY_LIGHTING) {
                    

                    //__m256i albedo_ao = _mm256_and_si256(_mm256_srli_epi32(albedos, 24), low_eight_bits_mask);
                    //__m256 ao_float = _mm256_div_ps(_mm256_cvtepi32_ps(albedo_ao), _mm256_set1_ps(255.0));
                    //float_rs = _mm256_mul_ps(ao_float, float_rs);
                    //float_gs = _mm256_mul_ps(ao_float, float_gs);
                    //float_bs = _mm256_mul_ps(ao_float, float_bs);
                } else if (lighting == SIDE_LIGHTING) {
                    __m256 dot_lights = //_mm256_add_ps(
                                            //_mm256_mul_ps(normal_xs, sun_vec_x_vec),
                                            _mm256_add_ps(_mm256_mul_ps(normal_ys, sun_vec_y_vec),
                                                            _mm256_mul_ps(normal_zs, sun_vec_z_vec));//);

                    __m256 color_norm_scales = _mm256_min_ps(one_ps_vec, 
                                                    _mm256_add_ps(_mm256_max_ps(zero_ps_vec, 
                                                                                _mm256_min_ps(dot_lights, _mm256_set1_ps(.25))), 
                                                                  amb_light_factor_vec));
                    
                    __m256 depth_color_factor = _mm256_andnot_ps((__m256)depths_eq_max, color_norm_scales);

                    float_rs = _mm256_mul_ps(depth_color_factor, float_rs); 
                    float_bs = _mm256_mul_ps(depth_color_factor, float_bs); 
                    float_gs = _mm256_mul_ps(depth_color_factor, float_gs); 

                } else {
                    //__m256 depth_amb_factor = _mm256_andnot_ps((__m256)depths_eq_max, amb_light_factor_vec);
                    //float_rs = _mm256_mul_ps(depth_amb_factor, float_rs); 
                    //float_bs = _mm256_mul_ps(depth_amb_factor, float_bs); 
                    //float_gs = _mm256_mul_ps(depth_amb_factor, float_gs); 
                }
                    //float_rs = _mm256_blendv_ps(float_rs, fog_rs, (__m256)depths_eq_max);
                    //float_gs = _mm256_blendv_ps(float_gs, fog_gs, (__m256)depths_eq_max);
                    //float_bs = _mm256_blendv_ps(float_bs, fog_bs, (__m256)depths_eq_max);
                    //float_rs = _mm256_mul_ps(depth_color_factor, float_rs); 
                    //float_bs = _mm256_mul_ps(depth_color_factor, float_bs); 
                    //float_gs = _mm256_mul_ps(depth_color_factor, float_gs); 

                if(fogmode) {
                    //f32 z_to_1 = (avg_dist*one_over_max_z)*(avg_dist*one_over_max_z);
                    __m256 z_to_ones = _mm256_min_ps(one_ps_vec, _mm256_mul_ps(float_depths, one_over_max_z_vec));
                    z_to_ones = _mm256_mul_ps(z_to_ones, z_to_ones);
                    z_to_ones = _mm256_mul_ps(z_to_ones, z_to_ones);
                    __m256 fog_factors = lerp256(zero_ps_vec, z_to_ones, one_ps_vec);
                    __m256 one_minus_fog_factors = _mm256_sub_ps(one_ps_vec, fog_factors);
                    //f32 fog_factor = lerp(0, z_to_1, 1);
                    //one_minus_fog = 1-fog_factor;
                    __m256 mult_fog_rs = _mm256_mul_ps(fog_factors, fog_rs);
                    __m256 mult_fog_gs = _mm256_mul_ps(fog_factors, fog_gs);
                    __m256 mult_fog_bs = _mm256_mul_ps(fog_factors, fog_bs);
                    __m256 mult_rs = _mm256_mul_ps(one_minus_fog_factors, float_rs);
                    __m256 mult_gs = _mm256_mul_ps(one_minus_fog_factors, float_gs);
                    __m256 mult_bs = _mm256_mul_ps(one_minus_fog_factors, float_bs);
                    float_rs = _mm256_add_ps(mult_fog_rs, mult_rs);
                    float_gs = _mm256_add_ps(mult_fog_gs, mult_gs);
                    float_bs = _mm256_add_ps(mult_fog_bs, mult_bs);
                }
                float_rs = _mm256_blendv_ps(float_rs, fog_rs, (__m256)depths_eq_max);
                float_gs = _mm256_blendv_ps(float_gs, fog_gs, (__m256)depths_eq_max);
                float_bs = _mm256_blendv_ps(float_bs, fog_bs, (__m256)depths_eq_max);
            }


            int_rs = linear_to_srgb_255_256(float_rs); // _mm256_cvtps_epi32(float_rs);
            int_gs = linear_to_srgb_255_256(float_gs); //_mm256_cvtps_epi32(float_gs);
            int_bs = linear_to_srgb_255_256(float_bs); //_mm256_cvtps_epi32(float_bs);


            __m256i rgbs = _mm256_or_si256(
                _mm256_or_si256(alpha_vec, _mm256_slli_epi32(int_bs, 16)),
                _mm256_or_si256(_mm256_slli_epi32(int_gs, 8), int_rs)
            );
            if(double_pixels == 2) {
                __m256i out_reg1_selectors = _mm256_setr_epi32(0,0,0,0,1,1,1,1);
                __m256i out_reg2_selectors = _mm256_setr_epi32(2,2,2,2,3,3,3,3);
                __m256i out_reg3_selectors = _mm256_setr_epi32(4,4,4,4,5,5,5,5);
                __m256i out_reg4_selectors = _mm256_setr_epi32(6,6,6,6,7,7,7,7);
                u32* pix_ptr = &pixels[(oy*4)*OUTPUT_WIDTH+(base_ox*4)+0];
                __m256i out_reg1 = _mm256_permutevar8x32_epi32(rgbs, out_reg1_selectors);
                __m256i out_reg2 = _mm256_permutevar8x32_epi32(rgbs, out_reg2_selectors);
                __m256i out_reg3 = _mm256_permutevar8x32_epi32(rgbs, out_reg3_selectors);
                __m256i out_reg4 = _mm256_permutevar8x32_epi32(rgbs, out_reg4_selectors);
                for(int i = 0; i < 4; i++) {
                    _mm256_store_si256((__m256i*)(pix_ptr+(OUTPUT_WIDTH*i)), out_reg1);
                    _mm256_store_si256((__m256i*)(pix_ptr+(OUTPUT_WIDTH*i)+8), out_reg2);
                    _mm256_store_si256((__m256i*)(pix_ptr+(OUTPUT_WIDTH*i)+16), out_reg3);
                    _mm256_store_si256((__m256i*)(pix_ptr+(OUTPUT_WIDTH*i)+24), out_reg4);
                }
               

            } else if (double_pixels == 1) { 
                __m256i out_reg1_selectors = _mm256_setr_epi32(0,0,1,1,2,2,3,3);
                __m256i out_reg2_selectors = _mm256_setr_epi32(4,4,5,5,6,6,7,7);
                u32* pix_ptr = &pixels[(oy*2)*OUTPUT_WIDTH+(base_ox*2)];
                __m256i out_reg1 = _mm256_permutevar8x32_epi32(rgbs, out_reg1_selectors);
                __m256i out_reg2 = _mm256_permutevar8x32_epi32(rgbs, out_reg2_selectors);
                _mm256_store_si256((__m256i*)pix_ptr, out_reg1);
                _mm256_store_si256((__m256i*)(pix_ptr+8), out_reg2);
                _mm256_store_si256((__m256i*)(pix_ptr+OUTPUT_WIDTH), out_reg1);
                _mm256_store_si256((__m256i*)(pix_ptr+OUTPUT_WIDTH+8), out_reg2);
            } else {
                //pixels[oy*render_width+(ox)] = pix;  
                u32* pix_ptr = &pixels[oy*render_width+base_ox];   
                //u32* pix_ptr = &pixels[oy*render_width];   
                _mm256_store_si256((__m256i*)pix_ptr, rgbs);
            }
            //pix_ptr += 8;
            //next_row_pix_ptr += 8;
        }
        //EndCountedTimeBlock(rotate_light_and_blend_per_row, (max_y-min_x)*8);
    #else
        //scalar version 
        for(int ox = 0; ox < render_width; ox++) {


            s32 iyy = yy+y_buffer;
            s32 ixx = xx+x_buffer;
            yy += dy_per_x;
            xx += dx_per_x;
            u32 fb_idx = fb_swizzle(ixx, iyy);
            g_buf_entry entry = g_buffer[fb_idx];

            //f32 z = (1/z_buffer[fb_idx])*scale_height;   //1/10 240
            //u8 ch = lerp(0, z/max_z, 255);
            //u32 pix = ((0xFF << 24) | (ch << 16) | (ch << 8) | ch);
            //u32 pix = inter_buffer[fb_idx];
            u16 short_pix = entry.albedo;
            u8 r = (short_pix&0b11111)<<3;
            u8 g = ((short_pix>>5)&0b11111)<<3;
            u8 b = ((short_pix>>10)&0b11111)<<3;
            u32 pix = (0xFF << 24) | (b << 16) | (g << 8) | r;

            if(double_pixels) {
                int scale_size = (1 << double_pixels); //== 1 ? 2 : 4);
                for(int iy = 0; iy < scale_size; iy++) {
                    for(int ix = 0; ix < scale_size; ix++) {
                        pixels[((oy*scale_size)+iy)*OUTPUT_WIDTH+(ox*scale_size)+ix] = pix;
                    }
                }
            } else {                
                pixels[oy*render_width+(ox)] = pix;       
            }
        }
    #endif
        
        col_x += dx_per_x*8;
        col_y += dy_per_x*8;
        //row_x += dx_per_y;
        //row_y += dy_per_y;
    }

    EndTimeBlock(rotate_light_and_blend_block);

}

typedef enum {
    X_SIDE = 0,
    Y_SIDE = 1,
} side;

void raycast_scalar(s32 min_x, s32 min_y, s32 max_x, s32 max_y) {
  
    
    f32 fog_r = (background_color&0xFF);
    f32 fog_g = ((background_color>>8)&0xFF);
    f32 fog_b = ((background_color>>16)&0xFF);

 

    float2 encoded_top_face_norm = encode_norm(0, -1, 0);
    float2 encoded_bot_face_norm = encode_norm(0, 1, 0);
    float2 encoded_x_side_norm = encode_norm( (dir_x > 0 ? 1 : -1), 0, 0);
    float2 encoded_y_side_norm = encode_norm(0, 0, (dir_y > 0 ? 1 : -1));
    

    f32 wrap_pos_x = pos_x;

    f32 wrap_pos_y = pos_y;
    
    f32 sun_vec_x = 0;
    f32 sun_vec_y = sinf(sun_ang);
    f32 sun_vec_z = cosf(sun_ang);
    f32 one_over_max_z = 1/max_z;
    u32 mip_0_steps = 0;
    u32 mip_1_steps = 0;
    f32 pitch = pitch_ang_to_pitch(pitch_ang);
    u8 prev_side, next_side;

    profile_block raycast_scalar_block;
    TimeBlock(raycast_scalar_block, "raycast scalar");
    for(int x = min_x; x < max_x; x++) {

        int next_drawable_min_y = min_y;
        int prev_drawn_max_y = max_y+1;//render_size;

        f32 camera_space_x = ((2 * x) / ((f32)render_size)) - 1; //x-coordinate in camera space


        f32 ray_dir_x = dir_x + (plane_x * camera_space_x);
        f32 ray_dir_y = dir_y + (plane_y * camera_space_x);
        //which box of the map we're in
        s32 map_x = ((s32)wrap_pos_x);
        s32 map_y = ((s32)wrap_pos_y);


        //length of ray from one x or y-side to next x or y-side
        f32 delta_dist_x = (ray_dir_x == 0) ? 1e30 : fabs(1 / ray_dir_x);
        f32 delta_dist_y = (ray_dir_y == 0) ? 1e30 : fabs(1 / ray_dir_y);

        f32 wrap_x_minus_map_x = (wrap_pos_x - map_x);
        f32 map_x_plus_one_minus_wrap_x = (map_x + (1.0 - wrap_pos_x));
        f32 wrap_y_minus_map_y = (wrap_pos_y - map_y);
        f32 map_y_plus_one_minus_wrap_y = (map_y + (1.0 - wrap_pos_y));


        //what direction to step in x or y-direction (either +1 or -1)
        int step_x = (ray_dir_x < 0) ? -1 : 1;
        int step_y = (ray_dir_y < 0) ? -1 : 1;
        f32 perp_wall_dist = 0;
        f32 next_perp_wall_dist = 0;

        //length of ray from current position to next x or y-side
        f32 side_dist_x = (ray_dir_x < 0 ? wrap_x_minus_map_x : map_x_plus_one_minus_wrap_x) * delta_dist_x;
        f32 side_dist_y = (ray_dir_y < 0 ? wrap_y_minus_map_y : map_y_plus_one_minus_wrap_y) * delta_dist_y;


        //_mm_setr_ps(side_dist_x, side_dist_y, side_dist_z);




        while(perp_wall_dist == 0) {
            int map_x_dx = (side_dist_x < side_dist_y) ? step_x : 0;
            int map_y_dy = (side_dist_x < side_dist_y) ? 0 : step_y;

            perp_wall_dist = next_perp_wall_dist;
            next_perp_wall_dist = (side_dist_x < side_dist_y) ? side_dist_x : side_dist_y;


            f32 side_dist_dx = (side_dist_x < side_dist_y) ? delta_dist_x : 0;
            f32 side_dist_dy = (side_dist_x < side_dist_y) ? 0 : delta_dist_y;
            prev_side = next_side;
            next_side = (side_dist_x < side_dist_y) ? X_SIDE : Y_SIDE;

            side_dist_x += side_dist_dx;
            side_dist_y += side_dist_dy;

            map_x += map_x_dx;
            map_y += map_y_dy;
        }

        f32 invz = scale_height / perp_wall_dist;
        f32 next_invz = scale_height / next_perp_wall_dist;

        profile_block z_step_block; int z_steps = 0;

        s32 max_x_steps = (ray_dir_x > 0 ? (1023-(s32)pos_x) : (((s32)pos_x))) + 1;
        s32 max_y_steps = (ray_dir_y > 0 ? (1023-(s32)pos_y) : (((s32)pos_y))) + 1;


        while(perp_wall_dist <= max_z && prev_drawn_max_y > next_drawable_min_y && max_x_steps > 0 && max_y_steps > 0) {
            //s32 texel_per_y = (s32)((f32)perp_wall_dist / scale_height);
            //if(log2_fast(texel_per_y) > 0) {
            //    mip_1_steps++;
            //} else {
            //    mip_0_steps++;
            //}

            f32 avg_dist = (perp_wall_dist+next_perp_wall_dist)/2;
            
            u16 near_side_dist_fix = min(65535, (int)(perp_wall_dist *32));
            u16 avg_wall_dist_fix = min(65535, (int)(avg_dist*32)); //11.5 fixed point

            // 8 bit z
            // 11 bit x
            // 11 bit y

            u32 prepared_combined_world_map_pos = ((map_x)<<19)|((map_y)<<8); 
            //u32 prepared_combined_world_map_pos = ((int)(avg_dist*32)<<16);

            //u16 near_side_dist_fix = min(65535, (int)((1.0/perp_wall_dist)*32768.0)); // *32
            //u16 avg_wall_dist_fix = min(65535, (int)((1.0/avg_dist)*32768.0)); // *32 //11.5 fixed point
            
            
            int map_x_dx = (side_dist_x < side_dist_y) ? step_x : 0;
            int map_y_dy = (side_dist_x < side_dist_y) ? 0 : step_y;

            perp_wall_dist = next_perp_wall_dist;
            next_perp_wall_dist = (side_dist_x < side_dist_y) ? side_dist_x : side_dist_y;


            f32 side_dist_dx = (side_dist_x < side_dist_y) ? delta_dist_x : 0;
            f32 side_dist_dy = (side_dist_x < side_dist_y) ? 0 : delta_dist_y;
            prev_side = next_side;
            u32 d_x_steps = (side_dist_x < side_dist_y) ? 1 : 0;
            u32 d_y_steps = (side_dist_x < side_dist_y) ? 0 : 1;
            max_x_steps -= d_x_steps;
            max_y_steps -= d_y_steps;

            next_side = (side_dist_x < side_dist_y) ? X_SIDE : Y_SIDE;
            u32 map_idx = get_swizzled_map_idx(map_x, map_y);
            u32 voxelmap_idx = get_voxelmap_idx(map_x, map_y);

            side_dist_x += side_dist_dx;
            side_dist_y += side_dist_dy;

            s32 next_map_x = map_x + map_x_dx;
            s32 next_map_y = map_y + map_y_dy;

            next_invz = scale_height / next_perp_wall_dist;

            if(map_x < 0 || map_x > 1023 || map_y < 0 || map_y > 1023) {
                goto next_z_step;
            }

            
            //f32 map_norm_pt1 = normal_pt1_data[map_idx];
            //f32 map_norm_pt2 = normal_pt2_data[map_idx];

            //f32 normal_pt1 = (lighting == SIDE_LIGHTING ? ((prev_side == X_SIDE) ? encoded_x_side_norm.x : encoded_y_side_norm.x) : map_norm_pt1);
            //f32 normal_pt2 = (lighting == SIDE_LIGHTING ? ((prev_side == X_SIDE) ? encoded_x_side_norm.y : encoded_y_side_norm.y) : map_norm_pt2);
            //f32 top_face_norm_pt1 = (lighting == SIDE_LIGHTING ? encoded_top_face_norm.x : map_norm_pt1);
            //f32 top_face_norm_pt2 = (lighting == SIDE_LIGHTING ? encoded_top_face_norm.y : map_norm_pt2);
            //f32 bot_face_norm_pt1 = (lighting == SIDE_LIGHTING ? encoded_bot_face_norm.x : map_norm_pt1);
            //f32 bot_face_norm_pt2 = (lighting == SIDE_LIGHTING ? encoded_bot_face_norm.y : map_norm_pt2);

            f32 normal_pt1 = ((prev_side == X_SIDE) ? encoded_x_side_norm.x : encoded_y_side_norm.x);
            f32 normal_pt2 = ((prev_side == X_SIDE) ? encoded_x_side_norm.y : encoded_y_side_norm.y);
            f32 top_face_norm_pt1 = encoded_top_face_norm.x;
            f32 top_face_norm_pt2 = encoded_top_face_norm.y;
            f32 bot_face_norm_pt1 = encoded_bot_face_norm.x;
            f32 bot_face_norm_pt2 = encoded_bot_face_norm.y;
            //f32 normal_pt1 = (SIDE_LIGHTING == SIDE_LIGHTING ? ((prev_side == X_SIDE) ? encoded_x_side_norm.x : encoded_y_side_norm.x) : map_norm_pt1);
            //f32 normal_pt2 = (SIDE_LIGHTING == SIDE_LIGHTING ? ((prev_side == X_SIDE) ? encoded_x_side_norm.y : encoded_y_side_norm.y) : map_norm_pt2);
            //f32 top_face_norm_pt1 = (SIDE_LIGHTING == SIDE_LIGHTING ? encoded_top_face_norm.x : map_norm_pt1);
            //f32 top_face_norm_pt2 = (SIDE_LIGHTING == SIDE_LIGHTING ? encoded_top_face_norm.y : map_norm_pt2);
            //f32 bot_face_norm_pt1 = (SIDE_LIGHTING == SIDE_LIGHTING ? encoded_bot_face_norm.x : map_norm_pt1);
            //f32 bot_face_norm_pt2 = (SIDE_LIGHTING == SIDE_LIGHTING ? encoded_bot_face_norm.y : map_norm_pt2);



            f32 mult_fog_r, mult_fog_g, mult_fog_b;
            f32 one_minus_fog;

        #if 0
            // heightmap renderer
            u32 depth = depthmap_u32s[idx];
            u32 abgr = cmap[idx];
            u8 r  = ((abgr&0xFF) * one_minus_fog) + mult_fog_r;
            u8 g = (((abgr>>8)&0xFF) * one_minus_fog) + mult_fog_g;
            u8 b = (((abgr>>16)&0xFF) * one_minus_fog) + mult_fog_b;
            r = r*one_minus_fog + mult_fog_r;
            g = g*one_minus_fog + mult_fog_g;
            b = b*one_minus_fog + mult_fog_b;
            abgr = (0xFF<<24)|(b<<16)|(g<<8)|r;
            f32 relative_height = height-depth;

            
            //f32 dist = (relative_top < 0 ? prev_perp_wall_dist : perp_wall_dist);
            //f32 invz = scale_height / dist;
            f32 invz = (relative_height < 0 ? near_invz : far_invz);

            f32 float_projected_height = relative_height*invz;


            s32 int_projected_height = floor(float_projected_height);
            s32 heightonscreen = int_projected_height + pitch;

            heightonscreen = min(prev_drawn_max_y, max(min_y, heightonscreen));
            if(heightonscreen < prev_drawn_max_y) { 
                u32 idx = fb_swizzle(x,heightonscreen); 
                for(int y = heightonscreen; y < prev_drawn_max_y; y++) {
                    inter_buffer[idx] = abgr;
                    idx += 8;
                }
                prev_drawn_max_y = heightonscreen;
            }
        #else

            column_header* header = &columns_header_data[voxelmap_idx];


            // check the top of this column against the bottom of the frustum skip drawing it

            f32 relative_top_of_col =  height - (255 - header->max_y); //[voxelmap_idx];
            f32 col_top_invz = (relative_top_of_col < 0 ? invz : next_invz);
            f32 float_top_of_col_projected_height = relative_top_of_col*col_top_invz;
            s32 int_top_of_col_projected_height = floor(float_top_of_col_projected_height) + pitch;

            span* span_info = columns_runs_data[voxelmap_idx].runs_info;
            
            if(int_top_of_col_projected_height >= prev_drawn_max_y) {
                goto next_z_step;
            }
            u8 num_runs = header->num_runs;

            f32 relative_bot_of_col = height - (255 - span_info[num_runs-1].bottom_voxels_end);
            f32 col_bot_invz = (relative_bot_of_col < 0 ? next_invz : invz);
            f32 float_bot_of_col_projected_height = relative_bot_of_col * col_bot_invz;
            s32 int_bot_of_col_projected_height = floor(float_bot_of_col_projected_height) + pitch;
            if(int_bot_of_col_projected_height <= next_drawable_min_y) {
                goto next_z_step;
            }



// draw top or bottom face

u32 mix_colors(u32 old_color, u32 new_color) {
    //return new_color;
    
    u32 old_col_without_alpha = (old_color & (~((u32)0b11<<24)));
    u32 new_col_without_alpha = (new_color & (~((u32)0b11<<24)));
    u8 both_transparent = ((((old_color>>24)&0b11) != 3) && ((new_color>>24)&0b11) !=3);
    

    u8 new_r = (new_color>>0)&0xFF;
    u8 new_g = (new_color>>8)&0xFF;
    u8 new_b = (new_color>>16)&0xFF;
    u8 new_color_coverage = (new_color>>24)&0b11;

    u8 old_r = (old_color>>0)&0xFF;
    u8 old_g = (old_color>>8)&0xFF;
    u8 old_b = (old_color>>16)&0xFF;
    u8 old_color_coverage = (old_color>>24)&0b11;


    u8 mixed_color_coverage = min((new_color_coverage + old_color_coverage), 0b11);
    u8 one_minus_old_color_coverage = (0b11-old_color_coverage); // 0 to 3?
    f32 one_minus_old_color_coverage_f = (one_minus_old_color_coverage)/3.0;
#if 0

    __m128 new_vec_ps = _mm_set_ps(0xFF, new_b, new_g, new_r);
    __m128 scaled_new_vec_ps = _mm_mul_ps(new_vec_ps, _mm_set1_ps(one_minus_old_color_coverage/3.0));
    __m64 new_vec_ps_int = _mm_cvtps_pi8(scaled_new_vec_ps);
    __m128i new_col_vec = _mm_set1_epi32(new_vec_ps_int[0]);
    __m128i old_col_vec = _mm_set1_epi32(old_color);

    __m128i summed = _mm_adds_epu8(new_col_vec, old_col_vec);

    u32 abgr = _mm_extract_epi32(summed, 0) & (0x00FFFFFF);

    return (both_transparent && (old_col_without_alpha == new_col_without_alpha)) ? old_color : (mixed_color_coverage<<24)|abgr;
    //return (mixed_color_coverage<<24)|abgr;
#else
    u8 r = old_r + (one_minus_old_color_coverage_f*new_r);
    u8 g = old_g + (one_minus_old_color_coverage_f*new_g);
    u8 b = old_b + (one_minus_old_color_coverage_f*new_b);
    

    return (both_transparent && (old_col_without_alpha == new_col_without_alpha)) ? old_color : ((u32)((mixed_color_coverage<<24)|(b<<16)|(g<<8)|(r)));
    //return ((u32)((mixed_color_coverage<<24)|(b<<16)|(g<<8)|(r)));
#endif
}

#define DRAW_CHUNK_FACE(top, bot, face_norm_1, face_norm_2, voxel_color_idx) {              \
    u32 fb_idx = fb_swizzle(x,top);                                                         \
    u32 color = top_of_col_color_ptr[voxel_color_idx];                                                                 \
    for(int y = top; y < bot; y++) {                                                        \
        u8 occlusion_bit = get_occlusion_bit(x, y);                                         \
        u32 combined_world_pos = prepared_combined_world_map_pos|(voxel_color_idx);         \
        f32 old_norm_pt1 = norm_buffer[fb_idx*2];                                           \
        f32 old_norm_pt2 = norm_buffer[fb_idx*2+1];                                         \
        u32 old_color = albedo_buffer[fb_idx];                                              \
        u32 mixed_color = mix_colors(old_color, color);                                               \
        f32 new_norm_pt1 = occlusion_bit ? old_norm_pt1 : (face_norm_1);                    \
        f32 new_norm_pt2 = occlusion_bit ? old_norm_pt2 : (face_norm_2);                    \
        u32 old_world_pos = world_pos_buffer[fb_idx];                                       \
        u32 new_world_pos = occlusion_bit ? old_world_pos : combined_world_pos;             \
        u32 new_color = (occlusion_bit ? old_color : mixed_color);                                \
        u8 new_occlusion_bit = (occlusion_bit ? 1 : (((new_color>>24)&0b11) == 0b11) ? 1 : 0); \
        min_coverage = min(new_occlusion_bit, min_coverage);                                \
        set_occlusion_bit(x, y, new_occlusion_bit);                                                            \
        norm_buffer[fb_idx*2] = new_norm_pt1;                                               \
        norm_buffer[fb_idx*2+1] = new_norm_pt2;                                             \
        world_pos_buffer[fb_idx] = new_world_pos;                                           \
        albedo_buffer[fb_idx] = new_color;                                                  \
        fb_idx += 8;                                                                        \
    }                                                                                       \
}

// draw side of chunk
// needs voxel color interpolation
#define DRAW_CHUNK_SIDE(top, bot, norm_pt1, norm_pt2) {                                     \
    s32 clipped_top_y = clipped_top_side_height - int_top_side_projected_height;            \
    f32 texel_per_y = ((f32)num_voxels) / unclipped_screen_dy;                              \
    f32 cur_voxel_color_idx = (f32)clipped_top_y * texel_per_y;                             \
    u32 fb_idx = fb_swizzle(x,top);                                                         \
    for(int y = top; y < bot; y++) {                                                        \
        u8 occlusion_bit = get_occlusion_bit(x, y);                                         \
        u16 voxel_color_idx = cur_voxel_color_idx;                                          \
        u32 color = color_ptr[voxel_color_idx];                                             \
        u32 combined_world_pos = prepared_combined_world_map_pos|((voxel_color_idx+color_ptr)-top_of_col_color_ptr);         \
        u32 old_color = albedo_buffer[fb_idx];                                              \
        u32 mixed_color = mix_colors(old_color, color);                                               \
        cur_voxel_color_idx += texel_per_y;                                                 \
        f32 old_norm_pt1 = norm_buffer[fb_idx*2];                                           \
        f32 old_norm_pt2 = norm_buffer[fb_idx*2+1];                                         \
        f32 new_norm_pt1 = occlusion_bit ? old_norm_pt1 : (norm_pt1);                       \
        f32 new_norm_pt2 = occlusion_bit ? old_norm_pt2 : (norm_pt2);                       \
        u32 old_world_pos = world_pos_buffer[fb_idx];                                       \
        u32 new_world_pos = occlusion_bit ? old_world_pos : combined_world_pos;             \
        u32 new_color = (occlusion_bit ? old_color : mixed_color);                                \
        u8 new_occlusion_bit = (occlusion_bit ? 1 : (((new_color>>24)&0b11) == 0b11) ? 1 : 0); \
        min_coverage = min(new_occlusion_bit, min_coverage);                                \
        set_occlusion_bit(x, y, new_occlusion_bit);                                                            \
        norm_buffer[fb_idx*2] = new_norm_pt1;                                               \
        norm_buffer[fb_idx*2+1] = new_norm_pt2;                                             \
        world_pos_buffer[fb_idx] = new_world_pos;                                           \
        albedo_buffer[fb_idx] = new_color;                                                  \
        fb_idx += 8;                                                                        \
    }                                                                                       \
}

#define CLAMP(a, mi, ma) max(mi, min(a, ma))

#define DRAW_CHUNK(chunk_top, chunk_bot, break_of_top_below_screen, break_if_bot_above_screen, side_norm_pt1, side_norm_pt2, face_norm_pt1, face_norm_pt2) {         \
    f32 relative_bot = height-chunk_bot;                                                 \
    f32 relative_top = height-chunk_top;                                                 \
    s32 int_top_face_projected_height = floor(relative_top*next_invz) + pitch;           \
    s32 int_top_side_projected_height = floor(relative_top*invz) + pitch;                \
    s32 int_bot_face_projected_height = floor(relative_bot*next_invz) + pitch;           \
    s32 int_bot_side_projected_height = floor(relative_bot*invz) + pitch;                \
    s32 top_projected_heightonscreen = min(int_top_side_projected_height, int_top_face_projected_height);\
    s32 bot_projected_heightonscreen = max(int_bot_side_projected_height,int_bot_face_projected_height);  \
    if(break_if_bot_above_screen && bot_projected_heightonscreen < next_drawable_min_y) { \
        break;\
    }\
    if(break_of_top_below_screen && top_projected_heightonscreen >= prev_drawn_max_y) {    \
        break;\
    }\
    s32 next_prev_drawn_max_y = (bot_projected_heightonscreen >= prev_drawn_max_y && top_projected_heightonscreen < prev_drawn_max_y) ? top_projected_heightonscreen : prev_drawn_max_y;             \
    s32 next_next_drawable_min_y = (top_projected_heightonscreen <= next_drawable_min_y && bot_projected_heightonscreen > next_drawable_min_y) ? bot_projected_heightonscreen : next_drawable_min_y; \
    s32 clipped_top_heightonscreen = max(next_drawable_min_y, top_projected_heightonscreen);    \
    s32 clipped_bot_heightonscreen = min(prev_drawn_max_y, bot_projected_heightonscreen);       \
    s32 unclipped_screen_dy = int_bot_side_projected_height - int_top_side_projected_height;    \
    s32 num_voxels = (chunk_top - chunk_bot);                                      \
    u8 min_coverage = 1;                                                                \
    if(clipped_top_heightonscreen < clipped_bot_heightonscreen) {                        \
        s32 clipped_top_face_height = CLAMP(int_top_face_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_top_side_height = CLAMP(int_top_side_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_bot_face_height = CLAMP(int_bot_face_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_bot_side_height = CLAMP(int_bot_side_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        if(clipped_top_face_height < clipped_top_side_height) {                          \
            u16 color_offset = (color_ptr - top_of_col_color_ptr);                       \
            DRAW_CHUNK_FACE(clipped_top_face_height, clipped_top_side_height, top_face_norm_pt1, top_face_norm_pt2, color_offset); \
        }                                                                                \
        if(clipped_top_side_height < clipped_bot_side_height) {                          \
            DRAW_CHUNK_SIDE(clipped_top_side_height, clipped_bot_side_height,  side_norm_pt1, side_norm_pt2);           \
        }                                                                                \
        if(clipped_bot_side_height < clipped_bot_face_height) {                          \
            u16 color_offset = ((color_ptr - top_of_col_color_ptr)+num_voxels)-1;                       \
            DRAW_CHUNK_FACE(clipped_bot_side_height, clipped_bot_face_height, bot_face_norm_pt1, bot_face_norm_pt2, color_offset); \
        }                                                                                \
        prev_drawn_max_y = (min_coverage != 0b1 ? prev_drawn_max_y : next_prev_drawn_max_y);             \
        next_drawable_min_y = (min_coverage != 0b1 ? next_drawable_min_y : next_next_drawable_min_y);       \
    }                                                                                    \
}  
                                                                      

            //int chunk_top = -1;
            
            // we have to search to find the middle chunk.
            // then draw up to the minimum chunk
            // and draw down to the maximum chunk
            // but i am going to ignore that for now

            // find mid point
            // draw up from there
            // draw down from there

            int first_downward_run = -1;
            u32* top_of_col_color_ptr = columns_colors_data[voxelmap_idx].colors;
            u32* color_ptr = top_of_col_color_ptr;
            for(int run = 0; run < num_runs; run++) {
                //f32 relative_top_of_run = height - (255 - span_info[run].bottom_voxels_end);
                f32 relative_bot_of_run = height - (255 - span_info[run].bottom_voxels_end);
                //f32 run_bot_invz = (relative_bot_of_run < 0 ? next_invz : invz);
                //f32 float_projected_bot_of_run = relative_bot_of_run * run_bot_invz;
                //s32 projected_bot_of_run = floor(float_projected_bot_of_run) + pitch;
                // if it's above, we've found it :)
                //u16 middle_y = min_y + ((max_y-min_y)/2);
                //if(projected_bot_of_run >= middle_y) {
                //    first_downward_run = run;
                //    break;
                ///}
                if(relative_bot_of_run >= 0) {
                    first_downward_run = run;
                    break;
                }
                color_ptr += (span_info[run].top_voxels_end-span_info[run].top_voxels_start);
                color_ptr += (span_info[run].bottom_voxels_end-span_info[run].bottom_voxels_start);
            }


            if(first_downward_run == -1) {
                // they're all above us
                // color pointer is below everything
                
                for(int run = num_runs-1; run >= 0; run--) {                    
                    int top_surface_top = 255 - span_info[run].top_voxels_start;
                    int top_surface_bot = 255 - (span_info[run].top_voxels_end);
                    int bot_surface_top = 255 - span_info[run].bottom_voxels_start;
                    int bot_surface_bot = 255 - span_info[run].bottom_voxels_end;
                    color_ptr -= (span_info[run].bottom_voxels_end-span_info[run].bottom_voxels_start);
                    if(bot_surface_bot < bot_surface_top) {
                        DRAW_CHUNK(bot_surface_top, bot_surface_bot, 0, 1, normal_pt1, normal_pt2, bot_face_norm_pt1, bot_face_norm_pt2);
                    }
                    color_ptr -= ((span_info[run].top_voxels_end-span_info[run].top_voxels_start));
                    DRAW_CHUNK(top_surface_top, top_surface_bot, 0, 1, normal_pt1, normal_pt2, top_face_norm_pt1, top_face_norm_pt2);\
                }
                

                
            } else {
                u32* middle_col_ptr = color_ptr;
                // upward runs are first_downward_run-1 and to 0
                // downward runs are first_downward_run and to num_runs-1
                for(int run = first_downward_run-1; run >= 0; run--) {                    
                    int top_surface_top = 255 - span_info[run].top_voxels_start;
                    int top_surface_bot = 255 - (span_info[run].top_voxels_end);
                    int bot_surface_top = 255 - span_info[run].bottom_voxels_start;
                    int bot_surface_bot = 255 - span_info[run].bottom_voxels_end;
                    color_ptr -= (span_info[run].bottom_voxels_end-span_info[run].bottom_voxels_start);
                    if(bot_surface_bot < bot_surface_top) {
                        DRAW_CHUNK(bot_surface_top, bot_surface_bot, 0, 0, normal_pt1, normal_pt2, bot_face_norm_pt1, bot_face_norm_pt2);
                    }
                    color_ptr -= ((span_info[run].top_voxels_end-span_info[run].top_voxels_start));
                    DRAW_CHUNK(top_surface_top, top_surface_bot, 0, 0, normal_pt1, normal_pt2, top_face_norm_pt1, top_face_norm_pt2);

                }
                assert(color_ptr == top_of_col_color_ptr);
                color_ptr = middle_col_ptr;
                // color pointer is currently at the first downward run, so let's do those
                for(int run = first_downward_run; run < num_runs; run++) {
                    int top_surface_top = 255 - span_info[run].top_voxels_start;
                    int top_surface_bot = 255 - span_info[run].top_voxels_end;
                    int bot_surface_top = 255 - span_info[run].bottom_voxels_start;
                    int bot_surface_bot = 255 - span_info[run].bottom_voxels_end;
                    DRAW_CHUNK(top_surface_top, top_surface_bot, 1, 0, normal_pt1, normal_pt2, top_face_norm_pt1, top_face_norm_pt2);
                    color_ptr += (span_info[run].top_voxels_end-span_info[run].top_voxels_start);

                    if(bot_surface_bot < bot_surface_top) {
                        DRAW_CHUNK(bot_surface_top, bot_surface_bot, 1, 0, normal_pt1, normal_pt2, bot_face_norm_pt1, bot_face_norm_pt2);
                    }
                    color_ptr += (span_info[run].bottom_voxels_end-span_info[run].bottom_voxels_start);
                }
            }


        next_z_step:;
        map_x = next_map_x;
        map_y = next_map_y;
        invz = next_invz;
    #endif
        }
    }
    EndTimeBlock(raycast_scalar_block);
}

void fill_empty_entries(s32 min_x, s32 min_y, s32 max_x, s32 max_y) {
    //u32 undrawn_albedo = background_color;
    u32 undrawn_world_pos = 0b10000000;
    //f32 norm_pt1_undrawn = 0;
    //f32 norm_pt2_undrawn = 0;
    int min_x_aligned = min_x & ~0b11111;
    __m256i undrawn_vec = _mm256_set1_epi32(undrawn_world_pos);
    __m256 undrawn_norm_pt1_vec = _mm256_set1_ps(0);
    __m256 undrawn_norm_pt2_vec = _mm256_set1_ps(0);

    __m256i selectors = _mm256_setr_epi32(
        1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7
    );

    __m256i norm_pt1_selectors = _mm256_setr_epi32(
        1<<0, 1<<0, 1<<1, 1<<1, 1<<2, 1<<2, 1<<3, 1<<3
    );
    __m256i norm_pt2_selectors = _mm256_setr_epi32(
        1<<4, 1<<4, 1<<5, 1<<5, 1<<6, 1<<6, 1<<7, 1<<7
    );
    __m256i shifters = _mm256_setr_epi32(
        31, 30, 29, 28, 27, 26, 25, 24 
    );
    __m256i norm_pt1_shifters = _mm256_setr_epi32(
        31, 31, 30, 30, 29, 29, 28, 28
    );
    __m256i norm_pt2_shifters = _mm256_setr_epi32(
        27, 27, 26, 26, 25, 25, 24, 24
    );

    // we need two normal vectors that look like this
    // bit 7 needs to be in
    // a b a b a b a b
    // a b a b a b a b 


    profile_block coverage_fill_empty_entries;
    TimeBlock(coverage_fill_empty_entries, "fill empty entries");
    for(int x = min_x_aligned; x < max_x; x += 8) {
        u32 base_fb_idx = fb_swizzle(x, min_y);
        u32* world_pos_ptr = &world_pos_buffer[base_fb_idx];
        f32* norm_ptr = &norm_buffer[base_fb_idx*2];

        //profile_block coverage_fill_empty_entries_per_row;
        //TimeBlock(coverage_fill_empty_entries_per_row, "fill empty entries per row");
        for(int y = min_y; y < max_y; y++) {
            u8 drawn_mask = get_occlusion_byte(x, y);
            
            // fill undrawn pixels with background color
            //__m256i g_buf_entries = _mm256_load_si256((__m256i*)albedo_buf_ptr);
            __m256i world_pos_entries = _mm256_load_si256((__m256i*)world_pos_ptr);
            __m256 norm_pt1_entries = _mm256_load_ps(norm_ptr);
            __m256 norm_pt2_entries = _mm256_load_ps((norm_ptr+8));
            __m256i drawn_mask_replicated = _mm256_set1_epi32(drawn_mask);

            // 0b00110011
            // needs to be 0x00000 0x1111

            __m256i wide_mask_shifted = _mm256_sllv_epi32(_mm256_and_si256(selectors, drawn_mask_replicated), shifters);
            __m256i norm_pt1_mask = _mm256_sllv_epi32(_mm256_and_si256(norm_pt1_selectors, drawn_mask_replicated), norm_pt1_shifters);
            __m256i norm_pt2_mask = _mm256_sllv_epi32(_mm256_and_si256(norm_pt2_selectors, drawn_mask_replicated), norm_pt2_shifters);

            __m256i blended_world_pos_entries = (__m256i)_mm256_blendv_ps((__m256)undrawn_vec, (__m256)world_pos_entries, (__m256)wide_mask_shifted);
            __m256 blended_norm_pt1_entries = _mm256_blendv_ps((__m256)undrawn_norm_pt1_vec, norm_pt1_entries, (__m256)norm_pt1_mask);
            __m256 blended_norm_pt2_entries = _mm256_blendv_ps((__m256)undrawn_norm_pt2_vec, norm_pt2_entries, (__m256)norm_pt2_mask);
            
            _mm256_store_si256((__m256i*)world_pos_ptr, blended_world_pos_entries);
            _mm256_store_ps(norm_ptr, blended_norm_pt1_entries);
            _mm256_store_ps((norm_ptr+8), blended_norm_pt2_entries);
            norm_ptr += 16;
            world_pos_ptr += 8;
        }
        //EndCountedTimeBlock(coverage_fill_empty_entries, max_y-min_y);
    }
    EndTimeBlock(coverage_fill_empty_entries);

}

u64 total_vector_samples;
u64 reused_vector_samples;

#if 0
void render_vector(s32 min_x, s32 min_y, s32 max_x, s32 max_y) {
    u32* cmap = (u32*)colormap;
    f32 fog_r = (background_color&0xFF);
    f32 fog_g = ((background_color>>8)&0xFF);
    f32 fog_b = ((background_color>>16)&0xFF);
    __m256 fog_r_vec = _mm256_set1_ps(fog_r);
    __m256 fog_g_vec = _mm256_set1_ps(fog_g);
    __m256 fog_b_vec = _mm256_set1_ps(fog_b);
    __m256 max_z_vec = _mm256_set1_ps(max_z);
    f32 wrap_pos_x = pos_x;
    while(wrap_pos_x < 0) {
        wrap_pos_x += 1024;
    }
    while(wrap_pos_x > 1024) {
        wrap_pos_x -= 1024;
    }
        
    f32 wrap_pos_y = pos_y;//fmod(pos_y, 1024);
    while(wrap_pos_y < 0) {
        wrap_pos_y += 1024;
    }
    while(wrap_pos_y > 1024) {
        wrap_pos_y -= 1024;
    }
    __m256 wrap_pos_x_vec = _mm256_set1_ps(wrap_pos_x);
    __m256 wrap_pos_y_vec = _mm256_set1_ps(wrap_pos_y);


    f32 pitch = (sinf(pitch_ang)*render_size)+(render_size/2);


    //int aligned_min_x = min_x & (~0b111);
    int aligned_min_x = min_x;
    //while(aligned_min_x & 0b111) { aligned_min_x--; }
    __m256i xs = _mm256_setr_epi32(aligned_min_x, aligned_min_x+1, aligned_min_x+2, aligned_min_x+3, aligned_min_x+4, 
    aligned_min_x+5, aligned_min_x+6, aligned_min_x+7);
    __m256i dx_vector = _mm256_set1_epi32(8);
    __m256i one_vec = _mm256_set1_epi32(1);
    __m256i zero_vec = _mm256_set1_epi32(0);
    __m256i negative_one_vec = _mm256_set1_epi32(-1);
    __m256 one_ps_vec = _mm256_set1_ps(1);
    __m256 two_ps_vec = _mm256_set1_ps(2);
    __m256 zero_ps_vec = _mm256_set1_ps(0);

    __m256 render_size_ps_vec = _mm256_set1_ps(render_size);

    __m256i render_size_vec = _mm256_set1_epi32(render_size);

    __m256 dir_x_vec = _mm256_set1_ps(dir_x);
    __m256 dir_y_vec = _mm256_set1_ps(dir_y);
    __m256 plane_x_vec = _mm256_set1_ps(plane_x);
    __m256 plane_y_vec = _mm256_set1_ps(plane_y);
    __m256i pitch_vec = _mm256_set1_epi32(pitch);

    __m256 height_vec = _mm256_set1_ps(height);
    __m256 scale_height_vec = _mm256_set1_ps(scale_height);

    __m256i MaskFF_vec = _mm256_set1_epi32(0xFF);
    __m256i OpaqueAlpha = _mm256_set1_epi32(0xFF000000);

    f32 one_over_max_z = 1/max_z;
    __m256 one_over_max_z_vec = _mm256_set1_ps(one_over_max_z);


    f32 sun_vec_x = 0;
    f32 sun_vec_y = sinf(sun_ang);
    f32 sun_vec_z = cosf(sun_ang);

    total_vector_samples = 0;
    reused_vector_samples = 0;
    for(int x = aligned_min_x; x <= max_x; x += 8) {
        __m256i min_y_vec = _mm256_set1_epi32(min_y);
        int prev_drawn_max_y = max_y+1;//render_size;
        __m256i prev_drawn_max_ys = _mm256_set1_epi32(max_y+1);

        f32 camera_space_x = ((2 * x) / ((f32)render_size)) - 1; //x-coordinate in camera space
        __m256 double_xs = _mm256_mul_ps(_mm256_set1_ps(2), _mm256_cvtepi32_ps(xs));
        __m256 camera_space_xs = _mm256_sub_ps(_mm256_div_ps(double_xs,
                                                             render_size_ps_vec),
                                                one_ps_vec);

        f32 ray_dir_x = dir_x + (plane_x * camera_space_x);
        f32 ray_dir_y = dir_y + (plane_y * camera_space_x);
        __m256 ray_dir_xs = _mm256_add_ps(dir_x_vec, _mm256_mul_ps(plane_x_vec, camera_space_xs));
        __m256 ray_dir_ys = _mm256_add_ps(dir_y_vec, _mm256_mul_ps(plane_y_vec, camera_space_xs));
        //which box of the map we're in
        s32 map_x = ((s32)wrap_pos_x);
        s32 map_y = ((s32)wrap_pos_y);
        __m256i map_xs = _mm256_set1_epi32(wrap_pos_x);
        __m256i map_ys = _mm256_set1_epi32(wrap_pos_y);


        //length of ray from one x or y-side to next x or y-side
        __m256 one_power_of_30_vec = _mm256_set1_ps(1e30);  
        __m256 ray_dir_x_zeros = _mm256_cmp_ps(ray_dir_xs, zero_ps_vec, _CMP_EQ_UQ);
        __m256 ray_dir_y_zeros = _mm256_cmp_ps(ray_dir_ys, zero_ps_vec, _CMP_EQ_UQ);
        __m256 swapped_ray_dir_xs = _mm256_blendv_ps(ray_dir_xs, one_ps_vec, ray_dir_x_zeros);
        __m256 swapped_ray_dir_ys = _mm256_blendv_ps(ray_dir_ys, one_ps_vec, ray_dir_y_zeros);
        __m256 delta_dist_xs_intermediate = abs_ps(_mm256_div_ps(one_ps_vec, swapped_ray_dir_xs));
        __m256 delta_dist_ys_intermediate = abs_ps(_mm256_div_ps(one_ps_vec, swapped_ray_dir_ys));
        __m256 delta_dist_xs = _mm256_blendv_ps(delta_dist_xs_intermediate, one_power_of_30_vec, ray_dir_x_zeros);
        __m256 delta_dist_ys = _mm256_blendv_ps(delta_dist_ys_intermediate, one_power_of_30_vec, ray_dir_y_zeros);

        f32 wrap_x_minus_map_x = (wrap_pos_x - map_x);
        f32 map_x_plus_one_minus_wrap_x = (map_x + (1.0 - wrap_pos_x));
        f32 wrap_y_minus_map_y = (wrap_pos_y - map_y);
        f32 map_y_plus_one_minus_wrap_y = (map_y + (1.0 - wrap_pos_y));
        __m256 wrap_x_minus_map_x_vec = _mm256_set1_ps(wrap_x_minus_map_x);
        __m256 map_x_plus_one_minus_wrap_x_vec = _mm256_set1_ps(map_x_plus_one_minus_wrap_x);
        __m256 wrap_y_minus_map_y_vec = _mm256_set1_ps(wrap_y_minus_map_y);
        __m256 map_y_plus_one_minus_wrap_y_vec = _mm256_set1_ps(map_y_plus_one_minus_wrap_y);




        __m256 ray_dir_xs_less_than_zero = _mm256_cmp_ps(ray_dir_xs, zero_ps_vec, _CMP_LT_OQ);
        __m256 ray_dir_ys_less_than_zero = _mm256_cmp_ps(ray_dir_ys, zero_ps_vec, _CMP_LT_OQ);
        //what direction to step in x or y-direction (either +1 or -1)
        __m256i step_xs = _mm256_blendv_epi8(one_vec, negative_one_vec, (__m256i)ray_dir_xs_less_than_zero);
        __m256i step_ys = _mm256_blendv_epi8(one_vec, negative_one_vec, (__m256i)ray_dir_ys_less_than_zero);

        //length of ray from current position to next x or y-side
        __m256 side_dist_xs = _mm256_mul_ps(_mm256_blendv_ps(map_x_plus_one_minus_wrap_x_vec, wrap_x_minus_map_x_vec, ray_dir_xs_less_than_zero),
                                            delta_dist_xs);
        __m256 side_dist_ys = _mm256_mul_ps(_mm256_blendv_ps(map_y_plus_one_minus_wrap_y_vec, wrap_y_minus_map_y_vec, ray_dir_ys_less_than_zero), 
                                            delta_dist_ys);

        __m256 perp_wall_dists = _mm256_set1_ps(0);
        __m256 prev_perp_wall_dists = _mm256_set1_ps(0);

        int dist_lte_max_z_mask = _mm256_movemask_ps(_mm256_cmp_ps(perp_wall_dists, max_z_vec, _CMP_LE_OQ));
        int prev_draw_y_gt_min_y_mask = _mm256_movemask_epi8(_mm256_cmpgt_epi32(prev_drawn_max_ys, min_y_vec));

    #if 1
        __m256 near_invzs = _mm256_set1_ps(0);
        while(dist_lte_max_z_mask && prev_draw_y_gt_min_y_mask) { 

    #else 
        while(dist_lte_max_z_mask && prev_draw_y_gt_min_y_mask) { 
            __m256i idxs = get_swizzled_map_idx_256(map_xs, map_ys);   

    #endif      

            prev_perp_wall_dists = perp_wall_dists;

            __m256 side_dist_xs_less_than_ys = _mm256_cmp_ps(side_dist_xs, side_dist_ys, _CMP_LT_OQ);
            perp_wall_dists = _mm256_blendv_ps(side_dist_ys, side_dist_xs, side_dist_xs_less_than_ys);


            __m256i map_xs_dxs = _mm256_blendv_epi8(zero_vec, step_xs, (__m256i)side_dist_xs_less_than_ys);
            __m256i map_ys_dys = _mm256_blendv_epi8(step_ys, zero_vec, (__m256i)side_dist_xs_less_than_ys);


            int map_xs_stash[8] __attribute__ ((aligned (16))); _mm256_store_si256((__m256i*)map_xs_stash, map_xs);
            int map_ys_stash[8] __attribute__ ((aligned (16))); _mm256_store_si256((__m256i*)map_ys_stash, map_ys);
            map_xs = _mm256_add_epi32(map_xs, map_xs_dxs);
            map_ys = _mm256_add_epi32(map_ys, map_ys_dys);

            __m256 side_dist_xs_dxs = _mm256_blendv_ps(zero_ps_vec, delta_dist_xs, side_dist_xs_less_than_ys);
            __m256 side_dist_ys_dys = _mm256_blendv_ps(delta_dist_ys, zero_ps_vec, side_dist_xs_less_than_ys);
            side_dist_xs = _mm256_add_ps(side_dist_xs, side_dist_xs_dxs);
            side_dist_ys = _mm256_add_ps(side_dist_ys, side_dist_ys_dys);


            __m256 perp_wall_dists_zero_mask = _mm256_cmp_ps(perp_wall_dists, zero_ps_vec, _CMP_EQ_UQ);
            __m256 swapped_perp_wall_dists = _mm256_blendv_ps(perp_wall_dists, one_ps_vec, perp_wall_dists_zero_mask);
            __m256 far_invzs = _mm256_div_ps(scale_height_vec, swapped_perp_wall_dists);
            
            __m256 one_minus_fogs;
            __m256 mult_fog_rs, mult_fog_gs, mult_fog_bs;
            if(fog) {
                __m256 avg_dists = perp_wall_dists; //(_mm256_add_ps(prev_perp_wall_dists, perp_wall_dists), two_ps_vec);
                __m256 z_to_1s = _mm256_mul_ps(avg_dists, one_over_max_z_vec);
                z_to_1s = _mm256_mul_ps(z_to_1s, z_to_1s);
                __m256 fog_factors = lerp256(zero_ps_vec, z_to_1s, one_ps_vec);
                one_minus_fogs = _mm256_sub_ps(one_ps_vec, fog_factors);
                mult_fog_rs = _mm256_mul_ps(fog_r_vec, fog_factors);
                mult_fog_gs = _mm256_mul_ps(fog_g_vec, fog_factors);
                mult_fog_bs = _mm256_mul_ps(fog_b_vec, fog_factors);
            }
        #if 1
    
            f32 near_invzs_stash[8] __attribute__ ((aligned (16))); _mm256_store_ps(near_invzs_stash, near_invzs); 
            f32 far_invzs_stash[8] __attribute__ ((aligned (16))); _mm256_store_ps(far_invzs_stash, far_invzs); 
            int min_y_vec_stash[8] __attribute__ ((aligned (16))); _mm256_store_si256((__m256i*)min_y_vec_stash, min_y_vec);
            int prev_drawn_max_ys_stash[8] __attribute__ ((aligned (16))); _mm256_store_si256((__m256i*)prev_drawn_max_ys_stash, prev_drawn_max_ys);
            f32 one_minus_fogs_stash[8] __attribute__ ((aligned (16))); _mm256_store_ps(one_minus_fogs_stash, one_minus_fogs);
            f32 mult_fog_rs_stash[8] __attribute__ ((aligned (16))); _mm256_store_ps(mult_fog_rs_stash, mult_fog_rs);
            f32 mult_fog_gs_stash[8] __attribute__ ((aligned (16))); _mm256_store_ps(mult_fog_gs_stash, mult_fog_gs);
            f32 mult_fog_bs_stash[8] __attribute__ ((aligned (16))); _mm256_store_ps(mult_fog_bs_stash, mult_fog_bs);



            for(int i = 0; i < 8; i++) { //7; i >= 0; i--) {
                int ix = x+i;
                int map_x = map_xs_stash[i];
                int map_y = map_ys_stash[i];
                int voxelmap_idx = get_voxelmap_idx(map_x, map_y);
                
                f32 color_norm_scale;
                if(lighting) {
                    f32 norm_x = depthmap_normal_xs[get_swizzled_map_idx(map_x, map_y)];
                    f32 norm_y = depthmap_normal_ys[get_swizzled_map_idx(map_x, map_y)];
                    f32 norm_z = depthmap_normal_zs[get_swizzled_map_idx(map_x, map_y)];
                    f32 dot_light = (((norm_x*sun_vec_x+norm_y*sun_vec_y+norm_z*sun_vec_z)+1)/2); // 0 to 1
                    //f32 color_norm_scale = dot_light+amb_light_factor;
                    dot_light *= dot_light;
                    color_norm_scale = dot_light; //min(1.0, dot_light*amb_light_factor+.1);
                    color_norm_scale = min(1.0, dot_light+amb_light_factor);
                    //f32 color_norm_scale = (((norm_x*1+norm_y*-1+norm_z*1)+1)/4)+.5;
                }
                column_header* header = &columns_header_data[voxelmap_idx];

                // WRITE TO MEMORY AND RELOAD IN AVX VARIABLE!
                f32 near_invz = near_invzs_stash[i];
                f32 far_invz = far_invzs_stash[i];
                int next_drawable_min_y = min_y_vec_stash[i];
                int col_prev_drawn_max_y = prev_drawn_max_ys_stash[i];
                f32 one_minus_fog = one_minus_fogs_stash[i];
                f32 mult_fog_r = mult_fog_rs_stash[i];
                f32 mult_fog_g = mult_fog_gs_stash[i];
                f32 mult_fog_b = mult_fog_bs_stash[i];
                u8 num_runs = header->num_runs;
                int chunk_top = -1;

                for(int run = 0; run < min(3, num_runs); run++) {                
                    u16 color16 = header->first_three_run_colors[run];
                    u8 run_skip = header->first_three_runs[(run<<1)];
                    u8 run_len = header->first_three_runs[(run<<1)+1];
                    u32 abgr;
                    EXPAND_COLOR_AND_FOG(abgr, color16, color_norm_scale, one_minus_fog, mult_fog_r, mult_fog_g, mult_fog_b);
                    int chunk_bot = chunk_top+run_skip;

                    chunk_top = chunk_bot + run_len;
                    f32 relative_bot = height-chunk_bot;
                    f32 relative_top = height-chunk_top;
                    f32 top_invz = (relative_top < 0 ? near_invz : far_invz);
                    f32 bot_invz = (relative_bot < 0 ? far_invz : near_invz);
                    f32 float_top_projected_height = relative_top*top_invz;
                    f32 float_bot_projected_height = relative_bot*bot_invz;
                    
                    s32 int_top_projected_height = floor(float_top_projected_height);
                    s32 int_bot_projected_height = floor(float_bot_projected_height);
                    s32 top_heightonscreen = int_top_projected_height + pitch;
                    s32 bot_heightonscreen = int_bot_projected_height + pitch;

                    if(bot_heightonscreen < next_drawable_min_y) { continue; }
                    int next_prev_drawn_max_y = (bot_heightonscreen >= col_prev_drawn_max_y && top_heightonscreen < col_prev_drawn_max_y) ? top_heightonscreen : col_prev_drawn_max_y;
                    int next_next_drawable_min_y = (top_heightonscreen <= next_drawable_min_y && bot_heightonscreen > next_drawable_min_y) ? bot_heightonscreen+1 : next_drawable_min_y;

                    
                    top_heightonscreen = max(next_drawable_min_y, top_heightonscreen);
                    bot_heightonscreen = min(col_prev_drawn_max_y-1, bot_heightonscreen);

                    if(top_heightonscreen <= bot_heightonscreen) {
                        u32 fb_idx = fb_swizzle(ix,top_heightonscreen);

                        for(int y = top_heightonscreen; y <= bot_heightonscreen; y++) {
                            u8 occlusion_bit = get_occlusion_bit(ix, y);
                            u32 old_abgr = inter_buffer[fb_idx];
                            u32 new_abgr = occlusion_bit ? old_abgr : abgr;
                            inter_buffer[fb_idx] = new_abgr;
                            set_occlusion_bit(ix, y);

                            fb_idx += 8;
                        }
                        col_prev_drawn_max_y = next_prev_drawn_max_y;
                        next_drawable_min_y = next_next_drawable_min_y;
                    }
                }

                // WRITE TO MEMORY AND RELOAD IN AVX VARIABLE!
                prev_drawn_max_ys_stash[i] = col_prev_drawn_max_y;
                min_y_vec_stash[i] = next_drawable_min_y;
            }
            near_invzs = far_invzs;

            prev_drawn_max_ys = _mm256_load_si256((__m256i*)prev_drawn_max_ys_stash);
            min_y_vec = _mm256_load_si256((__m256i*)min_y_vec_stash);

            //prev_drawn_max_ys = _mm256_blendv_epi8(prev_drawn_max_ys, heights_on_screen, shown_mask);
            prev_draw_y_gt_min_y_mask = _mm256_movemask_epi8(_mm256_cmpgt_epi32(prev_drawn_max_ys, min_y_vec));   
            dist_lte_max_z_mask = _mm256_movemask_ps(_mm256_cmp_ps(perp_wall_dists, max_z_vec, _CMP_LE_OQ));   
        #else
            __m256i depths = _mm256_i32gather_epi32(depthmap_u32s, idxs, 4);
            __m256i abgrs = _mm256_i32gather_epi32(cmap, idxs, 4);   
            __m256i depths = _mm256_i32gather_epi32(depthmap_u32s, idxs, 4);
            __m256i abgrs = _mm256_i32gather_epi32(cmap, idxs, 4);
            
            __m256 rs = _mm256_cvtepi32_ps(_mm256_and_si256(abgrs, MaskFF_vec));
            __m256 gs = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(abgrs, 8), MaskFF_vec));
            __m256 bs = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(abgrs, 16), MaskFF_vec));

            __m256 relative_heights = _mm256_sub_ps(height_vec, _mm256_cvtepi32_ps(depths));
            
            __m256 swapped_prev_perp_wall_dists = _mm256_blendv_ps(prev_perp_wall_dists, one_ps_vec, prev_perp_wall_dists_zero_mask);
            __m256 swapped_perp_wall_dists = _mm256_blendv_ps(perp_wall_dists, one_ps_vec, perp_wall_dists_zero_mask);

            __m256 canonical_dists = _mm256_blendv_ps(perp_wall_dists, prev_perp_wall_dists, _mm256_cmp_ps(relative_heights, zero_ps_vec, _CMP_LT_OQ));
            __m256 invzs = _mm256_div_ps(scale_height_vec, canonical_dists);
            
            __m256 float_projected_heights = _mm256_mul_ps(relative_heights, invzs);
            
            if(fog) {
                __m256 avg_dists = _mm256_div_ps(_mm256_add_ps(prev_perp_wall_dists, perp_wall_dists), two_ps_vec);
                __m256 z_to_1s = _mm256_mul_ps(avg_dists, one_over_max_z_vec);
                z_to_1s = _mm256_mul_ps(z_to_1s, z_to_1s);
                __m256 fog_factors = lerp256(zero_ps_vec, z_to_1s, one_ps_vec);
                __m256 one_minus_fogs = _mm256_sub_ps(one_ps_vec, fog_factors);
                __m256 mult_fog_rs = _mm256_mul_ps(fog_r_vec, fog_factors);
                __m256 mult_fog_gs = _mm256_mul_ps(fog_g_vec, fog_factors);
                __m256 mult_fog_bs = _mm256_mul_ps(fog_b_vec, fog_factors);


                rs =  _mm256_add_ps(_mm256_mul_ps(rs, one_minus_fogs), mult_fog_rs);
                gs =  _mm256_add_ps(_mm256_mul_ps(gs, one_minus_fogs), mult_fog_gs);
                bs =  _mm256_add_ps(_mm256_mul_ps(bs, one_minus_fogs), mult_fog_bs);
            }

            __v8su Intr = (__v8su)_mm256_cvtps_epi32(rs);
            __v8su Intg = (__v8su)_mm256_cvtps_epi32(gs);
            __v8su Intb = (__v8su)_mm256_cvtps_epi32(bs);

            __v8su Sr = Intr;
            __v8su Sg = (__v8su)_mm256_slli_epi32((__m256i)Intg, 8);
            __v8su Sb = (__v8su)_mm256_slli_epi32((__m256i)Intb, 16);
            __v8su Sa = (__v8su)OpaqueAlpha;
            abgrs = _mm256_or_si256(_mm256_or_si256((__m256i)Sr, (__m256i)Sg), _mm256_or_si256((__m256i)Sb, OpaqueAlpha));




            __m256i int_projected_heights = _mm256_cvtps_epi32(_mm256_floor_ps(float_projected_heights));
            __m256i heights_on_screen = _mm256_add_epi32(int_projected_heights, pitch_vec);
            heights_on_screen = _mm256_max_epi32(min_y_vec, heights_on_screen);         
            
            __m256i too_close_mask = _mm256_or_si256((__m256i)prev_perp_wall_dists_zero_mask, (__m256i)perp_wall_dists_zero_mask);
            //int too_close_mask = _mm256_movemask_ps(prev_perp_wall_dists_zero_mask) | _mm256_movemask_ps(perp_wall_dists_zero_mask);
            __m256i not_clipped_mask = _mm256_cmpgt_epi32(prev_drawn_max_ys, heights_on_screen);

            __m256i shown_mask = _mm256_andnot_si256(too_close_mask, not_clipped_mask);

            __v8si max_int_vec = (__v8si)_mm256_set1_epi32(SDL_MAX_SINT32);
            __v8si min_shown_heights = (__v8si)_mm256_blendv_epi8((__m256i)max_int_vec, (__m256i)heights_on_screen, (__m256i)shown_mask);
            
            s32 min_height = horizontal_min_256_epi32(min_shown_heights);
            __v8si min_int_vec = (__v8si)_mm256_set1_epi32(SDL_MIN_SINT32);
            __v8si max_shown_prevy = (__v8si)_mm256_blendv_epi8((__m256i)min_int_vec, (__m256i)prev_drawn_max_ys, (__m256i)shown_mask);
            s32 max_height = horizontal_max_256_epi32(max_shown_prevy);

            __v8si y_vec = (__v8si)_mm256_set1_epi32(min_height);
            __v8si heights_minus_one = (__v8si)_mm256_sub_epi32((__m256i)heights_on_screen, (__m256i)one_vec);
            u32 fb_idx = fb_swizzle(x,min_height);
            u32* ptr = &inter_buffer[fb_idx];


            for(int y = min_height; y < max_height; y++) {
                __v8si within_span =(__v8si) _mm256_cmpgt_epi32((__m256i)y_vec, (__m256i)heights_minus_one);
                __v8si not_hidden = (__v8si)_mm256_cmpgt_epi32((__m256i)prev_drawn_max_ys, (__m256i)y_vec);
                __v8si pixels_shown = (__v8si)_mm256_and_si256((__m256i)within_span, (__m256i)not_hidden);
                __v8su old_abgrs = (__v8su)_mm256_load_si256((__m256i*)ptr);
                __v8su new_pixels = (__v8su)_mm256_blendv_epi8((__m256i)old_abgrs, (__m256i)abgrs, (__m256i)pixels_shown);
                _mm256_store_si256((__m256i*)ptr, (__m256i)new_pixels);
                ptr += 8;
                y_vec = (__v8si)_mm256_add_epi32((__m256i)y_vec, (__m256i)one_vec); 
            }      
            
            prev_drawn_max_ys = _mm256_blendv_epi8(prev_drawn_max_ys, heights_on_screen, shown_mask);
            dist_lte_max_z_mask = _mm256_movemask_ps(_mm256_cmp_ps(perp_wall_dists, max_z_vec, _CMP_LE_OQ));
            prev_draw_y_gt_min_y_mask = _mm256_movemask_epi8(_mm256_cmpgt_epi32(prev_drawn_max_ys, min_y_vec));      
        #endif
        }
        
        xs = _mm256_add_epi32(xs, dx_vector);
    }

}

#endif 