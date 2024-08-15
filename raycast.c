#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "types.h"

#define DAY_BACKGROUND_COLOR 0xFFBBAE67
#define NIGHT_BACKGROUND_COLOR 0xFF000000//0xFF632D0F
#define DAY_AMB_LIGHT_FACTOR .7
#define NIGHT_AMB_LIGHT_FACTOR .05


#define DAY_MAX_Z 512
#define NIGHT_MAX_Z 512
#define NO_FOG_MAX_Z 2048
    
#define PI 3.14159265359

#define PROFILER 0
#define NUM_THREADS 8

#include "selectable_profiler.c"
#include <SDL2/SDL.h>


#include "utils.h"

u32 background_color = DAY_BACKGROUND_COLOR;
f32 amb_light_factor = .4;


u32 render_size = 0;



#define OUTPUT_WIDTH  1920
#define OUTPUT_HEIGHT 1080

#define DEFAULT_VOXEL_COLOR (0x002F6B | (0b11<<24))


f32 magnitude_vector(f32 x, f32 y, f32 z) {
    return sqrtf((x*x)+(y*y)+(z*z));
}

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




f32 lerp(f32 a, f32 t, f32 b);


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

float deg_to_rad(float degrees) {
    return degrees * PI / 180.0;
}

f32 desired_fov_degrees = 120;


f32 plane_x = 0.0;
f32 plane_y;// = -1.20;
static f32 height = 200.0;

static int plane_parameters_setup = 0;

void setup_ray_plane_parameters() {
    dir_x = 1.0;
    dir_y = 0.0;
    plane_x = 0.0;
    plane_y = - tanf(deg_to_rad(desired_fov_degrees/2.0));
    plane_parameters_setup = 1;
}

double fabs(double x);
f32 sqrtf(f32 x);
f32 atan2f(f32 y, f32 x);
f32 sinf(f32 x);
f32 cosf(f32 x);
double asin(double x);

f32 dt;
f32 pitch_ang = 0;
f32 sun_ang = 0;

int mouse_captured = 0;

// roll is a combination of the base roll, controlled by the Q/E keys, 
// and mouse roll, which is a slight roll added when you rapidly turn left or right
f32 roll = 0;//-1.57;
f32 baseroll = 0;
f32 mouseroll = 0;



f32 forwardbackward = 0;
f32 leftright = 0;
f32 strafe = 0;
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
    STATIC_LIGHTING = 1,
    DYNAMIC_LIGHTING = 2
    //FANCY_LIGHTING = 2,
} lighting_modes;

char* lighting_mode_strs[] = {
    "disabled",
    "static",
    "dynamic"
};

// currently these are the same
typedef enum {
    NO_FOG = 0,
    FOG = 1,
} fog_modes;

char* fog_mode_strs[] = {
    "disabled",
    "enabled",
};

char* transparency_mode_strs[] = {
    "disabled",
    "enabled"
};

f32 total_time;
static int frame = 0;
static int vector = 0;//1;
static int fogmode = FOG;
static int lighting = STATIC_LIGHTING;
static int transparency = 0;
static int ambient_occlusion = 1;
static int vectorized_rendering = 0;

static int view = VIEW_STANDARD;
static int gravmode = 0;

static int debug = 0;

static int double_pixels = 1;

int render_width = OUTPUT_WIDTH/2;   
int render_height = OUTPUT_HEIGHT/2;
f32 aspect_ratio = (OUTPUT_WIDTH/2.0)/(OUTPUT_HEIGHT/2.0);

static int swizzled = 0;
static int setup_render_size = 0;


f32 max_z = DAY_MAX_Z;

int cur_map = 0;
static int map_loaded = 0;


#include "thread_pool.h"

typedef struct {
    s32 min_x, min_y;
    s32 max_x, max_y;
    volatile uint64_t finished;
} thread_params;

typedef void (*raw_render_func)(s32 min_x, s32 min_y, s32 max_x, s32 max_y);

typedef struct {
    int num_jobs;
    PTP_WORK_CALLBACK func;
    raw_render_func raw_func;
    thread_params *parms;
} render_pool;

void start_pool(thread_pool* tp, render_pool* job_pool) {
    for(int i = 0; i < job_pool->num_jobs; i++) {
        if(NUM_THREADS > 1) {
            thread_pool_add_work(tp, job_pool->func, (void*)&job_pool->parms[i]);
        } else {        
            // non-threading
            job_pool->raw_func(
                job_pool->parms[i].min_x, 
                job_pool->parms[i].min_y,
                job_pool->parms[i].max_x,
                job_pool->parms[i].max_y
            );
        }
    }
}
void wait_for_render_pool_to_finish(render_pool* p) {
    if(NUM_THREADS > 1) {
        while(1) {
            top_of_wait_loop:;
            for(int i = 0; i < p->num_jobs; i++) {
                if(p->parms[i].finished == 0) { goto top_of_wait_loop; }
            }
            break;
        }
    }
}

thread_pool* pool;

#include "voxelmap.c"

void next_map() {
    load_map(cur_map++);
}

void reload_map() {
    load_map(--cur_map); 
}




void handle_keyup(SDL_KeyboardEvent key) {
    switch(key.keysym.scancode) {
        case SDL_SCANCODE_Q:
            rollleftright = 0;
            break;
        case SDL_SCANCODE_E:
            rollleftright = 0;
            break;

        case SDL_SCANCODE_W:
            forwardbackward = 0;
            break;
        case SDL_SCANCODE_S:
            forwardbackward = 0;
            break;
        case SDL_SCANCODE_A: 
            strafe = 0;
            break;
        case SDL_SCANCODE_D: 
            strafe = 0;
            break;
        case SDL_SCANCODE_Z:
            updown = 0;
            break;
        case SDL_SCANCODE_X:
            updown = 0;
            break;
        case SDL_SCANCODE_L:
            lighting++;
            if(lighting > 1) { lighting = 0; }
            //lighting = !lighting;
            break;
        case SDL_SCANCODE_O:
            ambient_occlusion = !ambient_occlusion;
            break;
        case SDL_SCANCODE_V:
            view++;
            if(view > 2) { view = 0;}
            break;
        case SDL_SCANCODE_B:
            vectorized_rendering = !vectorized_rendering;
            break;
        case SDL_SCANCODE_F:
            fogmode++;
            if(fogmode > 1) {
                fogmode = 0;
             }
            break;
        case SDL_SCANCODE_G:
            gravmode = !gravmode;
            break;
        case SDL_SCANCODE_N:
            next_map();
            break;
        case SDL_SCANCODE_R: do {
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
            aspect_ratio = ((f32)render_width)/render_height;
            setup_render_size = 0;
        } while(0);
        break;

        case SDL_SCANCODE_T:
            transparency = !transparency;
            break;

        case SDL_SCANCODE_ESCAPE: do {
            if(mouse_captured) {
                mouse_captured = 0;
                SDL_CaptureMouse(SDL_FALSE);
                SDL_ShowCursor(1);
            }
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

        case SDL_SCANCODE_W:
            forwardbackward = .4;
            break;
        case SDL_SCANCODE_S:
            forwardbackward = -.4;
            break;
        case SDL_SCANCODE_A: 
            strafe = -1.1;
            break;
        case SDL_SCANCODE_D: 
            strafe = +1.1;
            break;
        case SDL_SCANCODE_Z:
            updown = -1.0;
            break;
        case SDL_SCANCODE_X:
            updown = +1.0;
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

                    if(voxel_is_solid(x, y, h) || h >= cur_map_max_height) {
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
        f32 rot_speed = dt * leftright * 1.1;

        f32 old_dir_x = dir_x;
        dir_x = dir_x * cos(rot_speed) - dir_y * sin(rot_speed);
        dir_y = old_dir_x * sin(rot_speed) + dir_y * cos(rot_speed);
        f32 old_plane_x = plane_x;
        plane_x = plane_x * cos(rot_speed) - plane_y * sin(rot_speed);
        plane_y = old_plane_x * sin(rot_speed) + plane_y * cos(rot_speed);
    }

    pitch_ang += dt*0.017*5 * lookupdown;// -= dt*400;
    //pitch_ang = min(max(pitch_ang, -1.57), 1.57);//-.45), .45);


    
    if(strafe) {
           f32 new_pos_x = pos_x + dir_y * strafe * dt * 15;
           f32 new_pos_y = pos_y + (dir_x * -1) * strafe * dt * 15;
           pos_x = new_pos_x;
           pos_y = new_pos_y;
    }

    if(forwardbackward) {
        f32 new_pos_x = pos_x + dir_x * forwardbackward * dt * 100;
        f32 new_pos_y = pos_y + dir_y * forwardbackward * dt * 100;
        f32 new_pos_z = height + (forwardbackward * 2*sinf(pitch_ang)) * dt * 100;
        pos_x = new_pos_x;
        pos_y = new_pos_y;
        height = new_pos_z;
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
                //printf("X COLLISION!\n");
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
                //printf("X COLLISION!\n");
            }
        }

        
    }

    if(updown) {
        height += dt*updown*50;
    }
    baseroll -= rollleftright*dt*1.2;
    roll = baseroll + mouseroll;  
}






int mouse_is_down = 0;



static u32* inter_buffer = NULL;
static u32* base_inter_buffer = NULL;

static u8* occlusion_buffer = NULL;
static u8* base_occlusion_buffer = NULL;



int right_mouse_down = 0;
int cur_mouse_x, cur_mouse_y;

void handle_left_mouse_down() {
    if(!mouse_captured) {
        printf("capturing mouse\n");
        mouse_captured = 1;
        int err = SDL_CaptureMouse(SDL_TRUE);
        SDL_ShowCursor(0);
    }
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

void handle_mouse_input(f32 dt) {
    if(!mouse_captured) { 
        mouseroll = 0;
        lookupdown = 0;
        leftright = 0;
        return; 
    }
    s32 centerX = OUTPUT_WIDTH/2;
    s32 centerY = OUTPUT_HEIGHT/2;
    s32 dx = abs(centerX - cur_mouse_x) < 150 ? 0 : (centerX - cur_mouse_x);
    s32 dy = abs(centerY - cur_mouse_y) < 100 ? 0 : (centerY - cur_mouse_y);

    f32 lerp_term = (-dx/(OUTPUT_WIDTH/2.0));

    mouseroll = lerp(0, lerp_term*lerp_term, .25);
    if(dx > 0) { mouseroll = -mouseroll; }

    if(dx) {
        f32 leftrightportion = dx*cos(baseroll);
        f32 updownportion = dx*sinf(baseroll);
        leftright = (f32)leftrightportion / OUTPUT_WIDTH * 2;
        lookupdown = updownportion*.010;
    } else {
        leftright = 0;
        lookupdown = 0;
    }

    if(dy) {
        
        lookupdown += dy*cosf(baseroll)*.012;
        leftright += -dy*sinf(baseroll)*.004;

    } else {
        if(!dx) { lookupdown = 0; }
    }

}

void update_mouse_pos(int mouse_x, int mouse_y) {
    cur_mouse_x = mouse_x;
    cur_mouse_y = mouse_y;

}


#include "vc.c"

double fmod(double x, double y);


static u32* pixels = NULL; 



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


s32 horizontal_min_256_epi32(__m256i v) {
    __m128i i = _mm256_extractf128_si256(v, 1 );
    // compare lower and upper halves, get min(0,4), min(1,5), min(2,6), min(3,7)
    i = _mm_min_epi32(i, _mm256_castsi256_si128(v ) ); 
     // compare lower and upper 64-bit halves, get min(min(0,4), min(2,6)), min(min(1,5), min(3,7))
    i = _mm_min_epi32(i, _mm_shuffle_epi32(i, 0b00001110 ) ); 
    return min(i[0], i[1]);
}

s32 horizontal_max_256_epi32(__m256i v) {
    __m128i i = _mm256_extractf128_si256(v, 1 );
    // compare lower and upper halves, get min(0,4), min(1,5), min(2,6), min(3,7)
    i = _mm_max_epi32(i, _mm256_castsi256_si128( v ) ); 
     // compare lower and upper 64-bit halves, get min(min(0,4), min(2,6)), min(min(1,5), min(3,7))
    i = _mm_max_epi32(i, _mm_shuffle_epi32( i, 0b00001110 ) ); 
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
    
    f32 sinroll = sinf(angle);
    f32 cosroll = cosf(angle);            
    f32 yy = (s32)(input_y-render_height/2);
    f32 xx = (s32)(input_x-render_width/2);
    f32 temp_xx = xx * cosroll - yy * sinroll;
    f32 temp_yy = xx * sinroll + yy*cosroll; 
    return temp_xx + render_width/2;
}

f32 rotate_y(f32 angle, s32 input_x, s32 input_y) {
    
    f32 sinroll = sinf(angle);
    f32 cosroll = cosf(angle);            
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

static u8* premult_norm_buffer = NULL;
static u8* base_premult_norm_buffer = NULL;

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

// now it might be a bit faster vertically contiguous, for the scalar case
// but cpus are pretty good at handling constant stride stuff
// and the stride here would be 8

// however, this means we can write groups of 8 pixels all at once
    return (high_x<<(num_render_size_bits+3))|(y<<3)|low_x;
}


void set_occlusion_bit(u32 x, u32 y, u8 bit) {
    // bit logic
    u32 bit_pos_x = x & 0b111;
    u32 byte_x = x >> 3;
    u32 byte_offset = (byte_x<<num_render_size_bits)|y;
    occlusion_buffer[byte_offset] |= (bit << bit_pos_x);
}

void set_occlusion_byte(u32 x, u32 y, u8 byte) {
    // bit logic
    u32 byte_x = x >> 3;
    u32 byte_offset = (byte_x<<num_render_size_bits)|y;
    occlusion_buffer[byte_offset] = byte;
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

u64 get_occlusion_block(u32 x, u32 y) {
    u32 byte_x = x >> 3;
    u32 byte_offset = (byte_x<<num_render_size_bits)|y;
    u8* ptr = &occlusion_buffer[byte_offset];
    assert((((uintptr_t)ptr)&0b111) == 0);
    u64* block_ptr = (u64*)ptr;
    return *ptr;
}

#define VECTOR


void raycast_vector(s32 min_x, s32 min_y, s32 max_x, s32 max_y);
void raycast_scalar(s32 min_x, s32 min_y, s32 max_x, s32 max_y);
void clear_screen(s32 min_x, s32 min_y, s32 max_x, s32 max_y);
void fill_empty_entries(s32 min_x, s32 min_y, s32 max_x, s32 max_y);
void rotate_light_and_blend(s32 min_x, s32 min_y, s32 max_x, s32 max_y);


thread_pool_function(raycast_scalar_wrapper, arg_var)
{
	thread_params* tp = (thread_params*)arg_var;
    raycast_scalar(tp->min_x, tp->min_y, tp->max_x, tp->max_y);
	InterlockedIncrement64(&tp->finished);
}

thread_pool_function(raycast_vector_wrapper, arg_var)
{
	thread_params* tp = (thread_params*)arg_var;
    raycast_vector(tp->min_x, tp->min_y, tp->max_x, tp->max_y);
	InterlockedIncrement64(&tp->finished);
}

thread_pool_function(fill_empty_entries_wrapper, arg_var)
{
	thread_params* tp = (thread_params*)arg_var;
    fill_empty_entries(tp->min_x, tp->min_y, tp->max_x, tp->max_y);

	InterlockedIncrement64(&tp->finished);
}

thread_pool_function(clear_screen_wrapper, arg_var)
{
	thread_params* tp = (thread_params*)arg_var;
    clear_screen(tp->min_x, tp->min_y, tp->max_x, tp->max_y);

	InterlockedIncrement64(&tp->finished);
}

thread_pool_function(rotate_light_and_blend_wrapper, arg_var) 
{
	thread_params* tp = (thread_params*)arg_var;
    rotate_light_and_blend(tp->min_x, tp->min_y, tp->max_x, tp->max_y);

	InterlockedIncrement64(&tp->finished);

}


f32 scale_height;


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

    f32 scale = 1.0 / tanf(deg_to_rad(desired_fov_degrees)/2.0);

    //scale_height = ((((16/9)*(.5))/(4/3))*render_size);
    scale_height = (render_height) * scale * max(1, aspect_ratio);

    base_inter_buffer = malloc((sizeof(u32)*render_size*render_size)+32);
    base_world_pos_buffer = malloc((sizeof(u32)*render_size*render_size)+32);
    base_norm_buffer = malloc((sizeof(f32)*2*render_size*render_size)+32);
    base_occlusion_buffer = malloc((sizeof(u8)*render_size*render_size/8)+32);
    base_albedo_buffer = malloc((sizeof(u32)*render_size*render_size)+32);
    base_premult_norm_buffer = malloc((sizeof(char)*render_size*render_size)+32);


    
    inter_buffer = base_inter_buffer;
    world_pos_buffer = base_world_pos_buffer;
    norm_buffer = base_norm_buffer;
    premult_norm_buffer = base_premult_norm_buffer;
    occlusion_buffer = base_occlusion_buffer;
    albedo_buffer = base_albedo_buffer;

    // alignment shenanigans for avx2
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
    while(((intptr_t)premult_norm_buffer)&0b11111) {
        premult_norm_buffer++;
    }

    setup_render_size = 1;


}



__m256 srgb255_to_linear_256(__m256 cols) {

    __m256 one_over_255_vec = _mm256_set1_ps(1.0f / 255.0f);
    __m256 div_255 = _mm256_mul_ps(cols, one_over_255_vec);
    return _mm256_mul_ps(div_255, div_255);
}

__m256i linear_to_srgb_255_256(__m256 cols) {

    return _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_set1_ps(255.0f), _mm256_sqrt_ps(cols)));
}

void handle_mouse_input(f32 dt);

typedef struct {
    f32 x, y, z;
} vect3d;

typedef struct {
    f32 x, y;
} vect2d;

typedef struct {
    f32 x, y, z, w;
} vect4d;

f32 vect2d_len(vect2d v) {
    return sqrtf(v.x*v.x + v.y*v.y);
}

vect3d vect3d_mult_scalar(vect3d v, f32 scalar) {
    vect3d res;
    res.x = v.x * scalar;
    res.y = v.y * scalar;
    res.z = v.z * scalar;
    return res;
}

vect3d vect3d_add(vect3d a, vect3d b) {
    vect3d res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    res.z = a.z + b.z;
    return res;
}

vect4d vect4d_add(vect4d a, vect4d b) {
    vect4d res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    res.z = a.z + b.z;
    res.w = a.w;
    return res;
}

vect4d vect3d_to_4d(vect3d a) {
    vect4d res;
    res.x = a.x;
    res.y = a.y;
    res.z = a.z;
    res.w = 1;
    return res;
}

typedef struct {
    f32 els[4][4];
} mat44;

void mat44_mult(mat44* src1, mat44* src2, mat44* dest) {
    dest->els[0][0] = src1->els[0][0] * src2->els[0][0] + src1->els[0][1] * src2->els[1][0] + src1->els[0][2] * src2->els[2][0] + src1->els[0][3] * src2->els[3][0]; 
    dest->els[0][1] = src1->els[0][0] * src2->els[0][1] + src1->els[0][1] * src2->els[1][1] + src1->els[0][2] * src2->els[2][1] + src1->els[0][3] * src2->els[3][1]; 
    dest->els[0][2] = src1->els[0][0] * src2->els[0][2] + src1->els[0][1] * src2->els[1][2] + src1->els[0][2] * src2->els[2][2] + src1->els[0][3] * src2->els[3][2]; 
    dest->els[0][3] = src1->els[0][0] * src2->els[0][3] + src1->els[0][1] * src2->els[1][3] + src1->els[0][2] * src2->els[2][3] + src1->els[0][3] * src2->els[3][3]; 
    dest->els[1][0] = src1->els[1][0] * src2->els[0][0] + src1->els[1][1] * src2->els[1][0] + src1->els[1][2] * src2->els[2][0] + src1->els[1][3] * src2->els[3][0]; 
    dest->els[1][1] = src1->els[1][0] * src2->els[0][1] + src1->els[1][1] * src2->els[1][1] + src1->els[1][2] * src2->els[2][1] + src1->els[1][3] * src2->els[3][1]; 
    dest->els[1][2] = src1->els[1][0] * src2->els[0][2] + src1->els[1][1] * src2->els[1][2] + src1->els[1][2] * src2->els[2][2] + src1->els[1][3] * src2->els[3][2]; 
    dest->els[1][3] = src1->els[1][0] * src2->els[0][3] + src1->els[1][1] * src2->els[1][3] + src1->els[1][2] * src2->els[2][3] + src1->els[1][3] * src2->els[3][3]; 
    dest->els[2][0] = src1->els[2][0] * src2->els[0][0] + src1->els[2][1] * src2->els[1][0] + src1->els[2][2] * src2->els[2][0] + src1->els[2][3] * src2->els[3][0]; 
    dest->els[2][1] = src1->els[2][0] * src2->els[0][1] + src1->els[2][1] * src2->els[1][1] + src1->els[2][2] * src2->els[2][1] + src1->els[2][3] * src2->els[3][1]; 
    dest->els[2][2] = src1->els[2][0] * src2->els[0][2] + src1->els[2][1] * src2->els[1][2] + src1->els[2][2] * src2->els[2][2] + src1->els[2][3] * src2->els[3][2]; 
    dest->els[2][3] = src1->els[2][0] * src2->els[0][3] + src1->els[2][1] * src2->els[1][3] + src1->els[2][2] * src2->els[2][3] + src1->els[2][3] * src2->els[3][3]; 
    dest->els[3][0] = src1->els[3][0] * src2->els[0][0] + src1->els[3][1] * src2->els[1][0] + src1->els[3][2] * src2->els[2][0] + src1->els[3][3] * src2->els[3][0]; 
    dest->els[3][1] = src1->els[3][0] * src2->els[0][1] + src1->els[3][1] * src2->els[1][1] + src1->els[3][2] * src2->els[2][1] + src1->els[3][3] * src2->els[3][1]; 
    dest->els[3][2] = src1->els[3][0] * src2->els[0][2] + src1->els[3][1] * src2->els[1][2] + src1->els[3][2] * src2->els[2][2] + src1->els[3][3] * src2->els[3][2]; 
    dest->els[3][3] = src1->els[3][0] * src2->els[0][3] + src1->els[3][1] * src2->els[1][3] + src1->els[3][2] * src2->els[2][3] + src1->els[3][3] * src2->els[3][3]; 
}

vect3d mat44_mult_vec4(vect4d* src1, mat44* src2) {
    vect3d res;
    res.x = src2->els[0][0] * src1->x + src2->els[1][0] * src1->y + src2->els[2][0] * src1->z + src2->els[3][0] * src1->w;
    res.y = src2->els[0][1] * src1->x + src2->els[1][1] * src1->y + src2->els[2][1] * src1->z + src2->els[3][1] * src1->w;
    res.z = src2->els[0][2] * src1->x + src2->els[1][2] * src1->y + src2->els[2][2] * src1->z + src2->els[3][2] * src1->w;
    return res;
}

void get_camera_mat44(f32 pitch_ang, f32 roll_ang, f32 yaw_ang, mat44* res) {
    f32 cp = cos(pitch_ang);
    f32 sp = sin(pitch_ang);
    f32 cy = cos(yaw_ang);
    f32 sy = sin(yaw_ang);
    f32 cr = cos(roll_ang);
    f32 sr = sin(roll_ang);
    mat44 pitch = (mat44){.els = { 
        {1.0f, 0.0f, 0.0f, 0.0f},
        //{1.0f, cp, sp, 0.0f},
        //{1.0f, -sp, cp, 0.0f},
        {0.0f, cp, sp, 0.0f},
        {0.0f, -sp, cp, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
    }};
    mat44 yaw = (mat44){.els = {
        {cy, 0.0f, -sy,  0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {sy, 0.0f, cy, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
    }};
    mat44 roll = (mat44){.els = {
        {cr, sr, 0.0f, 0.0f},
        {-sr, cr, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f}
    }};

    mat44 inter;
    mat44_mult(&roll, &yaw, &inter);
    mat44_mult(&inter, &pitch, res);
}
    
void pix(int x, int y, uint32_t color) {
    u32 fb_idx = fb_swizzle(x, y);
    albedo_buffer[fb_idx] = color;  
    world_pos_buffer[fb_idx] = 0b01111111;
}

void draw_vert2d(vect2d v, u32 col) {
    if(v.x < 0 || v.y < 0) { return; }
    for(int y = v.y-2; y < v.y+3; y++) {
        for(int x = v.x-2; x < v.x+3; x++) {
            if(x > 2 && x < (s32)render_size-2 && y > 2 && y < (s32)render_size-2) {
                pix(x, y, col);
            }
       }
    }
}

int in_bounds(int x, int y) {
    return x >= 0 && x < (s32)render_size && y >= 0 && y < (s32)render_size;
}

void draw_line(int x1, int y1, int x2, int y2, uint32_t color) {
    int dx = x2 - x1;
    int dy = y2 - y1;

    // If both of the differences are 0 there will be a division by 0 below.
    if (dx == 0 && dy == 0) {
        if (in_bounds(x1, y1)) {
            pix(x1, y1, color);
        }
        return;
    }

    if (OLIVEC_ABS(int, dx) > OLIVEC_ABS(int, dy)) {
        if (x1 > x2) {
            OLIVEC_SWAP(int, x1, x2);
            OLIVEC_SWAP(int, y1, y2);
        }

        for (int x = x1; x <= x2; ++x) {
            int y = dy*(x - x1)/dx + y1;
            // TODO: move boundary checks out side of the loops in olivec_draw_line
            if (in_bounds(x, y)) {
                pix(x, y, color);
            }
        }
    } else {
        if (y1 > y2) {
            OLIVEC_SWAP(int, x1, x2);
            OLIVEC_SWAP(int, y1, y2);
        }

        for (int y = y1; y <= y2; ++y) {
            int x = dx*(y - y1)/dy + x1;
            // TODO: move boundary checks out side of the loops in olivec_draw_line
            if (in_bounds(x, y)) {
                pix(x, y, color);
            }
        }
    }
}

vect3d rotate_vect3d(vect3d v, float yaw, float pitch, float roll) { //X Y Z Rotation
    float cosa = cosf(yaw); float cosb = cosf(pitch); float cosc = cosf(roll);
    float sina = sinf(yaw); float sinb = sinf(pitch); float sinc = sinf(roll);

    float Axx = cosa * cosb;
    float Axy = cosa * sinb * sinc - sina * cosc;
    float Axz = cosa * sinb * cosc + sina * sinc;

    float Ayx = sina * cosb;
    float Ayy = sina * sinb * sinc + cosa * cosc;
    float Ayz = sina * sinb * cosc - cosa * sinc;

    float Azx = -sinb;
    float Azy = cosb * sinc;
    float Azz = cosb * cosc;

    float px = v.x; float py = v.y; float pz = v.z;
    vect3d r;
    r.x = Axx * px + Axy * py + Axz * pz;
    r.y = Ayx * px + Ayy * py + Ayz * pz;
    r.z = Azx * px + Azy * py + Azz * pz;
    return r;
}


vect2d project_and_adjust_vect3d(vect3d v) {
    f32 scale = 1.0 / tanf(deg_to_rad(desired_fov_degrees)/2.0);
    const f32 const1 = 0.5f * render_width;
    const f32 const2 = 0.5f * render_width * scale / min(1, aspect_ratio);
    const f32 const3 = 0.5f * render_height;
    const f32 const4 = 0.5f * render_height * scale * max(1, aspect_ratio);
    vect2d res;
    res.x = const1 + const2 * v.x / v.z;
    res.y = render_height - (const3 + const4 * v.y / v.z);
    
    s32 y_off = ((render_size-render_height)/2);
    s32 x_off = ((render_size-render_width)/2);
    res.x += x_off;
    res.y += y_off;
    return res;
}

typedef enum {
    CLIP_OUT,
    CLIP_IN
} state;


Olivec_Canvas vc_render(f32 dt) {
    handle_mouse_input(dt);
    handle_input(dt);   



    static int setup_thread_pool = 0;
    if(!setup_thread_pool) {
        pool = thread_pool_create(NUM_THREADS);

        setup_thread_pool = 1;
    }



    if(!setup_render_size) {
        setup_internal_render_buffers();
    }

    if(!map_loaded) {
        load_map(cur_map++);
        map_loaded = 1;
    }

    if(!plane_parameters_setup) {
        setup_ray_plane_parameters();
    }



    Olivec_Canvas oc = olivec_canvas(pixels, OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH);

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
        // TODO: check if clearing the screen benefits from multiple threads
    #if 0
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
            .func = clear_screen_wrapper,
            .raw_func = clear_screen,
            .parms = parms
        };
        start_pool(pool, &rp);
        wait_for_render_pool_to_finish(&rp);
    #else
        clear_screen(min_x, min_y, max_x, max_y);
    #endif
    }

#define NUM_ROTATE_LIGHT_BLEND_THREADS 8
#define RAYCAST_THREADS 8
    {
        thread_params parms[RAYCAST_THREADS];

        for(int i = 0; i < RAYCAST_THREADS; i++) {
            parms[i].finished = 0;
            parms[i].min_x = (i == 0) ? min_x : parms[i-1].max_x;
            parms[i].max_x = (i == RAYCAST_THREADS-1) ? max_x : (parms[i].min_x + draw_dx/RAYCAST_THREADS);
            parms[i].min_y = min_y;
            parms[i].max_y = max_y;
        }
        render_pool rp = {
            .num_jobs = RAYCAST_THREADS,
            .parms = parms
        };
            rp.func = vectorized_rendering ? raycast_vector_wrapper : raycast_scalar_wrapper,
            rp.raw_func = vectorized_rendering ? raycast_vector : raycast_scalar,
        //raycast_scalar(min_x, min_y, max_x, max_y);
        start_pool(pool, &rp);
        wait_for_render_pool_to_finish(&rp);
    }




    vect3d up_vp_vect = {.x = 0.0f, .y = -1.0f, .z = 0.0f};
    vect3d q0_left_vect = {.x = -.7f, .y = -1.0f, .z = .7f};
    vect3d q1_left_vect = {.x = 0.7f, .y = -1.0f, .z = .7f};
    vect3d q2_left_vect = {.x = 0.7f, .y = -1.0f, .z = -.7f};
    vect3d q3_left_vect = {.x = -.7f, .y = -1.0f, .z = -.7f};
    f32 cam_to_screen_dist = vect2d_len((vect2d){.x = dir_x, .y = dir_y});
    if(pitch_ang != 0) {
        vect3d screen_plane_up_vp = vect3d_mult_scalar(up_vp_vect, -cam_to_screen_dist/sin(pitch_ang));
        q0_left_vect = vect3d_mult_scalar(q0_left_vect, -cam_to_screen_dist/sin(pitch_ang));
        q1_left_vect = vect3d_mult_scalar(q1_left_vect, -cam_to_screen_dist/sin(pitch_ang));
        q2_left_vect = vect3d_mult_scalar(q2_left_vect, -cam_to_screen_dist/sin(pitch_ang));
        q3_left_vect = vect3d_mult_scalar(q3_left_vect, -cam_to_screen_dist/sin(pitch_ang));

        mat44 camera_mat44;
        //f32 yaw_ang = atan2f(dir_y, dir_x);
        get_camera_mat44(pitch_ang, 0, 0, &camera_mat44);
        //get_camera_mat44(pitch_ang, 0, yaw_ang, &camera_mat44);

        vect4d screen_plane_vp_4d = (vect4d){
            .x = screen_plane_up_vp.x, .y = screen_plane_up_vp.y, .z = screen_plane_up_vp.z, .w = 1
        };
        vect4d q0_left_vect_4d = (vect4d){
            .x = q0_left_vect.x, .y = q0_left_vect.y, .z = q0_left_vect.z, .w = 1
        };
        vect4d q1_left_vect_4d = (vect4d){
            .x = q1_left_vect.x, .y = q1_left_vect.y, .z = q1_left_vect.z, .w = 1
        };
        vect4d q2_left_vect_4d = (vect4d){
            .x = q2_left_vect.x, .y = q2_left_vect.y, .z = q2_left_vect.z, .w = 1
        };
        vect4d q3_left_vect_4d = (vect4d){
            .x = q3_left_vect.x, .y = q3_left_vect.y, .z = q3_left_vect.z, .w = 1
        };
        

        //vect3d res_w1;
        vect3d transformed_vp = mat44_mult_vec4(&screen_plane_vp_4d, &camera_mat44);
        vect3d transformed_q0_left = mat44_mult_vec4(&q0_left_vect_4d, &camera_mat44);
        vect3d transformed_q1_left = mat44_mult_vec4(&q1_left_vect_4d, &camera_mat44);
        vect3d transformed_q2_left = mat44_mult_vec4(&q2_left_vect_4d, &camera_mat44);
        vect3d transformed_q3_left = mat44_mult_vec4(&q3_left_vect_4d, &camera_mat44);

        vect3d cam_space_tris[4][3] = {
            {transformed_vp, transformed_q0_left, transformed_q1_left},
            {transformed_vp, transformed_q1_left, transformed_q2_left},
            {transformed_vp, transformed_q2_left, transformed_q3_left},
            {transformed_vp, transformed_q3_left, transformed_q0_left}
        };


        int num_clipped_tris = 0;
        int num_verts_per_clipped_tri[4];
        vect3d clipped_tris[4][4];


        int clip_and_draw_indexes[3][2] = {{0,1},{1,2},{2,0}};
        for(int i = 0; i < 4; i++) {
            vect3d v0 = cam_space_tris[i][0];
            vect3d v1 = cam_space_tris[i][1];
            vect3d v2 = cam_space_tris[i][2];
            int trivial_accept = (v0.z > 0 && v1.z > 0 && v2.z > 0);
            int trivial_reject = (v0.z <= 0 && v1.z <= 0 && v2.z <= 0);
            int maybe_accept = (v0.z > 0 || v1.z > 0 || v2.z > 0);
            if(trivial_reject) { continue; }
            if(trivial_accept) {
                vect2d proj_v0 = project_and_adjust_vect3d(v0);
                vect2d proj_v1 = project_and_adjust_vect3d(v1);
                vect2d proj_v2 = project_and_adjust_vect3d(v2);
                clipped_tris[num_clipped_tris][0] = v0;
                clipped_tris[num_clipped_tris][1] = v1;
                clipped_tris[num_clipped_tris][2] = v2;
                num_verts_per_clipped_tri[num_clipped_tris++] = 3;
                continue;
            }

            // if we get here, it's time to clip.


            int num_verts = 0;
            int cur_state = CLIP_IN;
            if(v0.z <= 0) {
                cur_state = CLIP_OUT;
            } else {
                clipped_tris[num_clipped_tris][num_verts++] = v0;
            }


            int prev_j = 0;

            for(int jidx = 0; jidx < 3; jidx++) {
                int prev_j = clip_and_draw_indexes[jidx][0];
                int cur_j = clip_and_draw_indexes[jidx][1];
                vect3d prev_vert = cam_space_tris[i][prev_j];
                vect3d cur_vert = cam_space_tris[i][cur_j];

                float dx_dz = (cur_vert.x - prev_vert.x)/(cur_vert.z - prev_vert.z);
                float dy_dz = (cur_vert.y - prev_vert.y)/(cur_vert.z - prev_vert.z);
                if(cur_state == CLIP_IN) {

                    if(cur_vert.z <= 0) {
                        float increment = prev_vert.z - 0.1f;

                        float intersection_x = prev_vert.x + (increment * dx_dz);
                        float intersection_y = prev_vert.y + (increment * dy_dz);
                        vect3d intersection_vert = {
                            .x = intersection_x,
                            .y = intersection_y,
                            .z = 0.1f,
                        };
                        // move from in to out
                        // add intersection point
                        clipped_tris[num_clipped_tris][num_verts++] = intersection_vert;
                        
                        cur_state = CLIP_OUT;
                    } else {
                        clipped_tris[num_clipped_tris][num_verts++] = cur_vert;
                    }
                } else {
                    if(cur_vert.z <= 0) {
                        // still out, do nothing
                    } else {
                        // we were out, now we're going in
                        float increment = 0.1f - prev_vert.z;
                        float intersection_x = prev_vert.x + (increment * dx_dz);
                        float intersection_y = prev_vert.x + (increment * dy_dz);
                        vect3d intersection_vert = {
                            .x = intersection_x,
                            .y = intersection_y,
                            .z = 0.1f,
                        };
                        clipped_tris[num_clipped_tris][num_verts++] = intersection_vert;
                        cur_state = CLIP_IN;
                    }
                }
            }
            num_verts_per_clipped_tri[num_clipped_tris++] = num_verts;
        }

        int num_verts_per_screen_space_tri[4];
        vect2d screen_space_tris[4][4];
        int num_screen_space_tris = 0;
        for(int i = 0; i < num_clipped_tris; i++) {
            for(int j = 0; j < num_verts_per_clipped_tri[i]; j++) {
                screen_space_tris[i][j] = project_and_adjust_vect3d(clipped_tris[i][j]);
            }
            num_verts_per_screen_space_tri[i] = num_verts_per_clipped_tri[i];
            num_screen_space_tris++;
        }


        const u32 PINK = 0xFFB469FF;
        const u32 RED = 0xFF0000FF;
        const u32 GREEN = 0xFF00FF00;
        const u32 BLUE = 0xFFFF0000;
        const u32 YELLOW = 0xFFFFFF00;
        u32 poly_colors[4] = {
            RED, GREEN, BLUE, YELLOW
        };

        for(int i = 0; i < num_screen_space_tris; i++) {
            int prev_j = 0;
            for(int j = num_verts_per_screen_space_tri[i]-1; j >= 0; j--) {
                vect2d v0 = screen_space_tris[i][prev_j];
                vect2d v1 = screen_space_tris[i][j];
                prev_j = j;
                u32 color = poly_colors[i];
                draw_line(v0.x, v0.y, v1.x, v1.y, color);

            }

        }

    }


    // post-render cleanup of non-drawn pixels DOES NOT WORK
    // because transparency will blend with the pixels of the previous frame :)
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
        start_pool(pool, &rp);
        wait_for_render_pool_to_finish(&rp);
    #endif
    }
    {
        thread_params parms[8];
        if(double_pixels == 2) {
            // the left edge of the screen breaks in this pass with 8 jobs, prob not divisible by 8
            // honestly, it runs so much faster that I sort of don't care about using 8 threads here
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
            start_pool(pool, &rp);
            wait_for_render_pool_to_finish(&rp);
        } else {
            for(int i = 0; i < NUM_ROTATE_LIGHT_BLEND_THREADS; i++) {
                parms[i].finished = 0;
                parms[i].min_x = (render_width*i/NUM_ROTATE_LIGHT_BLEND_THREADS);
                parms[i].max_x = parms[i].min_x + (render_width/NUM_ROTATE_LIGHT_BLEND_THREADS);
                parms[i].min_y = 0;
                parms[i].max_y = render_height;
            }
            render_pool rp = {
                .num_jobs = NUM_ROTATE_LIGHT_BLEND_THREADS,
                .func = rotate_light_and_blend_wrapper,
                .raw_func = rotate_light_and_blend,
                .parms = parms
            };
            start_pool(pool, &rp);
            wait_for_render_pool_to_finish(&rp);
        }   
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
    s32 screen_center_map_z = get_world_pos_for_color_slot(screen_center_map_x, screen_center_map_y, color_slot_in_column);

    column_header* header = &columns_header_data[vox_col_idx];
    int num_runs = header->num_runs;
    span* runs = &columns_runs_data[vox_col_idx].runs_info[0];
    u32* color_ptr = &columns_colors_data[vox_col_idx].colors[0];

    int modify_run_idx = -1;
    int has_a_run_directly_below = 0; int has_a_run_directly_above = 0;

    int above_color_slot_in_column = color_slot_in_column-1;
    int below_color_slot_in_column = color_slot_in_column+1;
    int run_has_non_transparent_colors_below = 0;
    int run_has_non_transparent_colors_above = 0;
    int prev_run_end_idx = 0;
    int next_run_start_idx = 0;

#if 0
    for(int i = 0; i < header->num_runs; i++) {
        if(screen_center_map_z >= runs[i].top && screen_center_map_z < runs[i].bot) {
            modify_run_idx = i;      
            if(screen_center_map_z > runs[i].top) {
                run_has_non_transparent_colors_above = ((color_ptr[color_slot_in_column-1]>>24)&0b11) == 0b11;
            }
            if(screen_center_map_z < ((s16)runs[i].bot)-1) {
                run_has_non_transparent_colors_below = ((color_ptr[color_slot_in_column+1]>>24)&0b11) == 0b11;
            }
            break;
        }
    }
#endif 
    //if(modify_run_idx == -1) {
    //    goto pixel_is_undrawn; // don't modify
    //}


    static int last_frame_middle_mouse_down = 0;
    if(middle_mouse_down) {
        if(last_frame_middle_mouse_down == 0) {
            last_frame_middle_mouse_down = 1;
            
            u8 new_r = 0xC1>>2; // 0x30
            u8 new_g = 0xE3>>2; // 0x38
            u8 new_b = 0xC7>>2; // 0x31
            *(color_ptr+color_slot_in_column) = (0b01<<24)|(0b11111100<<24)|(new_b<<16)|(new_g<<8)|(new_r);
            //runs[modify_run_idx].is_transparent = 1;
            
            //set_voxel_to_surface(screen_center_map_x, screen_center_map_y-1, screen_center_map_z, DEFAULT_VOXEL_COLOR);
            //set_voxel_to_surface(screen_center_map_x-1, screen_center_map_y, screen_center_map_z, DEFAULT_VOXEL_COLOR);
            //set_voxel_to_surface(screen_center_map_x+1, screen_center_map_y, screen_center_map_z, DEFAULT_VOXEL_COLOR);
            //set_voxel_to_surface(screen_center_map_x, screen_center_map_y+1, screen_center_map_z, DEFAULT_VOXEL_COLOR);

        }
    } else {
        last_frame_middle_mouse_down = 0;
    }
    if (right_mouse_down) {
        
        *(color_ptr+color_slot_in_column) = 0xFFB469FF; 
    }
    pixel_is_undrawn:;

    
    f32 ms = dt*1000;
    f32 fps = 1000/ms;


    printf("fps; %f\n", fps);
    
    total_time += ms;
    frame++;

    sun_ang += dt*0.13;
    if(lighting == STATIC_LIGHTING) {
        sun_ang = 5.5;
    }
    f32 sun_degrees = sun_ang * 57.2958;


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
    } else if (lighting == STATIC_LIGHTING) {
        background_color = DAY_BACKGROUND_COLOR;
        amb_light_factor = 0.8;
    } else {
        // TODO: re-enable the day/night cycle with a separate toggle from the lighting button
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


    {

        #define DEBUG_PRINT_TEXT(fmt_str, ...) do { \
            sprintf(buf, (fmt_str), __VA_ARGS__);   \
            olivec_text_no_blend(oc, buf, 5, y, olivec_default_font, 2, 0xFFFFFFFF);\
            y += 13;                                \
        } while(0);

        char buf[128];
        int y = 3;
        
        DEBUG_PRINT_TEXT("lighting:       %s", lighting_mode_strs[lighting]);
        DEBUG_PRINT_TEXT("amb. occlusion: %s", ambient_occlusion ? "enabled" : "disabled");
        DEBUG_PRINT_TEXT("fog:            %s", fogmode ? "enabled" : "disabled");
        DEBUG_PRINT_TEXT("transparency:   %s", transparency ? "enabled" : "disabled");
        DEBUG_PRINT_TEXT("render mode:    %s", vectorized_rendering ? "vector" : "scalar");
        DEBUG_PRINT_TEXT("view mode:      %s", view_mode_strs[view]);
        DEBUG_PRINT_TEXT("render resolution: %ix%i", max_x-min_x, (max_y+1)-min_y);
        DEBUG_PRINT_TEXT("frametime: %.2fms", (dt*1000));
        return oc;
    }
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
    __m256i low_two_bits_mask = _mm256_set1_epi32(0b11);
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
    __m256i one_vec = _mm256_set1_epi32(1);
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
    
    __m256 dir_x_vec = _mm256_set1_ps(dir_x);
    __m256 dir_y_vec = _mm256_set1_ps(dir_y);
    __m256 plane_x_vec = _mm256_set1_ps(plane_x);
    __m256 plane_y_vec = _mm256_set1_ps(plane_y);
    
    __m256 pos_x_vec = _mm256_set1_ps(pos_x);
    __m256 pos_y_vec = _mm256_set1_ps(pos_y);
    __m256 inverse_height_vec = _mm256_set1_ps(255-height);
    __m256 ten_twenty_four_vec = _mm256_set1_ps(1024.0);

    f32 sun_vec_x = 0;
    f32 sun_vec_y = sinf(sun_ang);
    f32 sun_vec_z = cosf(sun_ang);
    __m256 sun_vec_x_vec = _mm256_set1_ps(sun_vec_x);
    __m256 sun_vec_y_vec = _mm256_set1_ps(sun_vec_y);
    __m256 sun_vec_z_vec = _mm256_set1_ps(sun_vec_z);

    __m256 amb_light_factor_vec = _mm256_set1_ps(amb_light_factor);
    __m256i background_color_vec = _mm256_set1_epi32(background_color);
    __m256 linear_background_color_r_ps_vec = srgb255_to_linear_256(_mm256_set1_ps(background_color&0xFF));
    __m256 linear_background_color_g_ps_vec = srgb255_to_linear_256(_mm256_set1_ps((background_color>>8)&0xFF));
    __m256 linear_background_color_b_ps_vec = srgb255_to_linear_256(_mm256_set1_ps((background_color>>16)&0xFF));

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
            
            
            // !! this code was using world-coordinates to look up albedos
            // it works great if you don't need transparency.  requires 32 less bits in the g-buffer to not store albedos
            //__m256i voxelmap_idxs = get_voxelmap_idx_256(xs, ys);

            // each column is 128 rgbas
            // which is 512 
            //__m256i shifted_voxelmap_idxs = _mm256_slli_epi32(voxelmap_idxs, 6); // for 7 bits of color_idx // 8 for 256 color tall columns
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

            __m256i albedo_coverages = _mm256_and_si256(_mm256_srli_epi32(albedos, 24), low_two_bits_mask);
            __m256i opaque_pixels = _mm256_cmpeq_epi32(low_two_bits_mask, albedo_coverages);
            __m256i fully_transparent_pixels = _mm256_cmpeq_epi32(_mm256_set1_epi32(0), albedo_coverages);
            u32 opaque_pixels_mask = _mm256_movemask_epi8(opaque_pixels);
            u32 fully_transparent_pixels_mask = _mm256_movemask_epi8(fully_transparent_pixels);

            
            // 0 to 1
            // for opaque pixels this will be zero 
            // for partially transparent pixels this will be some non-zero number :) 
            __m256 one_minus_old_coverages = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(low_two_bits_mask, albedo_coverages)), _mm256_set1_ps(3.0));
            
            float_rs = _mm256_min_ps(one_ps_vec, 
                                    _mm256_add_ps(float_rs, 
                                                    _mm256_mul_ps(linear_background_color_r_ps_vec, one_minus_old_coverages)));
            float_gs = _mm256_min_ps(one_ps_vec, 
                                    _mm256_add_ps(float_gs, 
                                                _mm256_mul_ps(linear_background_color_g_ps_vec, one_minus_old_coverages)));
            float_bs = _mm256_min_ps(one_ps_vec, 
                                    _mm256_add_ps(float_bs, 
                                                _mm256_mul_ps(linear_background_color_b_ps_vec, one_minus_old_coverages)));
            

            // blend non-opaque pixels 
            //

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
                __m256 depth_zero_to_one = _mm256_div_ps(float_depths, _mm256_set1_ps(max_z));
                __m256 depth_brightness = lerp256(one_ps_vec, depth_zero_to_one, zero_ps_vec);
                depth_brightness = _mm256_blendv_ps(depth_brightness, zero_ps_vec, (__m256)depths_eq_max);
                float_rs = depth_brightness;
                float_gs = depth_brightness;
                float_bs = depth_brightness;
            
            } else if (view == VIEW_STANDARD) {

            
                if(lighting) { // } == FANCY_LIGHTING) {
                    __m256 dot_lights = //_mm256_add_ps(
                                            //_mm256_mul_ps(normal_xs, sun_vec_x_vec),
                                            _mm256_add_ps(_mm256_mul_ps(normal_ys, sun_vec_y_vec),
                                                            _mm256_mul_ps(normal_zs, sun_vec_z_vec));//);

                    __m256 color_norm_scales = _mm256_min_ps(one_ps_vec, 
                                                    _mm256_add_ps(_mm256_max_ps(zero_ps_vec, 
                                                                                _mm256_min_ps(dot_lights, _mm256_set1_ps(.25))), 
                                                                  amb_light_factor_vec));
                    
                    __m256 depth_color_factor = _mm256_andnot_ps((__m256)depths_eq_max, color_norm_scales);
                    

                    //for(int i = 0; i < 8; i++) {
                    //    if(_mm256_)
                    //}

                    float_rs = _mm256_mul_ps(depth_color_factor, float_rs); 
                    float_bs = _mm256_mul_ps(depth_color_factor, float_bs); 
                    float_gs = _mm256_mul_ps(depth_color_factor, float_gs); 
                } else if (lighting) {
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

                if(ambient_occlusion) {
                    __m256i albedo_ao = _mm256_and_si256(_mm256_srli_epi32(albedos, 26), low_eight_bits_mask);
                    __m256 ao_float = _mm256_div_ps(_mm256_cvtepi32_ps(albedo_ao), _mm256_set1_ps(63.0)); //255.0));
                    float_rs = _mm256_mul_ps(ao_float, float_rs);
                    float_gs = _mm256_mul_ps(ao_float, float_gs);
                    float_bs = _mm256_mul_ps(ao_float, float_bs);
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
        // scalar version of rotation
        // no support for lighting or fog
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


u32 mix_colors(u32 old_color, u32 new_color) {
    if(!transparency) { return (0b11<<24)|new_color; }
    u32 old_col_without_alpha = (old_color & (~((u32)0xFF<<24)));
    u32 new_col_without_alpha = (new_color & (~((u32)0xFF<<24)));
    u8 new_is_transparent = ((new_color>>24)&0b11) != 0b11;
    u8 both_transparent = ((((old_color>>24)&0b11) != 0b11) && new_is_transparent);

    // TODO: if any pixels under a transparent pixel were undrawn, we should blend with the background color
    // currently we blend with BLACK, which doesn't look good

    //old_color = (old_col_without_alpha == 0 && new_is_transparent) ? (background_color & (~(0b11<<24))) : old_color;
    

    u8 new_r = (new_color>>0)&0xFF;
    u8 new_g = (new_color>>8)&0xFF;
    u8 new_b = (new_color>>16)&0xFF;
    u8 new_color_coverage = (new_color>>24)&0b11;
    u8 new_color_ao = (new_color>>24)&0b11111100;

    u8 old_r = (old_color>>0)&0xFF;
    u8 old_g = (old_color>>8)&0xFF;
    u8 old_b = (old_color>>16)&0xFF;
    u8 old_color_coverage = (old_color>>24)&0b11;


    u8 mixed_color_coverage = min((new_color_coverage + old_color_coverage), 0b11);
    u8 mixed_color_coverage_and_ao = (new_color_ao | mixed_color_coverage);

    u8 one_minus_old_color_coverage = (0b11-old_color_coverage); // 0 to 3?
    f32 one_minus_old_color_coverage_f = (one_minus_old_color_coverage)/3.0;
#if 1
    __m128 new_vec_ps = _mm_set_ps(0xFF, new_b, new_g, new_r);
    __m128 scaled_new_vec_ps = _mm_mul_ps(new_vec_ps, _mm_set1_ps(one_minus_old_color_coverage/3.0));
    __m128i new_vec_int = _mm_cvtps_epi32(scaled_new_vec_ps);

    __m128i sixteen_bit_packed_new_color = _mm_packus_epi32(new_vec_int, new_vec_int);
    __m128i new_col_vec = _mm_packus_epi16(sixteen_bit_packed_new_color, sixteen_bit_packed_new_color);
    __m128i old_col_vec = _mm_set1_epi32(old_color);

    __m128i summed = _mm_adds_epu8(new_col_vec, old_col_vec);

    u32 abgr = _mm_extract_epi32(summed, 0) & (0x00FFFFFF);

    return (both_transparent && (old_col_without_alpha == new_col_without_alpha)) ? old_color : (mixed_color_coverage_and_ao<<24)|abgr;
#else
    u8 r = old_r + (one_minus_old_color_coverage_f*new_r);
    u8 g = old_g + (one_minus_old_color_coverage_f*new_g);
    u8 b = old_b + (one_minus_old_color_coverage_f*new_b);

    //u32 scalar_res = (both_transparent && (old_col_without_alpha == new_col_without_alpha)) ? old_color : ((u32)((mixed_color_coverage_and_ao<<24)|(b<<16)|(g<<8)|(r)));
    return ((u32)((mixed_color_coverage_and_ao<<24)|(b<<16)|(g<<8)|(r)));
#endif

}

typedef enum {
    X_SIDE = 0,
    Y_SIDE = 1,
} side;



void raycast_scalar(s32 min_x, s32 min_y, s32 max_x, s32 max_y) {
  
    
    f32 fog_r = (background_color&0xFF);
    f32 fog_g = ((background_color>>8)&0xFF);
    f32 fog_b = ((background_color>>16)&0xFF);

    u8 looking_up = (pitch_ang > 0.0);

    float2 encoded_top_face_norm = encode_norm(0, -1, 0);
    float2 encoded_bot_face_norm = encode_norm(0, 1, 0);
    float2 encoded_x_side_norm = encode_norm( (dir_x > 0 ? 1 : -1), 0, 0);
    float2 encoded_y_side_norm = encode_norm(0, 0, (dir_y > 0 ? 1 : -1));
    
    
    f32 pitch = pitch_ang_to_pitch(pitch_ang);
    u8 prev_side, next_side;

    profile_block raycast_scalar_block;
    TimeBlock(raycast_scalar_block, "raycast scalar");

    for(int x = min_x; x < max_x; x++) {

        int next_drawable_min_y = min_y;
        int prev_drawn_max_y = max_y+1;

        f32 camera_space_x = ((2 * x) / ((f32)render_size)) - 1; //x-coordinate in camera space


        f32 ray_dir_x = dir_x + (plane_x * camera_space_x);
        f32 ray_dir_y = dir_y + (plane_y * camera_space_x);
        //which box of the map we're in
        s32 map_x = ((s32)pos_x);
        s32 map_y = ((s32)pos_y);


        //length of ray from one x or y-side to next x or y-side
        f32 delta_dist_x = (ray_dir_x == 0) ? 1e30 : fabs(1 / ray_dir_x);
        f32 delta_dist_y = (ray_dir_y == 0) ? 1e30 : fabs(1 / ray_dir_y);

        f32 wrap_x_minus_map_x = (pos_x - map_x);
        f32 map_x_plus_one_minus_wrap_x = (map_x + (1.0 - pos_x));
        f32 wrap_y_minus_map_y = (pos_y - map_y);
        f32 map_y_plus_one_minus_wrap_y = (map_y + (1.0 - pos_y));


        //what direction to step in x or y-direction (either +1 or -1)
        int step_x = (ray_dir_x < 0) ? -1 : 1;
        int step_y = (ray_dir_y < 0) ? -1 : 1;
        f32 perp_wall_dist = 0;
        f32 next_perp_wall_dist = 0;

        //length of ray from current position to next x or y-side
        f32 side_dist_x = (ray_dir_x < 0 ? wrap_x_minus_map_x : map_x_plus_one_minus_wrap_x) * delta_dist_x;
        f32 side_dist_y = (ray_dir_y < 0 ? wrap_y_minus_map_y : map_y_plus_one_minus_wrap_y) * delta_dist_y;


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

        profile_block z_step_block;

        s32 max_x_steps = (ray_dir_x > 0 ? (1023-(s32)pos_x) : (((s32)pos_x))) + 1;
        s32 max_y_steps = (ray_dir_y > 0 ? (1023-(s32)pos_y) : (((s32)pos_y))) + 1;


        while(perp_wall_dist <= max_z && prev_drawn_max_y > next_drawable_min_y && max_x_steps > 0 && max_y_steps > 0) {


            u32 prepared_combined_world_map_pos = ((map_x)<<19)|((map_y)<<8); 
            //u32 prepared_combined_world_map_pos = ((int)(avg_dist*32)<<16);

            
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



            side_dist_x += side_dist_dx;
            side_dist_y += side_dist_dy;

            s32 next_map_x = map_x + map_x_dx;
            s32 next_map_y = map_y + map_y_dy;


            if(map_x < 0 || map_x > 1023 || map_y < 0 || map_y > 1023) {
                goto next_z_step;
            }

            
            next_invz = scale_height / next_perp_wall_dist;
            u32 voxelmap_idx = get_voxelmap_idx(map_x, map_y);


            int mip = invz >= 1.0 ? 0 : ceilf(1.0/invz);
            mip = min(1, mip);
            u32 mip_map_x = (map_x >> mip);
            u32 mip_map_y = (map_y >> mip);
            u32 mip_voxelmap_idx = get_voxelmap_idx(mip_map_x, mip_map_y);

            // having arbitrary 3d voxels makes normals more difficult
            // so just use face normals rather than precalculated normals
            
            //f32 map_norm_pt1 = normal_pt1_data[map_idx];
            //f32 map_norm_pt2 = normal_pt2_data[map_idx];

            //f32 normal_pt1 = (lighting ? ((prev_side == X_SIDE) ? encoded_x_side_norm.x : encoded_y_side_norm.x) : map_norm_pt1);
            //f32 normal_pt2 = (lighting ? ((prev_side == X_SIDE) ? encoded_x_side_norm.y : encoded_y_side_norm.y) : map_norm_pt2);
            //f32 top_face_norm_pt1 = (lighting ? encoded_top_face_norm.x : map_norm_pt1);
            //f32 top_face_norm_pt2 = (lighting ? encoded_top_face_norm.y : map_norm_pt2);
            //f32 bot_face_norm_pt1 = (lighting ? encoded_bot_face_norm.x : map_norm_pt1);
            //f32 bot_face_norm_pt2 = (lighting ? encoded_bot_face_norm.y : map_norm_pt2);

            
            f32 normal_pt1 = ((prev_side == X_SIDE) ? encoded_x_side_norm.x : encoded_y_side_norm.x);
            f32 normal_pt2 = ((prev_side == X_SIDE) ? encoded_x_side_norm.y : encoded_y_side_norm.y);
            f32 top_face_norm_pt1 = encoded_top_face_norm.x;
            f32 top_face_norm_pt2 = encoded_top_face_norm.y;
            f32 bot_face_norm_pt1 = encoded_bot_face_norm.x;
            f32 bot_face_norm_pt2 = encoded_bot_face_norm.y;
            
            //f32 normal_pt1 = (lighting ? ((prev_side == X_SIDE) ? encoded_x_side_norm.x : encoded_y_side_norm.x) : map_norm_pt1);
            //f32 normal_pt2 = (lighting ? ((prev_side == X_SIDE) ? encoded_x_side_norm.y : encoded_y_side_norm.y) : map_norm_pt2);
            //f32 top_face_norm_pt1 = (lighting ? encoded_top_face_norm.x : map_norm_pt1);
            //f32 top_face_norm_pt2 = (lighting ? encoded_top_face_norm.y : map_norm_pt2);
            //f32 bot_face_norm_pt1 = (lighting ? encoded_bot_face_norm.x : map_norm_pt1);
            //f32 bot_face_norm_pt2 = (lighting ? encoded_bot_face_norm.y : map_norm_pt2);


            column_header* header = &columns_header_data[voxelmap_idx];

            if(header->num_runs == 0) { goto next_z_step; }

            // check the top of this column against the bottom of the frustum skip drawing it

            f32 relative_top_of_col =  height - (255 - header->top_y); //[voxelmap_idx];
            f32 col_top_invz = (relative_top_of_col < 0 ? invz : next_invz);
            f32 float_top_of_col_projected_height = relative_top_of_col*col_top_invz;
            s32 int_top_of_col_projected_height = floor(float_top_of_col_projected_height) + pitch;



            span* span_info = columns_runs_data[voxelmap_idx].runs_info;
            
            if(int_top_of_col_projected_height >= prev_drawn_max_y) {
                goto next_z_step;
            }
            u8 num_runs = header->num_runs;

            f32 relative_bot_of_col = height - (255 - cur_map_max_height); //span_info[num_runs-1].bot);
            f32 col_bot_invz = (relative_bot_of_col < 0 ? next_invz : invz);
            f32 float_bot_of_col_projected_height = relative_bot_of_col * col_bot_invz;
            s32 int_bot_of_col_projected_height = floor(float_bot_of_col_projected_height) + pitch;
            if(int_bot_of_col_projected_height <= next_drawable_min_y) {
                goto next_z_step;
            }



// draw top or bottom face


        //f32 new_norm_pt1 = occlusion_bit ? old_norm_pt1 : (face_norm_1);                    
        //f32 new_norm_pt2 = occlusion_bit ? old_norm_pt2 : (face_norm_2);                    

#define DRAW_CHUNK_FACE(top, bot, face_norm_1, face_norm_2, voxel_color_idx) {              \
    u32 fb_idx = fb_swizzle(x,top);                                                         \
    u32 color = top_of_col_color_ptr[voxel_color_idx];                                      \
    f32 norm_pt1 = top_of_col_norm_pt1_ptr[voxel_color_idx];                                \
    f32 norm_pt2 = top_of_col_norm_pt2_ptr[voxel_color_idx];                                \
    for(int y = top; y < bot; y++) {                                                        \
        u8 occlusion_bit = get_occlusion_bit(x, y);                                         \
        u32 combined_world_pos = prepared_combined_world_map_pos|(voxel_color_idx);         \
        f32 old_norm_pt1 = norm_buffer[fb_idx*2];                                           \
        f32 old_norm_pt2 = norm_buffer[fb_idx*2+1];                                         \
        f32 new_norm_pt1 = occlusion_bit ? old_norm_pt1 : norm_pt1;                         \
        f32 new_norm_pt2 = occlusion_bit ? old_norm_pt2 : norm_pt2;                         \
        u32 old_color = albedo_buffer[fb_idx];                                              \
        u32 mixed_color = mix_colors(old_color, color);                                     \
        u32 old_world_pos = world_pos_buffer[fb_idx];                                       \
        u32 new_world_pos = occlusion_bit ? old_world_pos : combined_world_pos;             \
        u32 new_color = (occlusion_bit ? old_color : mixed_color);                          \
        u8 mixed_alpha_coverage = (mixed_color>>24)&0b11;                                   \
        u8 new_occlusion_bit = (occlusion_bit ? 1 : (mixed_alpha_coverage == 0b11) ? 1 : 0);\
        min_coverage = (occlusion_bit ? min_coverage : min(mixed_alpha_coverage, min_coverage));\
        set_occlusion_bit(x, y, new_occlusion_bit);                                         \
        norm_buffer[fb_idx*2] = new_norm_pt1;                                               \
        norm_buffer[fb_idx*2+1] = new_norm_pt2;                                             \
        world_pos_buffer[fb_idx] = new_world_pos;                                           \
        albedo_buffer[fb_idx] = new_color;                                                  \
        fb_idx += 8;                                                                        \
    }                                                                                       \
}

// draw side of chunk
// needs voxel color interpolation
#define DRAW_CHUNK_SIDE(top, bot, face_norm_pt1, face_norm_pt2) {                           \
    s32 clipped_top_y = clipped_top_side_height - int_top_side_projected_height;            \
    f32 texel_per_y = ((f32)num_voxels) / unclipped_screen_dy;                              \
    f32 cur_voxel_color_idx = (f32)clipped_top_y * texel_per_y;                             \
    u32 fb_idx = fb_swizzle(x,top);                                                         \
    for(int y = top; y < bot; y++) {                                                        \
        u8 occlusion_bit = get_occlusion_bit(x, y);                                         \
        u16 voxel_color_idx = cur_voxel_color_idx;                                          \
        u32 color = color_ptr[voxel_color_idx];                                             \
        f32 norm_pt1 = top_of_col_norm_pt1_ptr[voxel_color_idx];                            \
        f32 norm_pt2 = top_of_col_norm_pt2_ptr[voxel_color_idx];                            \
        u32 old_color = albedo_buffer[fb_idx];                                              \
        u32 mixed_color = mix_colors(old_color, color);                                     \
        u32 combined_world_pos = prepared_combined_world_map_pos|((voxel_color_idx+color_ptr)-top_of_col_color_ptr);         \
        cur_voxel_color_idx += texel_per_y;                                                 \
        f32 old_norm_pt1 = norm_buffer[fb_idx*2];                                           \
        f32 old_norm_pt2 = norm_buffer[fb_idx*2+1];                                         \
        f32 new_norm_pt1 = occlusion_bit ? old_norm_pt1 : (norm_pt1);                       \
        f32 new_norm_pt2 = occlusion_bit ? old_norm_pt2 : (norm_pt2);                       \
        u32 old_world_pos = world_pos_buffer[fb_idx];                                       \
        u32 new_world_pos = occlusion_bit ? old_world_pos : combined_world_pos;             \
        u32 new_color = (occlusion_bit ? old_color : mixed_color);                          \
        u8 mixed_alpha_coverage = (mixed_color>>24)&0b11;                                   \
        u8 new_occlusion_bit = (occlusion_bit ? 1 : (mixed_alpha_coverage == 0b11) ? 1 : 0);             \
        min_coverage = (occlusion_bit ? min_coverage : min(mixed_alpha_coverage, min_coverage));\
        set_occlusion_bit(x, y, new_occlusion_bit);                                          \
        norm_buffer[fb_idx*2] = new_norm_pt1;                                               \
        norm_buffer[fb_idx*2+1] = new_norm_pt2;                                             \
        world_pos_buffer[fb_idx] = new_world_pos;                                           \
        albedo_buffer[fb_idx] = new_color;                                                  \
        fb_idx += 8;                                                                        \
    }                                                                                       \
}

#define CLAMP(a, mi, ma) max(mi, min(a, ma))



#define DRAW_CHUNK(chunk_top, chunk_bot, is_top_chunk, is_transparent_chunk, break_if_top_below_screen, break_if_bot_above_screen, side_norm_pt1, side_norm_pt2, face_norm_pt1, face_norm_pt2) {         \
    f32 relative_bot = height-chunk_bot;                                                 \
    f32 relative_top = height-chunk_top;                                                 \
    s32 int_top_face_projected_height = floor(relative_top*next_invz) + pitch;           \
    s32 int_top_side_projected_height = floor(relative_top*invz) + pitch;                \
    s32 int_bot_face_projected_height = floor(relative_bot*next_invz) + pitch;           \
    s32 int_bot_side_projected_height = floor(relative_bot*invz) + pitch;                \
    s32 top_projected_heightonscreen = (!is_top_chunk) ? int_top_side_projected_height : min(int_top_side_projected_height, int_top_face_projected_height);\
    s32 bot_projected_heightonscreen = is_top_chunk ? int_bot_side_projected_height : max(int_bot_side_projected_height,int_bot_face_projected_height);  \
    u8 min_coverage = 0b11;                                                              \
    if(break_if_bot_above_screen && bot_projected_heightonscreen < next_drawable_min_y) { \
        break;                                                                            \
    }                                                                                     \
    if(break_if_top_below_screen && top_projected_heightonscreen >= prev_drawn_max_y) {   \
        break;                                                                            \
    }                                                                                     \
    s32 clipped_top_heightonscreen = CLAMP(top_projected_heightonscreen, next_drawable_min_y, prev_drawn_max_y);    \
    s32 clipped_bot_heightonscreen = CLAMP(bot_projected_heightonscreen, next_drawable_min_y, prev_drawn_max_y);       \
    s32 unclipped_screen_dy = int_bot_side_projected_height - int_top_side_projected_height;    \
    s32 num_voxels = (chunk_top - chunk_bot);                                           \
    if(clipped_top_heightonscreen < clipped_bot_heightonscreen) {                        \
        s32 clipped_top_face_height = CLAMP(int_top_face_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_top_side_height = CLAMP(int_top_side_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_bot_face_height = CLAMP(int_bot_face_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_bot_side_height = CLAMP(int_bot_side_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        if(is_top_chunk && clipped_top_face_height < clipped_top_side_height) {          \
            u16 color_offset = (color_ptr - top_of_col_color_ptr);                       \
            DRAW_CHUNK_FACE(clipped_top_face_height, clipped_top_side_height, top_face_norm_pt1, top_face_norm_pt2, color_offset); \
        }                                                                                \
        if(clipped_top_side_height < clipped_bot_side_height) {                          \
            DRAW_CHUNK_SIDE(clipped_top_side_height, clipped_bot_side_height,  side_norm_pt1, side_norm_pt2); \
        }                                                                                \
        if(clipped_bot_side_height < clipped_bot_face_height) {       \
            u16 color_offset = ((color_ptr - top_of_col_color_ptr)+num_voxels)-1;        \
            DRAW_CHUNK_FACE(clipped_bot_side_height, clipped_bot_face_height, bot_face_norm_pt1, bot_face_norm_pt2, color_offset); \
        }                                                                                \
        s32 next_prev_drawn_max_y = (min_coverage < 0b11) ? prev_drawn_max_y : (bot_projected_heightonscreen >= prev_drawn_max_y && top_projected_heightonscreen < prev_drawn_max_y) ? top_projected_heightonscreen : prev_drawn_max_y;             \
        s32 next_next_drawable_min_y = (min_coverage < 0b11) ? next_drawable_min_y :  (top_projected_heightonscreen <= next_drawable_min_y && bot_projected_heightonscreen > next_drawable_min_y) ? bot_projected_heightonscreen : next_drawable_min_y; \
        prev_drawn_max_y = next_prev_drawn_max_y; \
        next_drawable_min_y = next_next_drawable_min_y; \
    }   \
}           


       
#define DRAW_CHUNK_BOTTOM_UP(chunk_top, chunk_bot, is_top_chunk, is_transparent_chunk, break_if_top_below_screen, break_if_bot_above_screen, side_norm_pt1, side_norm_pt2, face_norm_pt1, face_norm_pt2) {         \
    f32 relative_bot = height-chunk_bot;                                                 \
    f32 relative_top = height-chunk_top;                                                 \
    s32 int_top_face_projected_height = floor(relative_top*next_invz) + pitch;           \
    s32 int_top_side_projected_height = floor(relative_top*invz) + pitch;                \
    s32 int_bot_face_projected_height = floor(relative_bot*next_invz) + pitch;           \
    s32 int_bot_side_projected_height = floor(relative_bot*invz) + pitch;                \
    s32 top_projected_heightonscreen = (!is_top_chunk) ? int_top_side_projected_height : min(int_top_side_projected_height, int_top_face_projected_height);\
    s32 bot_projected_heightonscreen = is_top_chunk ? int_bot_side_projected_height : max(int_bot_side_projected_height,int_bot_face_projected_height);  \
    if(break_if_bot_above_screen && bot_projected_heightonscreen < next_drawable_min_y) { \
        break;                                                                            \
    }                                                                                     \
    if(break_if_top_below_screen && top_projected_heightonscreen >= prev_drawn_max_y) {   \
        break;                                                                            \
    }                                                                                     \
    s32 next_prev_drawn_max_y = is_transparent_chunk ? prev_drawn_max_y : (bot_projected_heightonscreen >= prev_drawn_max_y && top_projected_heightonscreen < prev_drawn_max_y) ? top_projected_heightonscreen : prev_drawn_max_y;             \
    s32 next_next_drawable_min_y = is_transparent_chunk ? next_drawable_min_y :  (top_projected_heightonscreen <= next_drawable_min_y && bot_projected_heightonscreen > next_drawable_min_y) ? bot_projected_heightonscreen : next_drawable_min_y; \
    s32 clipped_top_heightonscreen = CLAMP(top_projected_heightonscreen, next_drawable_min_y, prev_drawn_max_y);    \
    s32 clipped_bot_heightonscreen = CLAMP(bot_projected_heightonscreen, next_drawable_min_y, prev_drawn_max_y);       \
    s32 unclipped_screen_dy = int_bot_side_projected_height - int_top_side_projected_height;    \
    s32 num_voxels = (chunk_top - chunk_bot);                                           \
    if(clipped_top_heightonscreen < clipped_bot_heightonscreen) {                        \
        s32 clipped_top_face_height = CLAMP(int_top_face_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_top_side_height = CLAMP(int_top_side_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_bot_face_height = CLAMP(int_bot_face_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_bot_side_height = CLAMP(int_bot_side_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
                                                                                        \
        if(clipped_bot_side_height < clipped_bot_face_height) {                         \
            u16 color_offset = ((color_ptr - top_of_col_color_ptr)+num_voxels)-1;        \
            DRAW_CHUNK_FACE(clipped_bot_side_height, clipped_bot_face_height, bot_face_norm_pt1, bot_face_norm_pt2, color_offset); \
        }                                                                                \
        if(clipped_top_side_height < clipped_bot_side_height) {                          \
            DRAW_CHUNK_SIDE(clipped_top_side_height, clipped_bot_side_height,  side_norm_pt1, side_norm_pt2); \
        }                                                                                   \
        if(is_top_chunk && clipped_top_face_height < clipped_top_side_height) {          \
            u16 color_offset = (color_ptr - top_of_col_color_ptr);                       \
            DRAW_CHUNK_FACE(clipped_top_face_height, clipped_top_side_height, top_face_norm_pt1, top_face_norm_pt2, color_offset); \
        }                                                                        \
        prev_drawn_max_y = next_prev_drawn_max_y; \
        next_drawable_min_y = next_next_drawable_min_y; \
    }   \
}           
  
            
            
            //int chunk_top = -1;
            
            // we have to search to find the middle chunk.
            // then draw up to the minimum chunk
            // and draw down to the maximum chunk
            // but i am going to ignore that for now

            // find mid point
            // draw up from there
            // draw down from there

            u32* top_of_col_color_ptr = columns_colors_data[voxelmap_idx].colors;
            f32* top_of_col_norm_pt1_ptr = columns_norm_data[voxelmap_idx].norm_pt1;
            f32* top_of_col_norm_pt2_ptr = columns_norm_data[voxelmap_idx].norm_pt2;
            u32* color_ptr = top_of_col_color_ptr;

            if(looking_up) { //looking_up) {
                // draw from bottom to the top 
                // iterate over runs real quick to get the color pointer to the bottom
                for(int run = 0; run < num_runs; run++) {
                    color_ptr += ((span_info[run].top_surface_end+1)-span_info[run].top_surface_start);
                    if(span_info[run].bot_surface_end > span_info[run].bot_surface_start) {
                        color_ptr += ((span_info[run].bot_surface_end)-span_info[run].bot_surface_start);
                    }
                }

                // now go back in reverse
                for(int run = num_runs-1; run >= 0; run--) {   
                    int bot_surface_top = 255 - span_info[run].bot_surface_start;
                    int bot_surface_end = 255 - span_info[run].bot_surface_end;
                    
                    if(span_info[run].bot_surface_end > span_info[run].bot_surface_start) {
                        color_ptr -= (span_info[run].bot_surface_end-span_info[run].bot_surface_start);
                        DRAW_CHUNK(
                            bot_surface_top, bot_surface_end, 
                            0, //span_info[run].is_top, 
                            ((color_ptr[0]>>24)&0b11) != 0b11,//0, //span_info[run].is_transparent, 
                            0, 1, normal_pt1, normal_pt2, top_face_norm_pt1, top_face_norm_pt2
                        );
                    }

                    int top_surface_top = 255 - span_info[run].top_surface_start;
                    int top_surface_bot = 255 - (span_info[run].top_surface_end+1);
                    color_ptr -= ((span_info[run].top_surface_end+1)-span_info[run].top_surface_start);
                    DRAW_CHUNK(
                        top_surface_top, top_surface_bot, 
                        1, //span_info[run].is_top, 
                        ((color_ptr[0]>>24)&0b11) != 0b11, //0, //span_info[run].is_transparent, 
                        0, 1, normal_pt1, normal_pt2, top_face_norm_pt1, top_face_norm_pt2
                    );
                } 


            } else {
                // TOP DOWN LOOP

                for(int run = 0; run < num_runs; run++) {
                    int top_surface_top = 255 - span_info[run].top_surface_start;
                    int top_surface_bot = 255 - (span_info[run].top_surface_end+1);
                    int bot_surface_top = 255 - span_info[run].bot_surface_start;
                    int bot_surface_bot = 255 - span_info[run].bot_surface_end;

                    DRAW_CHUNK(
                        top_surface_top, top_surface_bot, 
                        1, //span_info[run].is_top, 
                        ((color_ptr[0]>>24)&0b11) != 0b11, //0, //span_info[run].is_transparent, 
                        1, 0, normal_pt1, normal_pt2, top_face_norm_pt1, top_face_norm_pt2
                    );
                    color_ptr += ((span_info[run].top_surface_end+1)-span_info[run].top_surface_start);
                    if(span_info[run].bot_surface_end > span_info[run].bot_surface_start) {
                        DRAW_CHUNK(
                            bot_surface_top, bot_surface_bot, 
                            0, //span_info[run].is_top, 
                            ((color_ptr[0]>>24)&0b11) != 0b11, //0, //span_info[run].is_transparent, 
                            1, 0, normal_pt1, normal_pt2, top_face_norm_pt1, top_face_norm_pt2
                        );
                        color_ptr += (span_info[run].bot_surface_end-span_info[run].bot_surface_start);
                    }
                }

            }

        next_z_step:;
        map_x = next_map_x;
        map_y = next_map_y;
        invz = next_invz;
        }
    }
    EndTimeBlock(raycast_scalar_block);

    //printf("max mip level of %i\n", max_mip);
}

void clear_screen(s32 min_x, s32 min_y, s32 max_x, s32 max_y) {        
    u32 undrawn_world_pos = 0b10000000;
    int min_x_aligned = min_x & ~0b11111;
    __m256i undrawn_vec = _mm256_set1_epi32(undrawn_world_pos);
    __m256 undrawn_norm_pt1_vec = _mm256_set1_ps(0);
    __m256i undrawn_albedo_vec = _mm256_set1_epi32(0b00000000);

    //__m256 undrawn_norm_pt2_vec = _mm256_set1_ps(0);
    for(int x = min_x_aligned; x < max_x; x += 8) {
        u32 base_fb_idx = fb_swizzle(x, min_y);
        u32* world_pos_buf_ptr = &world_pos_buffer[base_fb_idx];
        f32* norm_ptr = &norm_buffer[base_fb_idx*2];
        u32* albedo_ptr = &albedo_buffer[base_fb_idx];
        //profile_block coverage_fill_empty_entries;
        //TimeBlock(coverage_fill_empty_entries, "fill framebuffer");
        
        for(int y = min_y; y <= max_y; y++) {     
            _mm256_store_si256((__m256i*)world_pos_buf_ptr, undrawn_vec);
            world_pos_buf_ptr += 8;
            _mm256_store_si256((__m256i*)albedo_ptr, undrawn_albedo_vec);
            albedo_ptr += 8;
            _mm256_store_ps((f32*)norm_ptr, undrawn_norm_pt1_vec);
            _mm256_store_ps((f32*)(norm_ptr+8), undrawn_norm_pt1_vec);
            norm_ptr += 16;
        }
        //EndCountedTimeBlock(coverage_fill_empty_entries, max_y-min_y);
    }

}

void fill_empty_entries(s32 min_x, s32 min_y, s32 max_x, s32 max_y) {
    //u32 undrawn_albedo = background_color;
    u32 undrawn_world_pos = 0b10000000;
    __m256i undrawn_albedo_vec = _mm256_set1_epi32(0x00000000);
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
        u32* albedo_ptr = &albedo_buffer[base_fb_idx];

        //profile_block coverage_fill_empty_entries_per_row;
        //TimeBlock(coverage_fill_empty_entries_per_row, "fill empty entries per row");
        for(int y = min_y; y < max_y; y++) {
            u8 drawn_mask = get_occlusion_byte(x, y);
            
            // fill undrawn pixels with background color
            //__m256i g_buf_entries = _mm256_load_si256((__m256i*)albedo_buf_ptr);
            __m256i world_pos_entries = _mm256_load_si256((__m256i*)world_pos_ptr);
            __m256i albedo_entries = _mm256_load_si256((__m256i*)albedo_ptr);
            __m256 norm_pt1_entries = _mm256_load_ps(norm_ptr);
            __m256 norm_pt2_entries = _mm256_load_ps((norm_ptr+8));
            __m256i drawn_mask_replicated = _mm256_set1_epi32(drawn_mask);

            // 0b00110011
            // needs to be 0x00000 0x1111

            __m256i wide_mask_shifted = _mm256_sllv_epi32(_mm256_and_si256(selectors, drawn_mask_replicated), shifters);
            __m256i norm_pt1_mask = _mm256_sllv_epi32(_mm256_and_si256(norm_pt1_selectors, drawn_mask_replicated), norm_pt1_shifters);
            __m256i norm_pt2_mask = _mm256_sllv_epi32(_mm256_and_si256(norm_pt2_selectors, drawn_mask_replicated), norm_pt2_shifters);

            __m256i blended_world_pos_entries = _mm256_blendv_epi8(undrawn_vec, world_pos_entries, wide_mask_shifted);
            __m256 blended_norm_pt1_entries = _mm256_blendv_ps(undrawn_norm_pt1_vec, norm_pt1_entries, (__m256)norm_pt1_mask);
            __m256 blended_norm_pt2_entries = _mm256_blendv_ps(undrawn_norm_pt2_vec, norm_pt2_entries, (__m256)norm_pt2_mask);

            __m256i blended_albedo_entries = (__m256i)_mm256_blendv_epi8(undrawn_albedo_vec, albedo_entries, wide_mask_shifted);
            
            _mm256_store_si256((__m256i*)world_pos_ptr, blended_world_pos_entries);
            _mm256_store_ps(norm_ptr, blended_norm_pt1_entries);
            _mm256_store_ps((norm_ptr+8), blended_norm_pt2_entries);
            _mm256_store_si256((__m256i*)albedo_ptr, blended_albedo_entries);
            norm_ptr += 16;
            world_pos_ptr += 8;
            albedo_ptr += 8;
        }
        //EndCountedTimeBlock(coverage_fill_empty_entries, max_y-min_y);
    }
    EndTimeBlock(coverage_fill_empty_entries);
}

u64 total_vector_samples;
u64 reused_vector_samples;


void raycast_vector(s32 min_x, s32 min_y, s32 max_x, s32 max_y) {
    
    
    f32 fog_r = (background_color&0xFF);
    f32 fog_g = ((background_color>>8)&0xFF);
    f32 fog_b = ((background_color>>16)&0xFF);

    u8 looking_up = (pitch_ang > 0.0);

    float2 encoded_top_face_norm = encode_norm(0, -1, 0);
    float2 encoded_bot_face_norm = encode_norm(0, 1, 0);
    float2 encoded_x_side_norm = encode_norm( (dir_x > 0 ? 1 : -1), 0, 0);
    float2 encoded_y_side_norm = encode_norm(0, 0, (dir_y > 0 ? 1 : -1));
    
    
    f32 pitch = pitch_ang_to_pitch(pitch_ang);


    profile_block raycast_vector_block;
    TimeBlock(raycast_vector_block, "raycast vector");

    //__m256i prev_sides, next_sides;
    //u8 prev_sides[8], next_sides[8];
    __m256i prev_sides = _mm256_set1_epi32(0);
    __m256i next_sides = _mm256_set1_epi32(0);

    //for(int i = 0; i < 8; i++) {
    //    camera_space_xs[i] = ((2*xs[i])/(f32)render_size)-1;
    //}
    f32 camera_space_x_per_dx = 2.0/render_size;
    f32 camera_space_x_per_8_x = 8*camera_space_x_per_dx;

    __m256i two_fifty_five_vec = _mm256_set1_epi32(255);
    __m256 zero_ps_vec = _mm256_set1_ps(0);
    __m256 one_ps_vec = _mm256_set1_ps(1);
    __m256 two_ps_vec = _mm256_set1_ps(2);
    __m256i one_vec = _mm256_set1_epi32(1);
    __m256i negative_one_vec = _mm256_set1_epi32(-1);
    __m256 one_e_30_vec = _mm256_set1_ps(1e30);
    __m256i zero_vec = _mm256_set1_epi32(0);
    __m256 height_vec = _mm256_set1_ps(height);
    __m256i pitch_vec = _mm256_set1_epi32(pitch);

    __m256 dir_x_vec = _mm256_set1_ps(dir_x);
    __m256 dir_y_vec = _mm256_set1_ps(dir_y);
    __m256 plane_x_vec = _mm256_set1_ps(plane_x);
    __m256 plane_y_vec = _mm256_set1_ps(plane_y);
    __m256 scale_height_vec = _mm256_set1_ps(scale_height);

    __m256i cur_map_max_height_vec = _mm256_set1_epi32(cur_map_max_height);

    __m256 max_z_vec = _mm256_set1_ps(max_z);

    //f32 dir_x_vec[8] = {dir_x,dir_x,dir_x,dir_x,dir_x,dir_x,dir_x,dir_x};
    //f32 dir_y_vec[8] = {dir_y,dir_y}

    int min_x_aligned = min_x & ~0b11111;
    //int xs[8] = {0,1,2,3,4,5,6,7};
    __m256i xs = _mm256_setr_epi32(
        min_x_aligned+0,min_x_aligned+1,min_x_aligned+2,min_x_aligned+3,
        min_x_aligned+4,min_x_aligned+5,min_x_aligned+6,min_x_aligned+7);
    __m256i dxs_vec = _mm256_set1_epi32(8);
    //for(int x = min_x; x < max_x; x += 8) {
    for(int x = min_x_aligned; x < max_x; x += 8) {
        //int next_drawable_min_y = min_y;
        //int prev_drawn_max_y = max_y+1;

        
        // TODO: vectorize
        //int next_drawable_min_ys[8] = {min_y, min_y, min_y, min_y, min_y, min_y, min_y, min_y};
        //int prev_drawn_max_ys[8] = {max_y+1, max_y+1, max_y+1, max_y+1, max_y+1, max_y+1, max_y+1, max_y+1};
        __m256i next_drawable_min_ys = _mm256_set1_epi32(min_y);
        __m256i prev_drawn_max_ys = _mm256_set1_epi32(max_y+1);

        //f32 camera_space_x = ((2 * x) / ((f32)render_size)) - 1; //x-coordinate in camera space
        //f32 camera_space_xs[8];
        //for(int i = 0; i < 8; i++) {
        //    camera_space_xs[i] = ((2 * (x+i)) / ((f32)render_size)) - 1;
        //}
        __m256 camera_space_xs = _mm256_sub_ps(
                                    _mm256_div_ps(
                                        _mm256_mul_ps(_mm256_cvtepi32_ps(xs), two_ps_vec),
                                        _mm256_set1_ps(render_size)
                                    ),
                                    one_ps_vec);

        // TODO: vectorize
        //int camera_space_xs[8];
        //for(int i = 0; i < 8; i++) {
        //}

        //f32 ray_dir_x = dir_x + (plane_x * camera_space_x);
        //f32 ray_dir_y = dir_y + (plane_y * camera_space_x);

        
        // TODO: vectorize
        __m256 ray_dir_xs = _mm256_add_ps(dir_x_vec, _mm256_mul_ps(plane_x_vec, camera_space_xs));
        __m256 ray_dir_ys = _mm256_add_ps(dir_y_vec, _mm256_mul_ps(plane_y_vec, camera_space_xs));
        
        //f32 ray_dir_xs[8], ray_dir_ys[8];
        //for(int i = 0; i < 8; i++) {
        //    ray_dir_xs[i] = dir_x + (plane_x * camera_space_xs[i]);
        //   ray_dir_ys[i] = dir_y + (plane_y * camera_space_xs[i]);
        //}

        //which box of the map we're in
        s32 map_x = ((s32)pos_x);
        s32 map_y = ((s32)pos_y);
        __m256i map_xs = _mm256_set1_epi32(map_x);
        __m256i map_ys = _mm256_set1_epi32(map_y);
        //s32 map_xs[8] __attribute__ ((aligned (32))) = {map_x,map_x,map_x,map_x,map_x,map_x,map_x,map_x};
        //s32 map_ys[8] __attribute__ ((aligned (32))) = {map_y,map_y,map_y,map_y,map_y,map_y,map_y,map_y}; 

        //length of ray from one x or y-side to next x or y-side

        //f32 delta_dist_x = (ray_dir_x == 0) ? 1e30 : fabs(1 / ray_dir_x);
        //f32 delta_dist_y = (ray_dir_y == 0) ? 1e30 : fabs(1 / ray_dir_y);

        
        // TODO: vectorize
        __m256 ray_dir_xs_eq_zero = _mm256_cmp_ps(ray_dir_xs, zero_ps_vec, _CMP_EQ_UQ);
        __m256 ray_dir_ys_eq_zero = _mm256_cmp_ps(ray_dir_xs, zero_ps_vec, _CMP_EQ_UQ);
        __m256 div_safe_ray_dir_xs = _mm256_blendv_ps(ray_dir_xs, one_ps_vec, ray_dir_xs_eq_zero);
        __m256 div_safe_ray_dir_ys = _mm256_blendv_ps(ray_dir_ys, one_ps_vec, ray_dir_ys_eq_zero);
        __m256 abs_one_over_ray_dir_xs = abs_ps(_mm256_div_ps(one_ps_vec, div_safe_ray_dir_xs));
        __m256 abs_one_over_ray_dir_ys = abs_ps(_mm256_div_ps(one_ps_vec, div_safe_ray_dir_ys));
        __m256 delta_dist_xs = _mm256_blendv_ps(abs_one_over_ray_dir_xs, one_e_30_vec, ray_dir_xs_eq_zero);
        __m256 delta_dist_ys = _mm256_blendv_ps(abs_one_over_ray_dir_ys, one_e_30_vec, ray_dir_ys_eq_zero);
        

        //f32 delta_dist_xs[8], delta_dist_ys[8];
        //for(int i = 0; i < 8; i++) {
        //    delta_dist_xs[i] = (ray_dir_xs[i] == 0.0f ? 1e30 : (fabs(1/ray_dir_xs[i])));
        //    delta_dist_ys[i] = (ray_dir_ys[i] == 0.0f ? 1e30 : (fabs(1/ray_dir_ys[i])));
        //}

        f32 wrap_x_minus_map_x = (pos_x - map_x);
        f32 map_x_plus_one_minus_wrap_x = (map_x + (1.0 - pos_x));
        f32 wrap_y_minus_map_y = (pos_y - map_y);
        f32 map_y_plus_one_minus_wrap_y = (map_y + (1.0 - pos_y));


        //what direction to step in x or y-direction (either +1 or -1)
        //int step_x = (ray_dir_x < 0) ? -1 : 1;
        //int step_y = (ray_dir_y < 0) ? -1 : 1;

        // TODO: vectorize
        //s32 step_xs[8], step_ys[8];
        //for(int i = 0; i < 8; i++) {
        //    step_xs[i] = (ray_dir_xs[i] < 0.0f) ? -1 : 1;
        //    step_ys[i] = (ray_dir_ys[i] < 0.0f) ? -1 : 1;
        //}
        __m256 ray_dir_xs_less_than_zero = _mm256_cmp_ps(ray_dir_xs, zero_ps_vec, _CMP_LT_OQ);
        __m256 ray_dir_ys_less_than_zero = _mm256_cmp_ps(ray_dir_ys, zero_ps_vec, _CMP_LT_OQ);
        __m256i step_xs = _mm256_blendv_epi8(one_vec, negative_one_vec, (__m256i)ray_dir_xs_less_than_zero);
        __m256i step_ys = _mm256_blendv_epi8(one_vec, negative_one_vec, (__m256i)ray_dir_ys_less_than_zero);


        //f32 perp_wall_dists[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        //f32 next_perp_wall_dists[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        __m256 perp_wall_dists = zero_ps_vec;
        __m256 next_perp_wall_dists = zero_ps_vec;
        //f32 perp_wall_dist = 0;
        //f32 next_perp_wall_dist = 0;

        //length of ray from current position to next x or y-side
        //f32 side_dist_x = (ray_dir_x < 0 ? wrap_x_minus_map_x : map_x_plus_one_minus_wrap_x) * delta_dist_x;
        //f32 side_dist_y = (ray_dir_y < 0 ? wrap_y_minus_map_y : map_y_plus_one_minus_wrap_y) * delta_dist_y;

        
        // TODO: vectorize
        //f32 side_dist_xs[8], side_dist_ys[8];
        //for(int i = 0; i < 8; i++) {
        //    side_dist_xs[i] = (ray_dir_xs[i] < 0.0f ? wrap_x_minus_map_x : map_x_plus_one_minus_wrap_x) * delta_dist_xs[i];
        //    side_dist_ys[i] = (ray_dir_ys[i] < 0.0f ? wrap_y_minus_map_y : map_y_plus_one_minus_wrap_y) * delta_dist_ys[i];
        //}
        __m256 wrap_x_minus_map_xs = _mm256_set1_ps(wrap_x_minus_map_x);
        __m256 wrap_y_minus_map_ys = _mm256_set1_ps(wrap_y_minus_map_y);
        __m256 map_x_plus_one_minus_wrap_xs = _mm256_set1_ps(map_x_plus_one_minus_wrap_x);
        __m256 map_y_plus_one_minus_wrap_ys = _mm256_set1_ps(map_y_plus_one_minus_wrap_y);
        __m256 side_dist_xs = _mm256_mul_ps(_mm256_blendv_ps(map_x_plus_one_minus_wrap_xs, wrap_x_minus_map_xs, ray_dir_xs_less_than_zero), delta_dist_xs);
        __m256 side_dist_ys = _mm256_mul_ps(_mm256_blendv_ps(map_y_plus_one_minus_wrap_ys, wrap_y_minus_map_ys, ray_dir_ys_less_than_zero), delta_dist_ys);
        

        // TODO: vectorize
        //int any_perp_wall_dists_are_zero = 0;
        //for(int i = 0; i < 8; i++) {
        //    any_perp_wall_dists_are_zero |= (perp_wall_dists[i] == 0.0f);
        //}
        int any_perp_wall_dists_are_zero = _mm256_movemask_ps(_mm256_cmp_ps(perp_wall_dists, zero_ps_vec, _CMP_EQ_UQ));

        while(any_perp_wall_dists_are_zero) {
            //int map_x_dx = (side_dist_x < side_dist_y) ? step_x : 0;
            //int map_y_dy = (side_dist_x < side_dist_y) ? 0 : step_y;
            
            // TODO: vectorize
            //s32 smap_x_dxs[8],smap_y_dys[8];
            //for(int i = 0; i < 8; i++) {
            //    smap_x_dxs[i] = (side_dist_xs[i] < side_dist_ys[i]) ? step_xs[i] : 0;
            //    smap_y_dys[i] = (side_dist_xs[i] < side_dist_ys[i]) ? 0 : step_ys[i];
            //}
            __m256 side_dist_xs_less_than_side_dist_ys = _mm256_cmp_ps(side_dist_xs, side_dist_ys, _CMP_LT_OQ);
            __m256i map_x_dxs = _mm256_blendv_epi8(zero_vec, step_xs, (__m256i)side_dist_xs_less_than_side_dist_ys);
            __m256i map_y_dys = _mm256_blendv_epi8(step_ys, zero_vec, (__m256i)side_dist_xs_less_than_side_dist_ys);


            // TODO: vectorize
            //for(int i = 0; i < 8; i++) {
            //    perp_wall_dists[i] = next_perp_wall_dists[i];
            //    next_perp_wall_dists[i] = (side_dist_xs[i] < side_dist_ys[i]) ? side_dist_xs[i] : side_dist_ys[i];
            //}

            perp_wall_dists = next_perp_wall_dists;
            next_perp_wall_dists = _mm256_blendv_ps(side_dist_ys, side_dist_xs, side_dist_xs_less_than_side_dist_ys);


            //f32 side_dist_dx = (side_dist_x < side_dist_y) ? delta_dist_x : 0;
            //f32 side_dist_dy = (side_dist_x < side_dist_y) ? 0 : delta_dist_y;
            //prev_side = next_side;
            //next_side = (side_dist_x < side_dist_y) ? X_SIDE : Y_SIDE;
            
            // TODO: vectorize
            //f32 side_dist_dxs[8], side_dist_dys[8];
            //for(int i = 0; i < 8; i++) {
            //    side_dist_dxs[i] = (side_dist_xs[i] < side_dist_ys[i]) ? delta_dist_xs[i] : 0.0f;
            //    side_dist_dys[i] = (side_dist_xs[i] < side_dist_ys[i]) ? 0.0f : delta_dist_ys[i];
            //}
            __m256 side_dist_dxs = _mm256_blendv_ps(zero_ps_vec, delta_dist_xs, side_dist_xs_less_than_side_dist_ys);
            __m256 side_dist_dys = _mm256_blendv_ps(delta_dist_ys, zero_ps_vec, side_dist_xs_less_than_side_dist_ys);

            // TODO: vectorize
            //int prev_sides[8],next_sides[8];
            //for(int i = 0; i < 8; i++) {
            //    prev_sides[i] = next_sides[i];
            //    next_sides[i] = (side_dist_xs[i] < side_dist_ys[i]) ? X_SIDE : Y_SIDE;
            //}
            prev_sides = next_sides;
            next_sides = _mm256_blendv_epi8(_mm256_set1_epi32(Y_SIDE), _mm256_set1_epi32(X_SIDE), (__m256i)side_dist_xs_less_than_side_dist_ys);

            //side_dist_x += side_dist_dx;
            //side_dist_y += side_dist_dy;
            // TODO: vectorize
            
            //for(int i = 0; i < 8; i++) {
            //    side_dist_xs[i] += side_dist_dxs[i];
            //    side_dist_ys[i] += side_dist_dys[i];
            //}
            side_dist_xs = _mm256_add_ps(side_dist_xs, side_dist_dxs);
            side_dist_ys = _mm256_add_ps(side_dist_ys, side_dist_dys);


            //map_x += map_x_dx;
            //map_y += map_y_dy;
            // TODO: vectorize
            map_xs = _mm256_add_epi32(map_xs, map_x_dxs);
            map_ys = _mm256_add_epi32(map_ys, map_y_dys);

            //for(int i = 0; i < 8; i++) {
            //    map_xs[i] += map_x_dxs[i];
            //    map_ys[i] += map_y_dys[i];
            //}


            // TODO: vectorize
            //any_perp_wall_dists_are_zero = 0;
            //for(int i = 0; i < 8; i++) {
            //    any_perp_wall_dists_are_zero |= (perp_wall_dists[i] == 0);
            //}
            any_perp_wall_dists_are_zero = _mm256_movemask_ps(_mm256_cmp_ps(perp_wall_dists, zero_ps_vec, _CMP_EQ_UQ));
        }

        //f32 invz = scale_height / perp_wall_dist;
        //f32 next_invz = scale_height / next_perp_wall_dist;
        // TODO: vectorize
        //f32 invzs[8], next_invzs[8];
        //for(int i = 0; i < 8; i++) {
        //    invzs[i] = scale_height / perp_wall_dists[i];
        //    next_invzs[i] = scale_height / next_perp_wall_dists[i];
        //}
        __m256 invzs = _mm256_div_ps(scale_height_vec, perp_wall_dists);
        __m256 next_invzs = _mm256_div_ps(scale_height_vec, next_perp_wall_dists);

        profile_block z_step_block;

        //s32 max_x_steps = (ray_dir_x > 0 ? (1023-(s32)pos_x) : (((s32)pos_x))) + 1;
        //s32 max_y_steps = (ray_dir_y > 0 ? (1023-(s32)pos_y) : (((s32)pos_y))) + 1;
        // TODO: vectorize
        
        //s32 rem_x_steps[8],rem_y_steps[8];
        //for(int i = 0; i < 8; i++) {
        //    rem_x_steps[i] = (ray_dir_xs[i] > 0.0f ? (MAP_X_SIZE - (s32)pos_x) : ((s32)pos_x)) + 1;
        //    rem_y_steps[i] = (ray_dir_ys[i] > 0.0f ? (MAP_Y_SIZE - (s32)pos_y) : ((s32)pos_y)) + 1;
        //}
        __m256i rem_x_steps = _mm256_add_epi32(
            _mm256_blendv_epi8(
                 _mm256_set1_epi32(MAP_X_SIZE - (s32)pos_x), 
                _mm256_set1_epi32(pos_x),
                 (__m256i)ray_dir_xs_less_than_zero), 
            one_vec);
        __m256i rem_y_steps = _mm256_add_epi32(
            _mm256_blendv_epi8(
                 _mm256_set1_epi32(MAP_Y_SIZE - (s32)pos_y), 
                 _mm256_set1_epi32(pos_y),
                 (__m256i)ray_dir_ys_less_than_zero), 
            one_vec);

        // TODO: vectorize
        //int any_perp_wall_dists_lte_max_z = 0;
        //for(int i = 0; i < 8; i++) {
        //    any_perp_wall_dists_lte_max_z |= perp_wall_dists[i] <= max_z;
        //}
        int any_perp_wall_dists_lte_max_z = _mm256_movemask_ps(_mm256_cmp_ps(perp_wall_dists, max_z_vec, _CMP_LE_OQ));

        // TODO: vectorize
        //int any_prev_drawn_max_ys_gt_next_drawable_min_ys = 0;
        //for(int i = 0; i < 8; i++) {
        //    any_prev_drawn_max_ys_gt_next_drawable_min_ys |= (prev_drawn_max_ys[i] > next_drawable_min_ys[i]);
        //}
        int any_prev_drawn_max_ys_gt_next_drawable_min_ys = _mm256_movemask_epi8(_mm256_cmpgt_epi32(prev_drawn_max_ys, next_drawable_min_ys));

        // TODO: vectorize
        //int any_rem_x_steps_gt_zero = 0;
        //for(int i = 0; i < 8; i++) {
        //    any_rem_x_steps_gt_zero |= (rem_x_steps[i] > 0);
        //}
        int any_rem_x_steps_gt_zero = _mm256_movemask_epi8(_mm256_cmpgt_epi32(rem_x_steps, zero_vec));

        // TODO: vectorize
        //int any_rem_y_steps_gt_zero = 0;
        //for(int i = 0; i < 8; i++) {
        //    any_rem_y_steps_gt_zero |= (rem_y_steps[i] > 0);
        //}
        int any_rem_y_steps_gt_zero = _mm256_movemask_epi8(_mm256_cmpgt_epi32(rem_y_steps, zero_vec));

        while(any_perp_wall_dists_lte_max_z && any_prev_drawn_max_ys_gt_next_drawable_min_ys && any_rem_x_steps_gt_zero && any_rem_y_steps_gt_zero) {

            u32 scalar_prepared_combined_world_map_poses[8] __attribute__ ((aligned (32)));
            _mm256_store_si256(
                (__m256i*)scalar_prepared_combined_world_map_poses,
                _mm256_or_si256(
                    _mm256_slli_epi32(map_xs, 19),
                    _mm256_slli_epi32(map_ys, 8)
                )
            );

                   
            
            __m256 side_dist_xs_less_than_side_dist_ys = _mm256_cmp_ps(side_dist_xs, side_dist_ys, _CMP_LT_OQ);
            __m256i map_x_dxs = _mm256_blendv_epi8(zero_vec, step_xs, (__m256i)side_dist_xs_less_than_side_dist_ys);
            __m256i map_y_dys = _mm256_blendv_epi8(step_ys, zero_vec, (__m256i)side_dist_xs_less_than_side_dist_ys);
            //s32 smap_x_dxs[8];
            //s32 smap_y_dys[8];
            //for(int i = 0; i < 8; i++) {
            //    smap_x_dxs[i] = (side_dist_xs[i] < side_dist_ys[i]) ? step_xs[i] : 0;
            //    smap_y_dys[i] = (side_dist_xs[i] < side_dist_ys[i]) ? 0 : step_ys[i];
            //}
            //__m256i map_x_dxs = _mm256_load_si256(smap_x_dxs);
            //__m256i map_y_dys = _mm256_load_si256(smap_y_dys);
            //__m256i map_x_dxs = _mm256_blendv_epi8(
            //    zero_vec,
            //    _mm256_load_si256(step_xs),
            //    (__m256i)_mm256_cmp_ps(
            //        _mm256_load_ps(side_dist_xs),
            //        _mm256_load_ps(side_dist_ys),
            //         _CMP_LT_OQ));
            //__m256i map_y_dys = _mm256_blendv_epi8(
            //    _mm256_load_si256(step_ys),
            //    zero_vec,
            //    (__m256i)_mm256_cmp_ps(
            //        _mm256_load_ps(side_dist_xs),
            //        _mm256_load_ps(side_dist_ys),
            //         _CMP_LT_OQ));
            

            //perp_wall_dist = next_perp_wall_dist;
            //next_perp_wall_dist = (side_dist_x < side_dist_y) ? side_dist_x : side_dist_y;
            // TODO: vectorize
            //for(int i = 0; i < 8; i++) {
            //    perp_wall_dists[i] = next_perp_wall_dists[i];
            //    next_perp_wall_dists[i] = (side_dist_xs[i] < side_dist_ys[i]) ? side_dist_xs[i] : side_dist_ys[i];
            //}
            perp_wall_dists = next_perp_wall_dists;
            next_perp_wall_dists = _mm256_blendv_ps(side_dist_ys, side_dist_xs, side_dist_xs_less_than_side_dist_ys);




             //f32 side_dist_dx = (side_dist_x < side_dist_y) ? delta_dist_x : 0;
            //f32 side_dist_dy = (side_dist_x < side_dist_y) ? 0 : delta_dist_y;
            //prev_side = next_side;
            //next_side = (side_dist_x < side_dist_y) ? X_SIDE : Y_SIDE;
            //max_x_steps -= (side_dist_x < side_dist_y) ? 1 : 0;
            //max_y_steps -= (side_dist_x < side_dist_y) ? 0 : 1;

            
            // TODO: vectorize
            //f32 side_dist_dxs[8], side_dist_dys[8];
            //for(int i = 0; i < 8; i++) {
            //    side_dist_dxs[i] = (side_dist_xs[i] < side_dist_ys[i]) ? delta_dist_xs[i] : 0.0f;
            //    side_dist_dys[i] = (side_dist_xs[i] < side_dist_ys[i]) ? 0.0f : delta_dist_ys[i];
            //}
            __m256 side_dist_dxs = _mm256_blendv_ps(zero_ps_vec, delta_dist_xs, side_dist_xs_less_than_side_dist_ys);
            __m256 side_dist_dys = _mm256_blendv_ps(delta_dist_ys, zero_ps_vec, side_dist_xs_less_than_side_dist_ys);

            // TODO: vectorize
            //int prev_sides[8], next_sides[8];
            //for(int i = 0; i < 8; i++) {
            //    prev_sides[i] = next_sides[i];
            //    next_sides[i] = (side_dist_xs[i] < side_dist_ys[i]) ? X_SIDE : Y_SIDE;
            //}
            prev_sides = next_sides;
            next_sides = _mm256_blendv_epi8(_mm256_set1_epi32(Y_SIDE), _mm256_set1_epi32(X_SIDE), (__m256i)side_dist_xs_less_than_side_dist_ys);


            // TODO: vectorize
            rem_x_steps = _mm256_sub_epi32(rem_x_steps, _mm256_blendv_epi8(zero_vec, one_vec, (__m256i)side_dist_xs_less_than_side_dist_ys));
            rem_y_steps = _mm256_sub_epi32(rem_y_steps, _mm256_blendv_epi8(one_vec, zero_vec, (__m256i)side_dist_xs_less_than_side_dist_ys));
            //for(int i = 0; i < 8; i++) {
            //    rem_x_steps[i] -= (side_dist_xs[i] < side_dist_ys[i]) ? 1 : 0;
            //    rem_y_steps[i] -= (side_dist_xs[i] < side_dist_ys[i]) ? 0 : 1;
            //}


            //side_dist_x += side_dist_dx;
            //side_dist_y += side_dist_dy;
            // TODO: vectorize
            //for(int i = 0; i < 8; i++) {
            //    side_dist_xs[i] += side_dist_dxs[i];
            //    side_dist_ys[i] += side_dist_dys[i];
            //}
            side_dist_xs = _mm256_add_ps(side_dist_xs, side_dist_dxs);
            side_dist_ys = _mm256_add_ps(side_dist_ys, side_dist_dys);


            //s32 next_map_x = map_x + map_x_dx;
            //s32 next_map_y = map_y + map_y_dy;
            // TODO: vectorize
            
            
            //s32 next_map_xs[8], next_map_ys[8];
            //for(int i = 0; i < 8; i++) {
            //    next_map_xs[i] = map_xs[i] + map_x_dxs[i];
            //    next_map_ys[i] = map_ys[i] + map_y_dys[i];
            //}

            __m256i next_map_xs = _mm256_add_epi32(map_xs, map_x_dxs);
            __m256i next_map_ys = _mm256_add_epi32(map_ys, map_y_dys);


            //if(map_x < 0 || map_x > 1023 || map_y < 0 || map_y > 1023) {
            //    goto next_z_step;
            //}
            // TODO: vectorize
            int inside_map_bounds[8];
            __m256i inside_map_bounds_vec = _mm256_and_si256(
                _mm256_and_si256(
                    _mm256_cmpgt_epi32(map_ys, _mm256_set1_epi32(-1)),
                    _mm256_cmpgt_epi32(_mm256_set1_epi32(1024), map_ys)
                ),
                _mm256_and_si256(
                    _mm256_cmpgt_epi32(map_xs, _mm256_set1_epi32(-1)),
                    _mm256_cmpgt_epi32(_mm256_set1_epi32(1024), map_xs)
                )
            );
            int any_inside_bounds = _mm256_movemask_epi8(inside_map_bounds_vec);
            _mm256_store_si256((__m256i*)inside_map_bounds, inside_map_bounds_vec);
            //for(int i = 0; i < 8; i++) {
                //inside_map_bounds[i] = (map_xs[i] >= 0) && (map_xs[i] < 1024) && (map_ys[i] >= 0) && (map_ys[i] < 1024);
                //any_inside_bounds |= inside_map_bounds[i];
            //}

            if(!any_inside_bounds) {
                goto next_z_step;
            }

            
            //next_invz = scale_height / next_perp_wall_dist;
            //u32 voxelmap_idx = get_voxelmap_idx(map_x, map_y);
            // TODO: vectorize
            //for(int i = 0; i < 8; i++) {
            //    next_invzs[i] = scale_height / next_perp_wall_dists[i];
            //}
            next_invzs = _mm256_div_ps(scale_height_vec, next_perp_wall_dists);
            
            __m256i voxelmap_idxs = get_voxelmap_idx_256(
                map_xs, map_ys
                //_mm256_load_si256((__m256i*)map_xs),
                //_mm256_load_si256((__m256i*)map_ys)
            );
            //u32 voxelmap_idxs[8];
            //for(int i = 0; i < 8; i++) {
            //    voxelmap_idxs[i] = get_voxelmap_idx(map_xs[i], map_ys[i]);
            //}

            
            //f32 normal_pt1 = ((prev_side == X_SIDE) ? encoded_x_side_norm.x : encoded_y_side_norm.x);
            //f32 normal_pt2 = ((prev_side == X_SIDE) ? encoded_x_side_norm.y : encoded_y_side_norm.y);
            //f32 top_face_norm_pt1 = encoded_top_face_norm.x;
            //f32 top_face_norm_pt2 = encoded_top_face_norm.y;
            //f32 bot_face_norm_pt1 = encoded_bot_face_norm.x;
            //f32 bot_face_norm_pt2 = encoded_bot_face_norm.y;
            

            //column_header* header = &columns_header_data[voxelmap_idx];
            // TODO: vectorize

            //column_header* headers[8];
            //for(int i = 0; i < 8; i++) {
            //    headers[i] = &columns_header_data[voxelmap_idxs[i]];
            //}
            __m256i headers = _mm256_i32gather_epi32((const int*)columns_header_data, voxelmap_idxs, 2);
            __m256i header_top_ys = _mm256_and_si256(_mm256_set1_epi32(0xFF), headers);
            __m256i header_num_runs = _mm256_and_si256(_mm256_set1_epi32(0xFF), _mm256_srli_epi32(headers, 8));


            //if(header->num_runs == 0) { goto next_z_step; }
            // TODO: vectorize
            //int any_headers_have_runs = 0;
            //for(int i = 0; i < 8; i++) {
            //    any_headers_have_runs |= (headers[i]->num_runs > 0);
            //}
            int any_headers_have_runs = _mm256_movemask_epi8(_mm256_cmpgt_epi32(header_num_runs, _mm256_set1_epi32(0)));

            if(!any_headers_have_runs) {
                goto next_z_step;
            }

            //s32 top_ys[8] __attribute__ ((aligned (32)));
            //_mm256_store_si256((__m256i*)top_ys, header_top_ys);


            // check the top of this column against the bottom of the frustum skip drawing it
            //f32 relative_top_of_col =  height - (255 - header->top_y); //[voxelmap_idx];
            //f32 col_top_invz = (relative_top_of_col < 0 ? invz : next_invz);
            //f32 float_top_of_col_projected_height = relative_top_of_col*col_top_invz;
            //s32 int_top_of_col_projected_height = floor(float_top_of_col_projected_height) + pitch;
            //if(int_top_of_col_projected_height >= prev_drawn_max_y) {
            //    goto next_z_step;
            //}
            // TODO: vectorize
            //f32 relative_tops_of_cols[8];
            //f32 col_top_invzs[8];
            //f32 float_top_of_col_projected_heights[8];
            //s32 int_top_of_col_projected_heights[8];

            //for(int i = 0; i < 8; i++) {
            //    relative_tops_of_cols[i] = height - (255 - top_ys[i]);
            //}

            __m256 relative_tops_of_cols = _mm256_sub_ps(height_vec, 
                                                _mm256_cvtepi32_ps(
                                                    _mm256_sub_epi32(two_fifty_five_vec, header_top_ys)));

            __m256 relative_tops_of_cols_lt_zero =  _mm256_cmp_ps(relative_tops_of_cols, zero_ps_vec, _CMP_LT_OQ);
            __m256 col_top_invzs = _mm256_blendv_ps(next_invzs, invzs, relative_tops_of_cols_lt_zero);

            __m256 float_top_of_col_projected_heights = _mm256_mul_ps(relative_tops_of_cols, col_top_invzs);
            __m256i int_top_of_col_projected_heights = _mm256_add_epi32(
                            _mm256_cvtps_epi32(
                                _mm256_floor_ps(float_top_of_col_projected_heights)
                            ), 
                        pitch_vec);

            int any_top_of_col_projected_heights_lt_prev_drawn_max_y = _mm256_movemask_epi8(
                _mm256_cmpgt_epi32(prev_drawn_max_ys, int_top_of_col_projected_heights)
            );
            if(!any_top_of_col_projected_heights_lt_prev_drawn_max_y) {
                goto next_z_step;
            }

            //for(int i = 0; i < 8; i++) {
            //    col_top_invzs[i] = (relative_tops_of_cols[i] < 0 ? invzs[i] : next_invzs[i]);
            //}
            //for(int i = 0; i < 8; i++) {
            //    float_top_of_col_projected_heights[i] = relative_tops_of_cols[i] * col_top_invzs[i];
            //}
            //for(int i = 0; i < 8; i++) {
            //    int_top_of_col_projected_heights[i] = floor(float_top_of_col_projected_heights[i])+pitch;
            //}
            //int all_top_of_col_projected_heights_gte_prev_drawn_max_y = 1;
            //for(int i = 0; i < 8; i++) {
            //    all_top_of_col_projected_heights_gte_prev_drawn_max_y &= (int_top_of_col_projected_heights[i] >= prev_drawn_max_ys[i]);
            //}
            //if(all_top_of_col_projected_heights_gte_prev_drawn_max_y) {
            //    goto next_z_step;
            //}



            //f32 relative_bot_of_col = height - (255 - cur_map_max_height); //span_info[num_runs-1].bot);
            //f32 col_bot_invz = (relative_bot_of_col < 0 ? next_invz : invz);
            //f32 float_bot_of_col_projected_height = relative_bot_of_col * col_bot_invz;
            //s32 int_bot_of_col_projected_height = floor(float_bot_of_col_projected_height) + pitch;
            //if(int_bot_of_col_projected_height < next_drawable_min_y) {
            //    goto next_z_step;
            //}
            // TODO: vectorize
            //f32 relative_bots_of_cols[8];
            //f32 col_bot_invzs[8];
            //f32 float_bot_of_col_projected_heights[8];
            //s32 int_bot_of_col_projected_heights[8];
            //for(int i = 0; i < 8; i++) {
            //    relative_bots_of_cols[i] = height - ( 255 - cur_map_max_height);
            //}
            __m256 relative_bots_of_cols = _mm256_sub_ps(height_vec, 
                                                _mm256_cvtepi32_ps(
                                                    _mm256_sub_epi32(two_fifty_five_vec, cur_map_max_height_vec)));
            //for(int i = 0; i < 8; i++) {
            //    col_bot_invzs[i] = (relative_bots_of_cols[i] < 0 ? next_invzs[i] : invzs[i]);
            //}
            __m256 relative_bots_of_cols_lt_zero =  _mm256_cmp_ps(relative_bots_of_cols, zero_ps_vec, _CMP_LT_OQ);
            __m256 col_bot_invzs = _mm256_blendv_ps(next_invzs, invzs, relative_bots_of_cols_lt_zero);
            //for(int i = 0; i < 8; i++) {
            //    float_bot_of_col_projected_heights[i] = relative_bots_of_cols[i] * col_bot_invzs[i];
            //}
            __m256 float_bot_of_col_projected_heights = _mm256_mul_ps(relative_bots_of_cols, col_bot_invzs);
            __m256i int_bot_of_col_projected_heights = _mm256_add_epi32(
                            _mm256_cvtps_epi32(
                                _mm256_floor_ps(float_bot_of_col_projected_heights)
                            ), 
                        pitch_vec);

            //for(int i = 0; i < 8; i++) {
            //    int_bot_of_col_projected_heights[i] = floor(float_bot_of_col_projected_heights[i]) + pitch;
            //}
            //int all_bot_of_col_projected_heights_lt_next_drawable_min_y = 1;
            //for(int i = 0; i < 8; i++) {
            //    all_bot_of_col_projected_heights_lt_next_drawable_min_y &= (int_bot_of_col_projected_heights[i] < next_drawable_min_ys[i]);
            //}
            
            int any_bot_of_col_projected_heights_gte_next_drawable_min_y = _mm256_movemask_epi8(
                _mm256_or_si256(
                    _mm256_cmpgt_epi32(prev_drawn_max_ys, int_top_of_col_projected_heights),
                    _mm256_cmpeq_epi32(prev_drawn_max_ys, int_top_of_col_projected_heights)
                )
            );

            if(!any_bot_of_col_projected_heights_gte_next_drawable_min_y) {
                goto next_z_step;
            }




          

#define DRAW_CHUNK_FACE(top, bot, face_norm_1, face_norm_2, voxel_color_idx) {              \
    u32 fb_idx = fb_swizzle(x,top);                                                         \
    u32 color = top_of_col_color_ptr[voxel_color_idx];                                      \
    f32 norm_pt1 = top_of_col_norm_pt1_ptr[voxel_color_idx];                                \
    f32 norm_pt2 = top_of_col_norm_pt2_ptr[voxel_color_idx];                                \
    for(int y = top; y < bot; y++) {                                                        \
        u8 occlusion_bit = get_occlusion_bit(x, y);                                         \
        u32 combined_world_pos = prepared_combined_world_map_pos|(voxel_color_idx);         \
        f32 old_norm_pt1 = norm_buffer[fb_idx*2];                                           \
        f32 old_norm_pt2 = norm_buffer[fb_idx*2+1];                                         \
        f32 new_norm_pt1 = occlusion_bit ? old_norm_pt1 : norm_pt1;                         \
        f32 new_norm_pt2 = occlusion_bit ? old_norm_pt2 : norm_pt2;                         \
        u32 old_color = albedo_buffer[fb_idx];                                              \
        u32 mixed_color = mix_colors(old_color, color);                                     \
        u32 old_world_pos = world_pos_buffer[fb_idx];                                       \
        u32 new_world_pos = occlusion_bit ? old_world_pos : combined_world_pos;             \
        u32 new_color = (occlusion_bit ? old_color : mixed_color);                          \
        u8 mixed_alpha_coverage = (mixed_color>>24)&0b11;                                   \
        u8 new_occlusion_bit = (occlusion_bit ? 1 : (mixed_alpha_coverage == 0b11) ? 1 : 0);\
        min_coverage = (occlusion_bit ? min_coverage : min(mixed_alpha_coverage, min_coverage));\
        set_occlusion_bit(x, y, new_occlusion_bit);                                         \
        norm_buffer[fb_idx*2] = new_norm_pt1;                                               \
        norm_buffer[fb_idx*2+1] = new_norm_pt2;                                             \
        world_pos_buffer[fb_idx] = new_world_pos;                                           \
        albedo_buffer[fb_idx] = new_color;                                                  \
        fb_idx += 8;                                                                        \
    }                                                                                       \
}

// draw side of chunk
// needs voxel color interpolation
#define DRAW_CHUNK_SIDE(top, bot, face_norm_pt1, face_norm_pt2) {                           \
    s32 clipped_top_y = clipped_top_side_height - int_top_side_projected_height;            \
    f32 texel_per_y = ((f32)num_voxels) / unclipped_screen_dy;                              \
    f32 cur_voxel_color_idx = (f32)clipped_top_y * texel_per_y;                             \
    u32 fb_idx = fb_swizzle(x,top);                                                         \
    for(int y = top; y < bot; y++) {                                                        \
        u8 occlusion_bit = get_occlusion_bit(x, y);                                         \
        u16 voxel_color_idx = cur_voxel_color_idx;                                          \
        u32 color = color_ptr[voxel_color_idx];                                             \
        f32 norm_pt1 = top_of_col_norm_pt1_ptr[voxel_color_idx];                            \
        f32 norm_pt2 = top_of_col_norm_pt2_ptr[voxel_color_idx];                            \
        u32 old_color = albedo_buffer[fb_idx];                                              \
        u32 mixed_color = mix_colors(old_color, color);                                     \
        u32 combined_world_pos = prepared_combined_world_map_pos|((voxel_color_idx+color_ptr)-top_of_col_color_ptr);         \
        cur_voxel_color_idx += texel_per_y;                                                 \
        f32 old_norm_pt1 = norm_buffer[fb_idx*2];                                           \
        f32 old_norm_pt2 = norm_buffer[fb_idx*2+1];                                         \
        f32 new_norm_pt1 = occlusion_bit ? old_norm_pt1 : (norm_pt1);                       \
        f32 new_norm_pt2 = occlusion_bit ? old_norm_pt2 : (norm_pt2);                       \
        u32 old_world_pos = world_pos_buffer[fb_idx];                                       \
        u32 new_world_pos = occlusion_bit ? old_world_pos : combined_world_pos;             \
        u32 new_color = (occlusion_bit ? old_color : mixed_color);                          \
        u8 mixed_alpha_coverage = (mixed_color>>24)&0b11;                                   \
        u8 new_occlusion_bit = (occlusion_bit ? 1 : (mixed_alpha_coverage == 0b11) ? 1 : 0);             \
        min_coverage = (occlusion_bit ? min_coverage : min(mixed_alpha_coverage, min_coverage));\
        set_occlusion_bit(x, y, new_occlusion_bit);                                          \
        norm_buffer[fb_idx*2] = new_norm_pt1;                                               \
        norm_buffer[fb_idx*2+1] = new_norm_pt2;                                             \
        world_pos_buffer[fb_idx] = new_world_pos;                                           \
        albedo_buffer[fb_idx] = new_color;                                                  \
        fb_idx += 8;                                                                        \
    }                                                                                       \
}

#define CLAMP(a, mi, ma) max(mi, min(a, ma))



#define DRAW_CHUNK(chunk_top, chunk_bot, is_top_chunk, is_transparent_chunk, break_if_top_below_screen, break_if_bot_above_screen, side_norm_pt1, side_norm_pt2, face_norm_pt1, face_norm_pt2) {         \
    f32 relative_bot = height-chunk_bot;                                                 \
    f32 relative_top = height-chunk_top;                                                 \
    s32 int_top_face_projected_height = floor(relative_top*next_invz) + pitch;           \
    s32 int_top_side_projected_height = floor(relative_top*invz) + pitch;                \
    s32 int_bot_face_projected_height = floor(relative_bot*next_invz) + pitch;           \
    s32 int_bot_side_projected_height = floor(relative_bot*invz) + pitch;                \
    s32 top_projected_heightonscreen = (!is_top_chunk) ? int_top_side_projected_height : min(int_top_side_projected_height, int_top_face_projected_height);\
    s32 bot_projected_heightonscreen = is_top_chunk ? int_bot_side_projected_height : max(int_bot_side_projected_height,int_bot_face_projected_height);  \
    u8 min_coverage = 0b11;                                                              \
    if(break_if_bot_above_screen && bot_projected_heightonscreen < next_drawable_min_y) { \
        break;                                                                            \
    }                                                                                     \
    if(break_if_top_below_screen && top_projected_heightonscreen >= prev_drawn_max_y) {   \
        break;                                                                            \
    }                                                                                     \
    s32 clipped_top_heightonscreen = CLAMP(top_projected_heightonscreen, next_drawable_min_y, prev_drawn_max_y);    \
    s32 clipped_bot_heightonscreen = CLAMP(bot_projected_heightonscreen, next_drawable_min_y, prev_drawn_max_y);       \
    s32 unclipped_screen_dy = int_bot_side_projected_height - int_top_side_projected_height;    \
    s32 num_voxels = (chunk_top - chunk_bot);                                           \
    if(clipped_top_heightonscreen < clipped_bot_heightonscreen) {                        \
        s32 clipped_top_face_height = CLAMP(int_top_face_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_top_side_height = CLAMP(int_top_side_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_bot_face_height = CLAMP(int_bot_face_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_bot_side_height = CLAMP(int_bot_side_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        if(is_top_chunk && clipped_top_face_height < clipped_top_side_height) {          \
            u16 color_offset = (color_ptr - top_of_col_color_ptr);                       \
            DRAW_CHUNK_FACE(clipped_top_face_height, clipped_top_side_height, top_face_norm_pt1, top_face_norm_pt2, color_offset); \
        }                                                                                \
        if(clipped_top_side_height < clipped_bot_side_height) {                          \
            DRAW_CHUNK_SIDE(clipped_top_side_height, clipped_bot_side_height,  side_norm_pt1, side_norm_pt2); \
        }                                                                                \
        if(clipped_bot_side_height < clipped_bot_face_height) {       \
            u16 color_offset = ((color_ptr - top_of_col_color_ptr)+num_voxels)-1;        \
            DRAW_CHUNK_FACE(clipped_bot_side_height, clipped_bot_face_height, bot_face_norm_pt1, bot_face_norm_pt2, color_offset); \
        }                                                                                \
        s32 next_prev_drawn_max_y = (min_coverage < 0b11) ? prev_drawn_max_y : (bot_projected_heightonscreen >= prev_drawn_max_y && top_projected_heightonscreen < prev_drawn_max_y) ? top_projected_heightonscreen : prev_drawn_max_y;             \
        s32 next_next_drawable_min_y = (min_coverage < 0b11) ? next_drawable_min_y :  (top_projected_heightonscreen <= next_drawable_min_y && bot_projected_heightonscreen > next_drawable_min_y) ? bot_projected_heightonscreen : next_drawable_min_y; \
        prev_drawn_max_y = next_prev_drawn_max_y; \
        next_drawable_min_y = next_next_drawable_min_y; \
    }   \
}           


       
#define DRAW_CHUNK_BOTTOM_UP(chunk_top, chunk_bot, is_top_chunk, is_transparent_chunk, break_if_top_below_screen, break_if_bot_above_screen, side_norm_pt1, side_norm_pt2, face_norm_pt1, face_norm_pt2) {         \
    f32 relative_bot = height-chunk_bot;                                                 \
    f32 relative_top = height-chunk_top;                                                 \
    s32 int_top_face_projected_height = floor(relative_top*next_invz) + pitch;           \
    s32 int_top_side_projected_height = floor(relative_top*invz) + pitch;                \
    s32 int_bot_face_projected_height = floor(relative_bot*next_invz) + pitch;           \
    s32 int_bot_side_projected_height = floor(relative_bot*invz) + pitch;                \
    s32 top_projected_heightonscreen = (!is_top_chunk) ? int_top_side_projected_height : min(int_top_side_projected_height, int_top_face_projected_height);\
    s32 bot_projected_heightonscreen = is_top_chunk ? int_bot_side_projected_height : max(int_bot_side_projected_height,int_bot_face_projected_height);  \
    if(break_if_bot_above_screen && bot_projected_heightonscreen < next_drawable_min_y) { \
        break;                                                                            \
    }                                                                                     \
    if(break_if_top_below_screen && top_projected_heightonscreen >= prev_drawn_max_y) {   \
        break;                                                                            \
    }                                                                                     \
    s32 next_prev_drawn_max_y = is_transparent_chunk ? prev_drawn_max_y : (bot_projected_heightonscreen >= prev_drawn_max_y && top_projected_heightonscreen < prev_drawn_max_y) ? top_projected_heightonscreen : prev_drawn_max_y;             \
    s32 next_next_drawable_min_y = is_transparent_chunk ? next_drawable_min_y :  (top_projected_heightonscreen <= next_drawable_min_y && bot_projected_heightonscreen > next_drawable_min_y) ? bot_projected_heightonscreen : next_drawable_min_y; \
    s32 clipped_top_heightonscreen = CLAMP(top_projected_heightonscreen, next_drawable_min_y, prev_drawn_max_y);    \
    s32 clipped_bot_heightonscreen = CLAMP(bot_projected_heightonscreen, next_drawable_min_y, prev_drawn_max_y);       \
    s32 unclipped_screen_dy = int_bot_side_projected_height - int_top_side_projected_height;    \
    s32 num_voxels = (chunk_top - chunk_bot);                                           \
    if(clipped_top_heightonscreen < clipped_bot_heightonscreen) {                        \
        s32 clipped_top_face_height = CLAMP(int_top_face_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_top_side_height = CLAMP(int_top_side_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_bot_face_height = CLAMP(int_bot_face_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
        s32 clipped_bot_side_height = CLAMP(int_bot_side_projected_height, next_drawable_min_y, prev_drawn_max_y);  \
                                                                                        \
        if(clipped_bot_side_height < clipped_bot_face_height) {                         \
            u16 color_offset = ((color_ptr - top_of_col_color_ptr)+num_voxels)-1;        \
            DRAW_CHUNK_FACE(clipped_bot_side_height, clipped_bot_face_height, bot_face_norm_pt1, bot_face_norm_pt2, color_offset); \
        }                                                                                \
        if(clipped_top_side_height < clipped_bot_side_height) {                          \
            DRAW_CHUNK_SIDE(clipped_top_side_height, clipped_bot_side_height,  side_norm_pt1, side_norm_pt2); \
        }                                                                                   \
        if(is_top_chunk && clipped_top_face_height < clipped_top_side_height) {          \
            u16 color_offset = (color_ptr - top_of_col_color_ptr);                       \
            DRAW_CHUNK_FACE(clipped_top_face_height, clipped_top_side_height, top_face_norm_pt1, top_face_norm_pt2, color_offset); \
        }                                                                        \
        prev_drawn_max_y = next_prev_drawn_max_y; \
        next_drawable_min_y = next_next_drawable_min_y; \
    }   \
}           
  
  
            //span* span_info = columns_runs_data[voxelmap_idx].runs_info;
            //u8 num_runs = header->num_runs;
            // TODO: vectorize
            span* span_infos[8] __attribute__ ((aligned (32)));
            s32 scalar_num_runs[8] __attribute__ ((aligned (32)));
            _mm256_store_si256((__m256i*)scalar_num_runs, header_num_runs);
            // load voxelmap indexes
            // load 

            u32 scalar_voxelmap_idxs[8] __attribute__ ((aligned (32)));
            _mm256_store_si256((__m256i*)scalar_voxelmap_idxs, voxelmap_idxs);

            for(int i = 0; i < 8; i++) {
                span_infos[i] = columns_runs_data[scalar_voxelmap_idxs[i]].runs_info;
            }


            //for(int i = 0; i < 8; i++) {
            //    col_num_runs[i] = headers[i]->num_runs;
            //}

            f32 scalar_invzs[8] __attribute__ ((aligned (32))); 
            _mm256_store_ps(scalar_invzs, invzs);
            f32 scalar_next_invzs[8] __attribute__ ((aligned (32))); 
            _mm256_store_ps(scalar_next_invzs, next_invzs);
            
            
            s32 scalar_prev_drawn_max_ys[8] __attribute__ ((aligned (32)));
            s32 scalar_next_drawable_min_ys[8] __attribute__ ((aligned (32)));
            _mm256_store_si256((__m256i*)scalar_prev_drawn_max_ys, prev_drawn_max_ys);
            _mm256_store_si256((__m256i*)scalar_next_drawable_min_ys, next_drawable_min_ys);

            int old_x = x;
            //__mm256_no
            for(int i = 0; i < 8; i++) {
                if(!inside_map_bounds[i]) { continue; }
                if(prev_drawn_max_ys[i] <= next_drawable_min_ys[i]) { continue; }
                //if(!any_prev_drawn_max_ys_gt_next_drawable_min_ys)
                int x = old_x + i;
                u32 voxelmap_idx = scalar_voxelmap_idxs[i];

                span* span_info = span_infos[i];
                f32 invz = scalar_invzs[i];
                f32 next_invz = scalar_next_invzs[i];
                f32 next_drawable_min_y = scalar_next_drawable_min_ys[i];
                f32 prev_drawn_max_y = scalar_prev_drawn_max_ys[i];
                u32 prepared_combined_world_map_pos = scalar_prepared_combined_world_map_poses[i];
                int num_runs = scalar_num_runs[i];


                u32* top_of_col_color_ptr = columns_colors_data[voxelmap_idx].colors;
                f32* top_of_col_norm_pt1_ptr = columns_norm_data[voxelmap_idx].norm_pt1;
                f32* top_of_col_norm_pt2_ptr = columns_norm_data[voxelmap_idx].norm_pt2;
                u32* color_ptr = top_of_col_color_ptr;

                if(looking_up) { //looking_up) {
                    // draw from bottom to the top 
                    // iterate over runs real quick to get the color pointer to the bottom
                    for(int run = 0; run < num_runs; run++) {
                        color_ptr += ((span_info[run].top_surface_end+1)-span_info[run].top_surface_start);
                        if(span_info[run].bot_surface_end > span_info[run].bot_surface_start) {
                            color_ptr += ((span_info[run].bot_surface_end)-span_info[run].bot_surface_start);
                        }
                    }

                    // now go back in reverse
                    for(int run = num_runs-1; run >= 0; run--) {   
                        int bot_surface_top = 255 - span_info[run].bot_surface_start;
                        int bot_surface_end = 255 - span_info[run].bot_surface_end;
                        
                                
                        if(span_info[run].bot_surface_end > span_info[run].bot_surface_start) {
                            color_ptr -= (span_info[run].bot_surface_end-span_info[run].bot_surface_start);

                            DRAW_CHUNK(
                                bot_surface_top, bot_surface_end, 
                                0, //span_info[run].is_top, 
                                ((color_ptr[0]>>24)&0b11) != 0b11,//0, //span_info[run].is_transparent, 
                                0, 1, normal_pt1, normal_pt2, top_face_norm_pt1, top_face_norm_pt2
                            );
                        }

                        int top_surface_top = 255 - span_info[run].top_surface_start;
                        int top_surface_bot = 255 - (span_info[run].top_surface_end+1);
                        color_ptr -= ((span_info[run].top_surface_end+1)-span_info[run].top_surface_start);

                        DRAW_CHUNK(
                            top_surface_top, top_surface_bot, 
                            1, //span_info[run].is_top, 
                            ((color_ptr[0]>>24)&0b11) != 0b11, //0, //span_info[run].is_transparent, 
                            0, 1, normal_pt1, normal_pt2, top_face_norm_pt1, top_face_norm_pt2
                        );
                    } 


                } else {
                    // TOP DOWN LOOP

                    for(int run = 0; run < num_runs; run++) {
                        int top_surface_top = 255 - span_info[run].top_surface_start;
                        int top_surface_bot = 255 - (span_info[run].top_surface_end+1);
                        int bot_surface_top = 255 - span_info[run].bot_surface_start;
                        int bot_surface_bot = 255 - span_info[run].bot_surface_end;

                        DRAW_CHUNK(
                            top_surface_top, top_surface_bot, 
                            1, //span_info[run].is_top, 
                            ((color_ptr[0]>>24)&0b11) != 0b11, //0, //span_info[run].is_transparent, 
                            1, 0, normal_pt1, normal_pt2, top_face_norm_pt1, top_face_norm_pt2
                        );
                        color_ptr += ((span_info[run].top_surface_end+1)-span_info[run].top_surface_start);
                        if(span_info[run].bot_surface_end > span_info[run].bot_surface_start) {
                            DRAW_CHUNK(
                                bot_surface_top, bot_surface_bot, 
                                0, //span_info[run].is_top, 
                                ((color_ptr[0]>>24)&0b11) != 0b11, //0, //span_info[run].is_transparent, 
                                1, 0, normal_pt1, normal_pt2, top_face_norm_pt1, top_face_norm_pt2
                            );
                            color_ptr += (span_info[run].bot_surface_end-span_info[run].bot_surface_start);
                        }

                    }
                }

                scalar_next_drawable_min_ys[i] = next_drawable_min_y;
                scalar_prev_drawn_max_ys[i] = prev_drawn_max_y;
            }
            x = old_x;
            next_drawable_min_ys = _mm256_load_si256((__m256i*)scalar_next_drawable_min_ys);
            prev_drawn_max_ys = _mm256_load_si256((__m256i*)scalar_prev_drawn_max_ys);

        next_z_step:;
            //map_x = next_map_x;
            //map_y = next_map_y;
            //invz = next_invz;
            // TODO: vectorize
            map_xs = next_map_xs;
            map_ys = next_map_ys;
            //for(int i = 0; i < 8; i++) {
            //    map_xs[i] = next_map_xs[i];
            //}
            //for(int i = 0; i < 8; i++) {
            //    map_ys[i] = next_map_ys[i];
            //}
            //for(int i = 0; i < 8; i++) {
            //    invzs[i] = next_invzs[i];
            //}
            invzs = next_invzs;

            // TODO: vectorize
            any_perp_wall_dists_lte_max_z = _mm256_movemask_ps(_mm256_cmp_ps(perp_wall_dists, max_z_vec, _CMP_LE_OQ));
            //for(int i = 0; i < 8; i++) {
            //    any_perp_wall_dists_lte_max_z |= perp_wall_dists[i] <= max_z;
            //}

            // TODO: vectorize
            any_prev_drawn_max_ys_gt_next_drawable_min_ys = _mm256_movemask_epi8(_mm256_cmpgt_epi32(prev_drawn_max_ys, next_drawable_min_ys));
            //for(int i = 0; i < 8; i++) {
            //    any_prev_drawn_max_ys_gt_next_drawable_min_ys |= (prev_drawn_max_ys[i] > next_drawable_min_ys[i]);
            //}

            // TODO: vectorize
            any_rem_x_steps_gt_zero = _mm256_movemask_epi8(_mm256_cmpgt_epi32(rem_x_steps, zero_vec));
            //for(int i = 0; i < 8; i++) {
            //    any_rem_x_steps_gt_zero |= (rem_x_steps[i] > 0);
            //}

            // TODO: vectorize
            any_rem_y_steps_gt_zero = _mm256_movemask_epi8(_mm256_cmpgt_epi32(rem_y_steps, zero_vec));
            //for(int i = 0; i < 8; i++) {
            //    any_rem_y_steps_gt_zero |= (rem_y_steps[i] > 0);
            //}

        }

        xs = _mm256_add_epi32(xs, dxs_vec);
        



    }
    EndTimeBlock(raycast_vector_block);

    //printf("max mip level of %i\n", max_mip);
}