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


#define DAY_MAX_Z 800
#define NIGHT_MAX_Z 512
#define NO_FOG_MAX_Z 2048
    
#define PI 3.14159265359

#define PROFILER 0
#ifdef DEBUG
// normally 12 :)
#define NUM_RENDER_THREADS 1
#define NUM_LIGHT_MAP_THREADS 1
#else
#define NUM_RENDER_THREADS 10
#define NUM_LIGHT_MAP_THREADS 6
#endif

#include "selectable_profiler.c"
#include <SDL2/SDL.h>


#include "utils.h"

u32 background_color = DAY_BACKGROUND_COLOR;
f32 amb_light_factor = .4;


s32 render_size = 0;


//int output_width = 1920;
//int output_height = 1080;

//#define OUTPUT_WIDTH  1920
//#define OUTPUT_HEIGHT 1080

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

f32 pos_x = 10.0f;
f32 pos_y = 10.0f;
f32 dir_x = 1.0f;
f32 dir_y = 0.0f;
f32 pos_z = -10.0f; //15.0f;
//static f32 height = 240.0;

float deg_to_rad(float degrees) {
    return degrees * PI / 180.0;
}

f32 desired_fov_degrees = 120;


f32 plane_x = 0.0;
f32 plane_y = -1.20;

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
int sprint = 0;
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
    //FACE_LIGHTING = 1,
    NORMALS_LIGHTING = 1
    //FANCY_LIGHTING = 2,
} lighting_modes;

char* lighting_mode_strs[] = {
    "disabled",
    //"face",
    "normal"
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
static int lighting = NO_LIGHTING;
static int transparency = 0;
static int ambient_occlusion = 1;
static int render_6dof = 0;

static int view = VIEW_STANDARD;
static int gravmode = 1;

static int double_pixels = 1;


const int output_widths[] = {2560,1920,1280,1024};
const int output_heights[] = {1440,1080,720,768};

u32 cur_output_size_idx = 2;

#define OUTPUT_WIDTH (output_widths[cur_output_size_idx])
#define OUTPUT_HEIGHT (output_heights[cur_output_size_idx])

int render_width = 1280/2;//1920/2; 
int render_height = 720/2;//1080/2;
f32 aspect_ratio = (1280/2.0)/(720/2.0); //(1920/2.0)/(1080/2.0);

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
    volatile uint64_t working;
} thread_params;

typedef void (*raw_render_func)(s32 min_x, s32 min_y, s32 max_x, s32 max_y);

typedef struct {
    int num_jobs;
    PTP_WORK_CALLBACK func;
    raw_render_func raw_func;
    thread_params *parms;
} job_pool;

void start_pool(thread_pool* tp, job_pool* rp) {
    if(rp->num_jobs > 1) {
        for(int i = 0; i < rp->num_jobs; i++) {
            thread_pool_add_work(tp, rp->func, (void*)&rp->parms[i]);
        }
    } else {
        // non-threading
        rp->raw_func(
            rp->parms[0].min_x, 
            rp->parms[0].min_y,
            rp->parms[0].max_x,
            rp->parms[0].max_y
        );
    }
}

void start_pool_new_jobs_always_threaded(thread_pool* tp, job_pool* rp) {
    for(int i = 0; i < rp->num_jobs; i++) {

        if(rp->parms[i].working) { continue; }
        rp->parms[i].working = 1;
        thread_pool_add_work(tp, rp->func, (void*)&rp->parms[i]);
    }
}

void wait_for_job_pool_to_finish(job_pool* p) {
    if(p->num_jobs > 1) {
        while(1) {
            top_of_wait_loop:;
            for(int i = 0; i < p->num_jobs; i++) {
                if(p->parms[i].finished == 0) { goto top_of_wait_loop; }
            }
            break;
        }
    }
}

thread_pool *render_pool, *map_pool;


int falling = 1;

int eye_height = 11;
int knee_height = 4;

f32 accel = .68;
f32 vel = 0.0f;
f32 max_accel = 80.0f;

f32 move_forward_speed = .7f;
f32 strafe_speed = 1.8f;
f32 fly_speed = 2.2f;

f32 cam_pos_z = 0.0f;
#include "voxelmap.c"

void init_player_positions() {
    vel = 0.0f;
    pos_x = 10.0f;
    pos_y = 10.0f;
    pos_z = columns_header_data[get_voxelmap_idx(pos_x,pos_y)].top_y - 20.0f;
    cam_pos_z = pos_z;
}

void next_map() {
    load_map(cur_map++);
    init_player_positions();
}

void reload_map() {
    load_map(--cur_map); 
    init_player_positions();
}


int debug_print = 0;

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
        case SDL_SCANCODE_LSHIFT:
            sprint = 0;
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
            render_6dof = !render_6dof;
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
        case SDL_SCANCODE_P:
            debug_print = !debug_print;
            break;

        // change output resolution, window size
        case SDL_SCANCODE_Y: do {
            cur_output_size_idx++;
            if(cur_output_size_idx >= (sizeof(output_widths)/sizeof(output_widths[0]))) { cur_output_size_idx = 0; }

            if(double_pixels == 1) {
                render_width = OUTPUT_WIDTH/2;
                render_height = OUTPUT_HEIGHT/2;
            } else if (double_pixels == 2) {
                render_width = OUTPUT_WIDTH/4;
                render_height = OUTPUT_HEIGHT/4;
            } else {
                render_width = OUTPUT_WIDTH;
                render_height = OUTPUT_HEIGHT;
            }
            aspect_ratio = ((f32)render_width)/render_height;
            setup_render_size = 0;
        } while(0);
        break;

        // change internal render resolution
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
                //SDL_CaptureMouse(SDL_FALSE);
                //SDL_SetRelativeMouseMode(SDL_FALSE);
                //int err = SDL_CaptureMouse(SDL_TRUE);
                SDL_SetRelativeMouseMode(SDL_FALSE);
                SDL_CaptureMouse(SDL_FALSE);
                //SDL_ShowCursor(0);
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
            forwardbackward = move_forward_speed;
            break;
        case SDL_SCANCODE_S:
            forwardbackward = -move_forward_speed;
            break;
        case SDL_SCANCODE_A: 
            strafe = -strafe_speed;
            break;
        case SDL_SCANCODE_D: 
            strafe = +strafe_speed;
            break;
        case SDL_SCANCODE_Z:
            updown = -fly_speed;
            break;
        case SDL_SCANCODE_X:
            updown = +fly_speed;
            break;
        case SDL_SCANCODE_LSHIFT:
            sprint = 1;
            break;
    }
}


// forward, backward, left, and right movement
void move_4dof(f32 old_pos_x, f32 old_pos_y, f32 old_pos_z, f32 new_pos_x, f32 new_pos_y) {
    if(!gravmode) {
        pos_x = new_pos_x;
        pos_y = new_pos_y;
        return;
    }
    f32 contact_point = old_pos_z+eye_height;
    //int go_up_height = (pos_z-KNEE_HEIGHT)+EYE_HEIGHT;

    s32 floor_new_pos_x1 = floor(new_pos_x);
    s32 floor_new_pos_x2 = floor_new_pos_x1+1;
    s32 floor_new_pos_y1 = floor(new_pos_y);
    s32 floor_new_pos_y2 = floor_new_pos_y1+1;
    
    int too_tall_to_go_up_height = (pos_z-knee_height-1)+eye_height;
    s32 blocked_x1y1 = check_for_solid_voxel_in_aabb(floor_new_pos_x1, floor_new_pos_y1, too_tall_to_go_up_height, 1, 1, 0);
    s32 blocked_x2y1 = check_for_solid_voxel_in_aabb(floor_new_pos_x2, floor_new_pos_y1, too_tall_to_go_up_height, 1, 1, 0);
    s32 blocked_x1y2 = check_for_solid_voxel_in_aabb(floor_new_pos_x1, floor_new_pos_y2, too_tall_to_go_up_height, 1, 1, 0);
    s32 blocked_x2y2 = check_for_solid_voxel_in_aabb(floor_new_pos_x2, floor_new_pos_y2, too_tall_to_go_up_height, 1, 1, 0);
    if(blocked_x1y1 || blocked_x2y1 || blocked_x1y2 || blocked_x2y2) {
        // can't move, just return
        return;
    }

    int got_x1y1 = 0;
    int got_x1y2 = 0;
    int got_x2y1 = 0;
    int got_x2y2 = 0;
    float x1y1_pos_z = pos_z;
    float x2y1_pos_z = pos_z;
    float x1y2_pos_z = pos_z; // either above the player by less than knee height, or some point below the player
    float x2y2_pos_z = pos_z;
    // find if there's a height up to knee height where each of these is solid
    // if not then just put it at current height i guess

    for(int kh = 0; kh <= knee_height; kh++) {
        int go_up_height = (pos_z-kh)+eye_height;
        if(!got_x1y1 && !check_for_solid_voxel_in_aabb(floor_new_pos_x1, floor_new_pos_y1, go_up_height, 1, 1, 0)) {
            x1y1_pos_z = go_up_height - eye_height;
            got_x1y1 = 1;
        }
        if(!got_x2y1 && !check_for_solid_voxel_in_aabb(floor_new_pos_x2, floor_new_pos_y1, go_up_height, 1, 1, 0)) {
            x2y1_pos_z = go_up_height - eye_height;
            got_x2y1 = 1;
        }
        if(!got_x1y2 && !check_for_solid_voxel_in_aabb(floor_new_pos_x1, floor_new_pos_y2, go_up_height, 1, 1, 0)) {
            x1y2_pos_z = go_up_height - eye_height;
            got_x1y2 = 1;
        }
        if(!got_x2y2 && !check_for_solid_voxel_in_aabb(floor_new_pos_x2, floor_new_pos_y2, go_up_height, 1, 1, 0)) {
            x2y2_pos_z = go_up_height - eye_height;
            got_x2y2 = 1;
        }
    }

    f32 subX = new_pos_x - floor_new_pos_x1;
    f32 subY = new_pos_y - floor_new_pos_y1;
    pos_x = new_pos_x;
    pos_y = new_pos_y;
    // find lowest and highest points out of the four.
    pos_z = lerp(lerp(x1y1_pos_z, subX, x2y1_pos_z),
                    subY,
                    lerp(x1y2_pos_z, subX, x2y2_pos_z));

        
}

void handle_input(f32 dt) {
    if(!map_loaded) { return ;}
    int head_margin = 1;

    if(gravmode) {
        if((pos_z+eye_height) >= cur_map_max_height) {
            pos_z = cur_map_max_height-eye_height;
        }
        int collide_z = 0;
        f32 new_world_pos_z = pos_z+vel*dt;//15*dt;
        f32 contact_point = new_world_pos_z+eye_height;
        s32 floor_x = floor(pos_x);
        s32 floor_y = floor(pos_y);
        s32 next_x = floor_x+1;
        s32 next_y = floor_y+1;
        s32 solid_a = check_for_solid_voxel_in_aabb(pos_x, next_y, contact_point, 1, 1, 0);
        s32 solid_b = check_for_solid_voxel_in_aabb(next_x, next_y, contact_point, 1, 1, 0);
        s32 solid_c = check_for_solid_voxel_in_aabb(pos_x, next_y, contact_point, 1, 1, 0);
        s32 solid_d = check_for_solid_voxel_in_aabb(next_x, next_y, contact_point, 1, 1, 0);

        if(solid_a || solid_b || solid_c || solid_d) { 
        //if(check_for_solid_voxel_in_aabb(pos_x, pos_y, contact_point, 1, 1, 0)) {
            falling = 0;
            vel = 0.0f;
        } else {
            pos_z = contact_point-((f32)eye_height);
            falling = 1;
            vel = (vel+accel) > max_accel ? max_accel : vel+accel;
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

    //pitch_ang += dt*0.017*5 * lookupdown;




    
    if(strafe) {
        f32 contact_point = pos_z+eye_height;
        //int go_up_height = (pos_z-KNEE_HEIGHT)+EYE_HEIGHT;
        f32 new_pos_x = pos_x + dir_y * strafe * ((sprint+1) * 2) * dt * 5;
        f32 new_pos_y = pos_y + (dir_x * -1) * strafe * ((sprint+1) * 2) * dt * 5;
        
        //move_4dof(pos_x, pos_y, pos_z, new_pos_x, pos_y);
        //move_4dof(pos_x, pos_y, pos_z, new_pos_x, new_pos_y);

        if(gravmode && check_for_solid_voxel_in_aabb(new_pos_x, pos_y, contact_point, 1, 1, 0)) {
            for(int kh = 1; kh <= knee_height; kh++) {
                int go_up_height = (pos_z-kh)+eye_height;
                if(check_for_solid_voxel_in_aabb(new_pos_x, pos_y, go_up_height, 1, 1, 0)) {

                } else {
                    // go up a step
                    pos_x = new_pos_x;
                    pos_z = go_up_height-eye_height+0.999f;//go_up_height-EYE_HEIGHT; //ceilf(go_up_height-EYE_HEIGHT);
                    //go_up_height = go_up_height-1;
                    break;
                }
            }
        } else {
            pos_x = new_pos_x;
        }
        if(gravmode && check_for_solid_voxel_in_aabb(pos_x, new_pos_y, contact_point, 1, 1, 0)) {
            for(int kh = 1; kh <= knee_height; kh++) {
                int go_up_height = (pos_z-kh)+eye_height;
                if(check_for_solid_voxel_in_aabb(pos_x, new_pos_y, go_up_height, 1, 1, 0)) {

                } else {
                    // go up a step
                    pos_y = new_pos_y;
                    pos_z = go_up_height-eye_height+0.999f;//go_up_height-EYE_HEIGHT;//ceilf(go_up_height-EYE_HEIGHT);
                    break;
                }
            }
        } else {
            pos_y = new_pos_y;
        }
    }

    if(forwardbackward) {
        f32 contact_point = pos_z+eye_height;
        int go_up_height = (pos_z-knee_height)+eye_height;
        f32 new_pos_x = pos_x + dir_x * forwardbackward * ((sprint+1) * 2) * dt * 15;
        f32 new_pos_y = pos_y + dir_y * forwardbackward * ((sprint+1) * 2) * dt * 15;
        
        //move_4dof(pos_x, pos_y, pos_z, new_pos_x, pos_y);
        //move_4dof(pos_x, pos_y, pos_z, new_pos_x, new_pos_y);

        
        if(gravmode && check_for_solid_voxel_in_aabb(new_pos_x, pos_y, contact_point, 1, 1, 0)) {
            for(int kh = 1; kh <= knee_height; kh++) {
                int go_up_height = (pos_z-kh)+eye_height;
                if(check_for_solid_voxel_in_aabb(new_pos_x, pos_y, go_up_height, 1, 1, 0)) {

                } else {
                    // go up a step
                    pos_x = new_pos_x;
                    pos_z = go_up_height-eye_height+0.999f;//go_up_height-EYE_HEIGHT; //ceilf(go_up_height-EYE_HEIGHT);
                    go_up_height = go_up_height-1;
                    break;
                }
            }
        } else {
            pos_x = new_pos_x;
        }
        

        
        if(gravmode && check_for_solid_voxel_in_aabb(pos_x, new_pos_y, contact_point, 1, 1, 0)) {
            for(int kh = 1; kh <= knee_height; kh++) {
                int go_up_height = (pos_z-kh)+eye_height;
                if(check_for_solid_voxel_in_aabb(pos_x, new_pos_y, go_up_height, 1, 1, 0)) {

                } else {
                    // go up a step
                    pos_y = new_pos_y;
                    pos_z = go_up_height-eye_height+0.999f;//go_up_height-EYE_HEIGHT;//ceilf(go_up_height-EYE_HEIGHT);
                    break;
                }
            }
        } else {
            pos_y = new_pos_y;
        }
        
    }

    if(updown) {
        pos_z -= dt*updown*50;
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
int capt_mouse_x, capt_mouse_y;
void handle_left_mouse_down() {
    if(!mouse_captured) {
        capt_mouse_x = cur_mouse_x;
        capt_mouse_y = cur_mouse_y;
        printf("capturing mouse at %i,%i\n", capt_mouse_x, capt_mouse_y);
        mouse_captured = 1;
        rollleftright = 0.0f;
        baseroll = 0.0f;
        //int err = SDL_CaptureMouse(SDL_TRUE);
        SDL_SetRelativeMouseMode(SDL_TRUE);
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

#define MOUSE_SENSITIVITY 0.1f

s32 last_mouse_x, last_mouse_y;
void handle_mouse_input(f32 dt) {
    if(!mouse_captured) { 
        mouseroll = 0;
        lookupdown = 0;
        leftright = 0;
        return; 
    }

    f32 halfWidth = OUTPUT_WIDTH/2;
    f32 halfHeight = OUTPUT_HEIGHT/2;
    //s32 dx = abs(centerX - cur_mouse_x) < 150 ? 0 : (centerX - cur_mouse_x);
    //s32 dy = abs(centerY - cur_mouse_y) < 100 ? 0 : (centerY - cur_mouse_y);
    f32 dx = capt_mouse_x - cur_mouse_x;
    f32 dy = capt_mouse_y - cur_mouse_y;
    f32 y_mult = (dy >= 0 ? 1 : -1);
    dy *= y_mult;
    dy *= MOUSE_SENSITIVITY;

    // pitch angle should go down?
    dy = dy > halfHeight ? halfHeight : dy;
    f32 pct = dy / halfHeight;
    pitch_ang = y_mult * lerp(0.0f, pct, 1.5708f);

    //f32 x_mult = (dx >= 0 ? 1 : -1);
    //printf("dx %f\n", dx);
    dx -= halfWidth;
    //dx *= x_mult;
    dx *= MOUSE_SENSITIVITY;

    
    // 
    pct = dx / halfWidth; 

    //pct * 6.28;
    pct = fmod(pct, 1.0f);

    f32 ang = (pct *  6.28f)-3.1415f;//3.14159f;
    //f32 ang = lerp(-3.14f, pct, 3.14f);
    f32 old_dir_x = dir_x;
    dir_x = cos(ang); //dir_x * cos(ang) - dir_y * sin(ang);
    dir_y = sin(ang); //old_dir_x * sin(ang) + dir_y * cos(ang);
    f32 old_plane_x = plane_x;
    //plane_x = plane_x * cos(ang) - plane_y * sin(ang);
    plane_x = dir_y; //1.2f;//sin((desired_fov_degrees+ang)/2);
    plane_y = -dir_x;//1.2f;//cos((desired_fov_degrees+ang)/2);
    //tan((desired_fov_degrees+ang)/2)
    //plane_y = old_plane_x * sin(ang) + plane_y * cos(ang);
    


    /* 
    f32 lerp_term = (-dx/(OUTPUT_WIDTH/2.0));

    mouseroll = lerp(0, lerp_term*lerp_term, .25);
    if(cur_mouse_x != last_mouse_x) {
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
    }

    if(cur_mouse_y != last_mouse_y) {
        if(dy) {
            
            lookupdown += dy*cosf(baseroll)*.012;
            leftright += -dy*sinf(baseroll)*.004;

        } else {
            if(!dx) { lookupdown = 0; }
        }
    }
    last_mouse_x = cur_mouse_x;
    last_mouse_y = cur_mouse_y;
    */


}

void update_mouse_pos(int mouse_x, int mouse_y) {
    cur_mouse_x = mouse_x;
    cur_mouse_y = mouse_y;
}


thread_params light_chunk_parms[6] = {
    {.finished = 1, .working = 0}, {.finished = 1, .working = 0},
    {.finished = 1, .working = 0}, {.finished = 1, .working = 0},
    {.finished = 1, .working = 0}, {.finished = 1, .working = 0}
    };

job_pool light_pool_jobs = {
    .num_jobs = 6
};

void cleanup_threads() {
    wait_for_job_pool_to_finish(&light_pool_jobs);
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

//thread_pool_function(raycast_6dof_wrapper, arg_var)
//{
//	thread_params* tp = (thread_params*)arg_var;
//    raycast_6dof(tp->min_x, tp->min_y, tp->max_x, tp->max_y);
//	InterlockedIncrement64(&tp->finished);
//}

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
    s32 min_size = (s32) ceilf(sqrtf((render_width*render_width)+(render_width*render_width)));
    render_size = 2;
    while(render_size < min_size) {
        // if render size isn't big enough based on the render width
        // we need to reallocate the internal buffer
        render_size *= 2;
    }


    //pixels = realloc_wrapper(pixels, sizeof(u32)*OUTPUT_WIDTH*OUTPUT_HEIGHT+32);
    //while(((intptr_t)pixels)&0b11111) {
    //    pixels++;
    //}
    //}

    num_render_size_bits = LOG2(render_size);

    f32 scale = 1.0 / tanf(deg_to_rad(desired_fov_degrees)/2.0);

    //scale_height = ((((16/9)*(.5))/(4/3))*render_size);
    scale_height = ((render_height) * scale * 1.3 * max(1, aspect_ratio));

    //base_inter_buffer = realloc_wrapper(base_inter_buffer, (sizeof(u32)*render_size*render_size)+32);
    base_world_pos_buffer = realloc_wrapper(base_world_pos_buffer, (sizeof(u32)*render_size*render_size)+32, "world_pos buffer");
    base_norm_buffer = realloc_wrapper(base_norm_buffer, (sizeof(f32)*2*render_size*render_size)+32, "normal buffer");
    base_occlusion_buffer = realloc_wrapper(base_occlusion_buffer, (sizeof(u8)*render_size*render_size/8)+32, "occlusion buffer");
    base_albedo_buffer = realloc_wrapper(base_albedo_buffer, (sizeof(u32)*render_size*render_size)+32, "albedo buffer");
    //base_premult_norm_buffer = realloc_wrapper(base_premult_norm_buffer, (sizeof(char)*render_size*render_size)+32);


    
    //inter_buffer = base_inter_buffer;
    world_pos_buffer = base_world_pos_buffer;
    norm_buffer = base_norm_buffer;
    //premult_norm_buffer = base_premult_norm_buffer;
    occlusion_buffer = base_occlusion_buffer;
    albedo_buffer = base_albedo_buffer;

    // alignment shenanigans for avx2
    //while(((intptr_t)inter_buffer)&0b11111) {
    //    inter_buffer++;
    //}
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
    //while(((intptr_t)premult_norm_buffer)&0b11111) {
    //    premult_norm_buffer++;
    //}

    setup_render_size = 1;


}



__m256 srgb255_to_linear_256(__m256 cols) {

    __m256 one_over_255_vec = _mm256_set1_ps(1.0f / 255.0f);
    __m256 div_255 = _mm256_mul_ps(cols, one_over_255_vec);
    return _mm256_mul_ps(div_255, div_255);
}

f32 srgb255_to_linear(f32 col) {

    f32 one_over_255 = (1.0f / 255.0f);
    f32 div_255 = (col*one_over_255);
    return (div_255*div_255);
}

__m256i linear_to_srgb_255_256(__m256 cols) {

    return _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_set1_ps(255.0f), _mm256_sqrt_ps(cols)));
}

f32 linear_to_srgb_255(f32 col) {
    return 255.0 * sqrtf(col);
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

#define CLAMP(a, mi, ma) max(mi, min(a, ma))

typedef enum {
    CLIP_OUT,
    CLIP_IN
} state;
 
//#include "raycast_6dof.c"

Olivec_Canvas vc_render(double dt) {
    handle_mouse_input(dt);
    //prev_pos_z = pos_z;
    handle_input(dt);
    if(fabs(cam_pos_z - pos_z) < 0.1f) {
        cam_pos_z = pos_z;
    } else {
        cam_pos_z += (pos_z-cam_pos_z)*(min(dt,.1f)/0.1f);
    }


    static int setup_thread_pool = 0;
    if(!setup_thread_pool) {
        render_pool = thread_pool_create(NUM_RENDER_THREADS);
        map_pool = thread_pool_create(NUM_LIGHT_MAP_THREADS);

        setup_thread_pool = 1;
    }





    pixels = vc_sdl_request_pixel_buffer(render_width, render_height);

    if(!setup_render_size) {
        printf("setting up render buffer\n");
        setup_internal_render_buffers();
    }

    if(!map_loaded) {
        load_map(cur_map++);
        map_loaded = 1;
        init_player_positions();
    }

    // working 0 -> not started
    // working 1, finished 0 -> started, not finished
    // working 1, finished 1 -> started, finished

    int gathered_jobs = 0;
    for(int i = 0; i < NUM_LIGHT_MAP_THREADS; i++) {
        if(light_chunk_parms[i].finished == 0) {
            continue;
        }
        //printf("thread %i is free!\n", i);
        // we have a free slot
        // check if any chunks need to be lit
        int per_axis = light_map_size / LIGHT_CHUNK_SIZE;
        int chunk_to_light = find_closest_unlit_chunk(pos_x, pos_y);
        if(chunk_to_light == -1) { break; }
        //printf("lighting chunk %i,%i\n", LIGHT_CHUNK_SIZE*(chunk_to_light%per_axis),
        //                                LIGHT_CHUNK_SIZE*(chunk_to_light/per_axis));
        light_chunk_parms[i].finished = 0;
        light_chunk_parms[i].min_x = light_map_chunks[chunk_to_light].min_x;
        light_chunk_parms[i].min_y = light_map_chunks[chunk_to_light].min_y;
        light_chunk_parms[i].max_x = light_map_chunks[chunk_to_light].min_x+LIGHT_CHUNK_SIZE;
        light_chunk_parms[i].max_y = light_map_chunks[chunk_to_light].min_y+LIGHT_CHUNK_SIZE;
        gathered_jobs++;
        light_map_chunks[chunk_to_light].lit = 1;
    }
    if(gathered_jobs) {
        //printf("lighting %i chunks\n", gathered_jobs);
        light_pool_jobs.func = light_map_wrapper;
        light_pool_jobs.raw_func = light_map;
        light_pool_jobs.parms = light_chunk_parms;
                    
        start_pool_new_jobs_always_threaded(map_pool, &light_pool_jobs);
    }

    if(!plane_parameters_setup) {
        setup_ray_plane_parameters();
    }



    Olivec_Canvas oc = olivec_canvas(pixels, render_width, render_height, render_width, 1<<double_pixels);

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

    //memset(columns_bitmaps_data, 0, sizeof(column_bitmaps)*1024*1024);
    

    double start_clear = GetTicks();
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
        job_pool rp = {
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
    double end_clear = GetTicks();

    double start_raycast = GetTicks();
    {
        thread_params parms[NUM_RENDER_THREADS];

        for(int i = 0; i < NUM_RENDER_THREADS; i++) {
            parms[i].finished = 0;
            parms[i].min_x = (i == 0) ? min_x : parms[i-1].max_x;
            parms[i].max_x = (i == NUM_RENDER_THREADS-1) ? max_x : (parms[i].min_x + draw_dx/NUM_RENDER_THREADS);
            parms[i].min_y = min_y;
            parms[i].max_y = max_y;
        }
        job_pool rp = {
            .num_jobs = NUM_RENDER_THREADS,
            .parms = parms
        };
            rp.func = render_6dof ? raycast_scalar_wrapper : raycast_scalar_wrapper,
            rp.raw_func = render_6dof ? raycast_scalar : raycast_scalar,
        
        //raycast_scalar(min_x, min_y, max_x, max_y);
        start_pool(render_pool, &rp);
        wait_for_job_pool_to_finish(&rp);
    }
    double end_raycast = GetTicks();


    /*
        SCREEN SPACE QUADRANTS FOR 6DOF VOXLAP STYLE RENDERING

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

    */

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
        job_pool rp = {
            .num_jobs = 8,
            .func = fill_empty_entries_wrapper,
            .raw_func = fill_empty_entries,
            .parms = parms
        };
        start_pool(pool, &rp);
        wait_for_render_pool_to_finish(&rp);
    #endif
    }
    double start_light = GetTicks();

    {
        thread_params parms[NUM_RENDER_THREADS];
        
        if(double_pixels == 4) {
            // the left edge of the screen breaks in this pass with 8 jobs, prob not divisible by 8
            // honestly, it runs so much faster that I sort of don't care about using 8 threads here
            for(int i = 0; i < 2; i++) {
                parms[i].finished = 0;
                parms[i].min_x = (render_width*i/2);
                parms[i].max_x = parms[i].min_x + (render_width/2);
                parms[i].min_y = 0;
                parms[i].max_y = render_height;
            }
            job_pool rp = {
                .num_jobs = 2,
                .func = rotate_light_and_blend_wrapper,
                .raw_func = rotate_light_and_blend,
                .parms = parms
            };
            start_pool(render_pool, &rp);
            wait_for_job_pool_to_finish(&rp);
        } else if(double_pixels == 2) {
            // the left edge of the screen breaks in this pass with 8 jobs, prob not divisible by 8
            // honestly, it runs so much faster that I sort of don't care about using 8 threads here
            for(int i = 0; i < 4; i++) {
                parms[i].finished = 0;
                parms[i].min_x = (render_width*i/4);
                parms[i].max_x = parms[i].min_x + (render_width/4);
                parms[i].min_y = 0;
                parms[i].max_y = render_height;
            }
            job_pool rp = {
                .num_jobs = 4,
                .func = rotate_light_and_blend_wrapper,
                .raw_func = rotate_light_and_blend,
                .parms = parms
            };
            start_pool(render_pool, &rp);
            wait_for_job_pool_to_finish(&rp);
        } else {
            for(int i = 0; i < NUM_RENDER_THREADS; i++) {
                parms[i].finished = 0;
                parms[i].min_x = (render_width*i/NUM_RENDER_THREADS);
                parms[i].max_x = parms[i].min_x + (render_width/NUM_RENDER_THREADS);
                parms[i].min_y = 0;
                parms[i].max_y = render_height;
            }
            job_pool rp = {
                .num_jobs = NUM_RENDER_THREADS,
                .func = rotate_light_and_blend_wrapper,
                .raw_func = rotate_light_and_blend,
                .parms = parms
            };
            start_pool(render_pool, &rp);
            wait_for_job_pool_to_finish(&rp);
        }   
    }
    vc_sdl_release_pixel_buffer();

    double end_light = GetTicks();

    int mouse_off_x = (cur_mouse_x - (OUTPUT_WIDTH/2))/(1<<double_pixels);
    int mouse_off_y = (cur_mouse_y - (OUTPUT_HEIGHT/2))/(1<<double_pixels);


    int center_plus_mouse_off_x = ((render_size/2)+mouse_off_x);
    int center_plus_mouse_off_y = ((render_size/2)+mouse_off_y);
    int clamp_x = CLAMP(center_plus_mouse_off_x, 0, render_size-1);
    int clamp_y = CLAMP(center_plus_mouse_off_y, 0, render_size-1);

    u32 picked_fb_idx;
    if(mouse_captured) {
        picked_fb_idx = fb_swizzle(render_size/2, render_size/2);
    } else {
        picked_fb_idx = fb_swizzle(clamp_x, clamp_y);
    }
    u32 combined_world_idx = world_pos_buffer[picked_fb_idx];


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
        if(1) { //last_frame_middle_mouse_down == 0) {
            last_frame_middle_mouse_down = 1;
            for(int i = 0; i < 32; i++) { //128; i++) {
                float rx = (rand()-(RAND_MAX/2)) / ((double)RAND_MAX);
                float ry = (rand()-(RAND_MAX/2)) / ((double)RAND_MAX);
                float rz = (rand()-(RAND_MAX/2)) / ((double)RAND_MAX);
                float scale = 10.0f * (rand()/((double)RAND_MAX)); // 100.0f;

                float norm_mul = scale/sqrtf(rx*rx+ry*ry+rz*rz);
                rx *= norm_mul;
                ry *= norm_mul;
                rz *= norm_mul;
                s32 map_x = rx+screen_center_map_x;
                s32 map_y = ry+screen_center_map_y;
                s32 map_z = rz+screen_center_map_z;
                remove_voxel_at(map_x,
                                map_y,
                                map_z);
                int chunk_x = (map_x / LIGHT_CHUNK_SIZE);
                int chunk_y = (map_y / LIGHT_CHUNK_SIZE);
                int chunks_per_axis = light_map_size / LIGHT_CHUNK_SIZE;
                light_map_chunks[(chunk_y*chunks_per_axis)+chunk_x].lit = 0;
            }


            //remove_voxel_at(screen_center_map_x, screen_center_map_y, screen_center_map_z);
        }
    } else {
        last_frame_middle_mouse_down = 0;
    }
    if (right_mouse_down) {
            u8 prev_ao = (*(color_ptr+color_slot_in_column))>>24;
        *(color_ptr+color_slot_in_column) = (prev_ao<<24)|0xB469FF; 
    }
    pixel_is_undrawn:;

    
    f32 ms = dt*1000;
    f32 fps = 1000/ms;

    
    total_time += ms;
    frame++;

    sun_ang += dt*0.13;
    if(lighting == NORMALS_LIGHTING) { //} || lighting == FACE_LIGHTING) {
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
    } else if (lighting == NORMALS_LIGHTING) { //} || lighting == FACE_LIGHTING) {
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


    if (debug_print) {

        #define DEBUG_PRINT_TEXT(fmt_str, ...) do { \
            sprintf(buf, (fmt_str), __VA_ARGS__);   \
            olivec_text_no_blend(oc, buf, 5, y, olivec_default_font, double_pixels?1:2, 0xFF0000FF);\
            y += double_pixels ? 7 : 13;                                \
        } while(0);

        char buf[128];
        int y = 3;
        
        DEBUG_PRINT_TEXT("lighting:       %s", lighting_mode_strs[lighting]);
        DEBUG_PRINT_TEXT("amb. occlusion: %s", ambient_occlusion ? "enabled" : "disabled");
        DEBUG_PRINT_TEXT("fog:            %s", fogmode ? "enabled" : "disabled");
        DEBUG_PRINT_TEXT("transparency:   %s", transparency ? "enabled" : "disabled");
        //DEBUG_PRINT_TEXT("render mode:    %s", render_6dof ? "6DOF" : "5DOF");
        DEBUG_PRINT_TEXT("view mode:      %s", view_mode_strs[view]);
        DEBUG_PRINT_TEXT("render resolution: %ix%i", max_x-min_x, (max_y+1)-min_y);
        DEBUG_PRINT_TEXT("output resolution: %ix%i", OUTPUT_WIDTH, OUTPUT_HEIGHT);
        //DEBUG_PRINT_TEXT("voxels drawn: %llu", count_set_bits_in_voxelmap());
        DEBUG_PRINT_TEXT("clear %.2fms", (end_clear-start_clear)*1000.0f);
        DEBUG_PRINT_TEXT("raycast %.2fms", (end_raycast-start_raycast)*1000.0f);
        DEBUG_PRINT_TEXT("light %.2fms", (end_light-start_light)*1000.0f);
        DEBUG_PRINT_TEXT("frametime: %.2fms", (dt*1000.0f));
        DEBUG_PRINT_TEXT("fps: %.2f", (1000.0f / (dt*1000.0f)));
        DEBUG_PRINT_TEXT("%f,%f,%f", pos_x, pos_y, pos_z+eye_height);

    }
    
    vc_sdl_release_pixel_buffer();
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

    __m256 height_vec = _mm256_set1_ps(255.0f-cam_pos_z);

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
    //__m256 inverse_height_vec = _mm256_set1_ps(pos_z);
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

    #if 1
        for(int base_ox = min_x; base_ox < max_x; base_ox += 8) {
            //f32 yy = row_y;
            //f32 xx = row_x;
            f32 yy = col_y;
            f32 xx = col_x;
            //if(oy > (render_height*.75)) { 
            //    printf("wtf\n");
            //}
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
                // undrawn = world_space_poses == undrawn_bit_mask_vec, which is 0b10000000
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
                if(0) { //double_pixels == 2) {
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
                

                } else if (0) { //double_pixels == 1) { 
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

            
            col_x += dx_per_x*8;
            col_y += dy_per_x*8;
            //row_x += dx_per_y;
            //row_y += dy_per_y;
        }
    #else
        // scalar version of rotation
        // no support for lighting or fog
        for(int oy = min_y; oy < max_y; oy++) {
            int dy = (oy-min_y);
            int off_y = dy * dy_per_y;

            for(int ox = min_x; ox < max_x; ox++) { 
                int dx = (ox - min_x);
                int off_x = dx * dx_per_x;

                s32 iyy = col_y + y_buffer + off_y; //(oy-min_y)*dy_per_y;
                s32 ixx = col_x + x_buffer + off_x;
                //yy += dy_per_y;
                //xx += dx_per_y;
                u32 fb_idx = fb_swizzle(ixx, iyy);
                //g_buf_entry entry = g_buffer[fb_idx];

                //f32 z = (1/z_buffer[fb_idx])*scale_height;   //1/10 240
                //u8 ch = lerp(0, z/max_z, 255);
                //u32 pix = ((0xFF << 24) | (ch << 16) | (ch << 8) | ch);
                //u32 pix = inter_buffer[fb_idx];
                u32 pix = albedo_buffer[fb_idx];// entry.albedo;
                //u8 r = (short_pix&0b11111)<<3;
                //u8 g = ((short_pix>>5)&0b11111)<<3;
                //u8 b = ((short_pix>>10)&0b11111)<<3;
                //u32 pix = (0xFF << 24) | (b << 16) | (g << 8) | r;
                
                u32 world_space_pos = world_pos_buffer[fb_idx];
                u32 x = (world_space_pos>>19)&0b1111111111;
                u32 y = (world_space_pos>>8)&0b11111111111;
                u32 z = (world_space_pos & 0xFF);
                

                f32 world_dx = pos_x-x;
                f32 world_dy = pos_y-y;
                f32 float_depth = sqrtf(world_dx*world_dx+world_dy*world_dy);

                if(0) { //double_pixels) {
                    int scale_size = (1 << double_pixels); //== 1 ? 2 : 4);
                    for(int iy = 0; iy < scale_size; iy++) {
                        for(int ix = 0; ix < scale_size; ix++) {
                            pixels[((oy*scale_size)+iy)*OUTPUT_WIDTH+(ox*scale_size)+ix] = pix;
                        }
                    }
                } else {
                    u32 undrawn = world_space_pos == 0b10000000;
                    u32 depth_eq_max = undrawn;
                        
                    if(0) { //fogmode) {
                        u32 albedo = albedo_buffer[fb_idx];

                        // aaaaabbbbbgggggrrrrr
                        //             rrrrr000
                        //             ggggg000
                        u32 albedo_r = albedo & 0xFF;
                        u32 albedo_g = (albedo >> 8) & 0xFF;
                        u32 albedo_b = (albedo >> 16) & 0xFF;
                        u32 albedo_coverage = (albedo >> 24)&0b11;
                        f32 one_minus_old_coverage = (0b11-albedo_coverage)/3.0f;


                        f32 float_r = srgb255_to_linear(albedo_r);
                        f32 float_g  = srgb255_to_linear(albedo_g);
                        f32 float_b = srgb255_to_linear(albedo_b);

                        f32 linear_background_color_r = srgb255_to_linear(background_color&0xFF);
                        f32 linear_background_color_g = srgb255_to_linear((background_color>>8)&0xFF);
                        f32 linear_background_color_b = srgb255_to_linear((background_color>>16)&0xFF);
                        float_r = min(1.0f, float_r + linear_background_color_r*one_minus_old_coverage);
                        float_g = min(1.0f, float_g + linear_background_color_g*one_minus_old_coverage);
                        float_b = min(1.0f, float_b + linear_background_color_b*one_minus_old_coverage);
                
                        //f32 z_to_1 = (avg_dist*one_over_max_z)*(avg_dist*one_over_max_z);
                        f32 one_over_max_z = 1.0f/max_z;

                        f32 z_to_one = min(1.0f, float_depth*one_over_max_z);
                        z_to_one = z_to_one*z_to_one;
                        z_to_one = z_to_one*z_to_one;
                        f32 fog_factor = lerp(0.0f, z_to_one, 1.0f);
                        f32 one_minus_fog_factors = 1.0f-fog_factor;
                        //f32 fog_factor = lerp(0, z_to_1, 1);
                        //one_minus_fog = 1-fog_factor;
                        f32 mult_fog_r = fog_factor*fog_r;
                        f32 mult_fog_g = fog_factor*fog_g;
                        f32 mult_fog_b = fog_factor*fog_b;
                        f32 mult_r = one_minus_fog_factors*float_r;
                        f32 mult_g = one_minus_fog_factors*float_g;
                        f32 mult_b = one_minus_fog_factors*float_b;
                        float_r = mult_fog_r*mult_r;
                        float_g = mult_fog_g*mult_g;
                        float_b = mult_fog_b*mult_b;
                        
                        u32 int_r = linear_to_srgb_255(float_r); // _mm256_cvtps_epi32(float_rs);
                        u32 int_g = linear_to_srgb_255(float_g); //_mm256_cvtps_epi32(float_gs);
                        u32 int_b = linear_to_srgb_255(float_b); //_mm256_cvtps_epi32(float_bs);

                        u32 fogged_pix = (0xFF<<24)|(int_b<<16)|(int_g<<8)|int_r;
                        pixels[oy*render_width+(ox)] = depth_eq_max ? background_color : fogged_pix;
                    } else {
                        pixels[oy*render_width+(ox)] = depth_eq_max ? background_color : pix;
                    }
                }
            }
        }
    #endif


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



s32 world_to_screen(s32 world_pos, f32 height, f32 invz, f32 next_invz, f32 pitch) {
    // invz is really scale_height / z
    // so multiplying by it is really (something * scale_height / z)
    f32 relative_height =  height - (255 - world_pos); //[voxelmap_idx];
    f32 selected_invz = (relative_height < 0 ? invz : next_invz);
    f32 float_projected_height = relative_height * selected_invz;
    s32 int_projected_height = floor(float_projected_height) + pitch;
    return int_projected_height;
}


f32 screen_to_world(s32 int_projected_height, f32 z, f32 pitch) {
    f32 height = 255.0f - cam_pos_z;
    f32 float_projected_height = int_projected_height - pitch;
    f32 relative_height = float_projected_height * z / scale_height;

    f32 world_pos = 255 - (pitch * z / scale_height) - height;
    //return world_pos;
}

void raycast_skipped_block_scalar(
    int min_x, int min_y, int max_x, int max_y, 
    __m256i rem_x_steps_vec, __m256i rem_y_steps_vec,
    __m256 perp_wall_dists_vec, __m256 next_perp_wall_dists_vec,
    __m256 side_dists_x_vec, __m256 side_dists_y_vec,
    __m256i map_x_vec, __m256i map_y_vec,
    __m256i step_x_vec, __m256i step_y_vec,
    __m256 delta_dist_x_vec, __m256 delta_dist_y_vec,
    __m256 invz_vec, __m256 next_invz_vec
);


void raycast_scalar(s32 min_x, s32 min_y, s32 max_x, s32 max_y) {
    // allocate on stack for this hack lol :)
    #ifdef CALC_COLUMN_STATS 
    int skipped_cells_per_column[1920]; // until onscreen
    int z_steps_per_column[1920];
    for(int i = 0; i < max_x-min_x; i++) {
        skipped_cells_per_column[i] = -1;
    }
    #endif

    f32 fog_r = (background_color&0xFF);
    f32 fog_g = ((background_color>>8)&0xFF);
    f32 fog_b = ((background_color>>16)&0xFF);

    u8 looking_up = (pitch_ang > 0.0);

    float2 encoded_top_face_norm = encode_norm(0, -1, 0);
    float2 encoded_bot_face_norm = encode_norm(0, 1, 0);
    float2 encoded_x_side_norm = encode_norm( (dir_x > 0 ? 1 : -1), 0, 0);
    float2 encoded_y_side_norm = encode_norm(0, 0, (dir_y > 0 ? 1 : -1));
    
    
    f32 pitch = pitch_ang_to_pitch(pitch_ang);

    profile_block raycast_scalar_block;
    TimeBlock(raycast_scalar_block, "raycast scalar");

    for(int x = min_x; x < max_x; x++) {
        int skipped_cells_for_this_column = 0;
        int got_possibly_onscreen_column = 0;
        int z_steps_for_this_column = 0;

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

            side_dist_x += side_dist_dx;
            side_dist_y += side_dist_dy;

            map_x += map_x_dx;
            map_y += map_y_dy;
            skipped_cells_for_this_column++;
            z_steps_for_this_column++;
        }

        f32 invz = scale_height / perp_wall_dist;
        f32 next_invz = scale_height / next_perp_wall_dist;

        profile_block z_step_block;

        s32 max_x_steps = (ray_dir_x > 0 ? (1023-(s32)pos_x) : (((s32)pos_x))) + 1;
        s32 max_y_steps = (ray_dir_y > 0 ? (1023-(s32)pos_y) : (((s32)pos_y))) + 1;


        while(perp_wall_dist <= max_z && prev_drawn_max_y > next_drawable_min_y && max_x_steps > 0 && max_y_steps > 0) {


            u32 prepared_combined_world_map_pos = ((map_x)<<19)|((map_y)<<8); 
            
            int map_x_dx = (side_dist_x < side_dist_y) ? step_x : 0;
            int map_y_dy = (side_dist_x < side_dist_y) ? 0 : step_y;

            perp_wall_dist = next_perp_wall_dist;
            next_perp_wall_dist = (side_dist_x < side_dist_y) ? side_dist_x : side_dist_y;


            f32 side_dist_dx = (side_dist_x < side_dist_y) ? delta_dist_x : 0;
            f32 side_dist_dy = (side_dist_x < side_dist_y) ? 0 : delta_dist_y;
            u32 d_x_steps = (side_dist_x < side_dist_y) ? 1 : 0;
            u32 d_y_steps = (side_dist_x < side_dist_y) ? 0 : 1;
            max_x_steps -= d_x_steps;
            max_y_steps -= d_y_steps;




            side_dist_x += side_dist_dx;
            side_dist_y += side_dist_dy;

            s32 next_map_x = map_x + map_x_dx;
            s32 next_map_y = map_y + map_y_dy;


            if(map_x < 0 || map_x > 1023 || map_y < 0 || map_y > 1023) {
                skipped_cells_for_this_column += (got_possibly_onscreen_column == 0 ? 1 : 0);
                goto next_z_step;
            }

            
            next_invz = scale_height / next_perp_wall_dist;
            u32 voxelmap_idx = get_voxelmap_idx(map_x, map_y);


            //int mip = invz >= 1.0 ? 0 : ceilf(1.0/invz);
            //mip = min(1, mip);
            //u32 mip_map_x = (map_x >> mip);
            //u32 mip_map_y = (map_y >> mip);
            //u32 mip_voxelmap_idx = get_voxelmap_idx(mip_map_x, mip_map_y);
            
            column_header* header = &columns_header_data[voxelmap_idx];

            if(header->num_runs == 0) { 
                skipped_cells_for_this_column += (got_possibly_onscreen_column == 0 ? 1 : 0);
                goto next_z_step; 
            }

            // check the top of this column against the bottom of the frustum skip drawing it
            f32 height = 255.0f - cam_pos_z;
            s32 int_top_of_col_projected_height = world_to_screen(header->top_y, height, invz, next_invz, pitch);
            f32 reverse_projected = screen_to_world(int_top_of_col_projected_height, 1.0f/invz, pitch);
            f32 next_reverse_projected = screen_to_world(int_top_of_col_projected_height, 1.0f/next_invz, pitch);

            span* span_info = columns_runs_data[voxelmap_idx].runs_info;
            
            if(int_top_of_col_projected_height >= prev_drawn_max_y) {
                skipped_cells_for_this_column += (got_possibly_onscreen_column == 0 ? 1 : 0);
                goto next_z_step;
            }
            u8 num_runs = header->num_runs;

            s32 int_bot_of_col_projected_height = world_to_screen(cur_map_max_height, height, invz, next_invz, pitch);
            if(int_bot_of_col_projected_height <= next_drawable_min_y) {
                skipped_cells_for_this_column += (got_possibly_onscreen_column == 0 ? 1 : 0);
                goto next_z_step;
            }
            got_possibly_onscreen_column = 1;

// draw top or bottom face


        //f32 new_norm_pt1 = occlusion_bit ? old_norm_pt1 : (face_norm_1);                    
        //f32 new_norm_pt2 = occlusion_bit ? old_norm_pt2 : (face_norm_2);                    


        //set_bit_in_bitmap(map_x, map_y, map_z, occlusion_bit ? 0 : 1);                      

#define DRAW_CHUNK_FACE(top, bot, voxel_color_idx) {              \
    u32 fb_idx = fb_swizzle(x,top);                                                         \
    u32 color = top_of_col_color_ptr[voxel_color_idx];                                      \
    f32 norm_pt1 = top_of_col_norm_pt1_ptr[voxel_color_idx];                                \
    f32 norm_pt2 = top_of_col_norm_pt2_ptr[voxel_color_idx];                                \
    for(int y = top; y < bot; y++) {                                                        \
        u8 occlusion_bit = get_occlusion_bit(x, y);                                         \
        s32 map_z = voxel_color_idx;                                                        \
        u32 combined_world_pos = prepared_combined_world_map_pos|map_z;                     \
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

        //set_bit_in_bitmap(map_x, map_y, map_z, occlusion_bit ? 0 : 1);

// draw side of chunk
// needs voxel color interpolation
#define DRAW_CHUNK_SIDE(top, bot) {                           \
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
        s32 map_z = ((voxel_color_idx+color_ptr)-top_of_col_color_ptr);                     \
        u32 combined_world_pos = prepared_combined_world_map_pos|map_z;                     \
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




#define DRAW_CHUNK(chunk_top, chunk_bot, is_top_chunk, is_transparent_chunk, break_if_top_below_screen, break_if_bot_above_screen) {         \
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
            DRAW_CHUNK_FACE(clipped_top_face_height, clipped_top_side_height, color_offset); \
        }                                                                                \
        if(clipped_top_side_height < clipped_bot_side_height) {                          \
            DRAW_CHUNK_SIDE(clipped_top_side_height, clipped_bot_side_height); \
        }                                                                                \
        if(clipped_bot_side_height < clipped_bot_face_height) {       \
            u16 color_offset = ((color_ptr - top_of_col_color_ptr)+num_voxels)-1;        \
            DRAW_CHUNK_FACE(clipped_bot_side_height, clipped_bot_face_height, color_offset); \
        }                                                                                \
        s32 next_prev_drawn_max_y = (min_coverage < 0b11) ? prev_drawn_max_y : (bot_projected_heightonscreen >= prev_drawn_max_y && top_projected_heightonscreen < prev_drawn_max_y) ? top_projected_heightonscreen : prev_drawn_max_y;             \
        s32 next_next_drawable_min_y = (min_coverage < 0b11) ? next_drawable_min_y :  (top_projected_heightonscreen <= next_drawable_min_y && bot_projected_heightonscreen > next_drawable_min_y) ? bot_projected_heightonscreen : next_drawable_min_y; \
        prev_drawn_max_y = next_prev_drawn_max_y; \
        next_drawable_min_y = next_next_drawable_min_y; \
    }   \
}           


    
#define DRAW_CHUNK_BOTTOM_UP(chunk_top, chunk_bot, is_top_chunk, is_transparent_chunk, break_if_top_below_screen, break_if_bot_above_screen) {         \
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
                                                                                        \
        if(clipped_bot_side_height < clipped_bot_face_height) {                         \
            u16 color_offset = ((color_ptr - top_of_col_color_ptr)+num_voxels)-1;        \
            DRAW_CHUNK_FACE(clipped_bot_side_height, clipped_bot_face_height, color_offset); \
        }                                                                                \
        if(clipped_top_side_height < clipped_bot_side_height) {                          \
            DRAW_CHUNK_SIDE(clipped_top_side_height, clipped_bot_side_height); \
        }                                                                                   \
        if(is_top_chunk && clipped_top_face_height < clipped_top_side_height) { \
            u16 color_offset = (color_ptr - top_of_col_color_ptr);                       \
            DRAW_CHUNK_FACE(clipped_top_face_height, clipped_top_side_height, color_offset); \
        }                                                                        \
        s32 next_prev_drawn_max_y = (min_coverage < 0b11) ? prev_drawn_max_y : (bot_projected_heightonscreen >= prev_drawn_max_y && top_projected_heightonscreen < prev_drawn_max_y) ? top_projected_heightonscreen : prev_drawn_max_y;             \
        s32 next_next_drawable_min_y = (min_coverage < 0b11) ? next_drawable_min_y :  (top_projected_heightonscreen <= next_drawable_min_y && bot_projected_heightonscreen > next_drawable_min_y) ? bot_projected_heightonscreen : next_drawable_min_y; \
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
                        DRAW_CHUNK_BOTTOM_UP(
                            bot_surface_top, bot_surface_end, 
                            0, //span_info[run].is_top, 
                            ((color_ptr[0]>>24)&0b11) != 0b11,//0, //span_info[run].is_transparent, 
                            0, 1
                        );
                    }

                    int top_surface_top = 255 - span_info[run].top_surface_start;
                    int top_surface_bot = 255 - (span_info[run].top_surface_end+1);
                    color_ptr -= ((span_info[run].top_surface_end+1)-span_info[run].top_surface_start);
                    DRAW_CHUNK_BOTTOM_UP(
                        top_surface_top, top_surface_bot, 
                        1, //span_info[run].is_top, 
                        ((color_ptr[0]>>24)&0b11) != 0b11, //0, //span_info[run].is_transparent, 
                        0, 1
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
                        1, 0
                    );
                    color_ptr += ((span_info[run].top_surface_end+1)-span_info[run].top_surface_start);
                    if(span_info[run].bot_surface_end > span_info[run].bot_surface_start) {
                        DRAW_CHUNK(
                            bot_surface_top, bot_surface_bot, 
                            0, //span_info[run].is_top, 
                            ((color_ptr[0]>>24)&0b11) != 0b11, //0, //span_info[run].is_transparent, 
                            1, 0
                        );
                        color_ptr += (span_info[run].bot_surface_end-span_info[run].bot_surface_start);
                    }
                }

            }

        next_z_step:;
        map_x = next_map_x;
        map_y = next_map_y;
        invz = next_invz;
        z_steps_for_this_column++;
        }
        #ifdef CALC_COLUMN_STATS 
        skipped_cells_per_column[x-min_x] = skipped_cells_for_this_column;
        z_steps_per_column[x-min_x] = z_steps_for_this_column;
        #endif
    }
    EndTimeBlock(raycast_scalar_block);

    #ifdef CALC_COLUMN_STATS 
    {
        u64 total_skipped_cells = 0;
        for(int i = 0; i < max_x-min_x; i++) {
            total_skipped_cells += skipped_cells_per_column[i];
        }

        double total_waste = 0.0f;
        for(int i = 0; i < max_x-min_x; i++) {
            double waste = skipped_cells_per_column[i] / ((double) z_steps_per_column[i]);
            total_waste += waste;
        }
        printf("avg skipped cells per column: %f\n", ((double)total_skipped_cells)/(max_x-min_x));
        printf("avg wasted steps: %f\n", ((double)total_waste)/(max_x-min_x));
    }
    #endif
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


