
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>





//1440
#define BACKGROUND_COLOR 0xFFDBCE87

#define ArrayCount(Array) (sizeof(Array) / sizeof((Array)[0]))

typedef uint64_t u64;
typedef int64_t s64;
typedef uint32_t u32;
typedef int32_t s32;
typedef uint16_t u16;
typedef int16_t s16;
typedef uint8_t u8;
typedef int8_t s8;

typedef double f64;
typedef float f32;

#include "depth.c"
#include "color.c"

#define PROFILER 1
#include "selectable_profiler.c"
#include <SDL2/SDL.h>

u32 render_size;


#define WIDTH (1920/2)
#define HEIGHT (1080/2)
#define DOUBLE

	// HELPER METHOD for Magic bits encoding - split by 2
u32 morton2D_SplitBy2Bits(const u32 a) {
    const u32 magicbit2D_masks32[6] = { 0xFFFFFFFF, 0x0000FFFF, 0x00FF00FF, 0x0F0F0F0F, 0x33333333, 0x55555555 };

    u32 x = a;
    x = (x | x << 16) & magicbit2D_masks32[1];
    x = (x | x << 8) & magicbit2D_masks32[2];
    x = (x | x << 4) & magicbit2D_masks32[3];
    x = (x | x << 2) & magicbit2D_masks32[4];
    x = (x | x << 1) & magicbit2D_masks32[5];
    return x;
}

	// ENCODE 2D Morton code : Magic bits
u32 m2D_e_magicbits(const u32 x, const u32 y) {
    return morton2D_SplitBy2Bits(x) | (morton2D_SplitBy2Bits(y) << 1);
}


static u32 get_map_idx(s32 x, s32 y) {
    // yyyyyyyy xxxxxxxx yy xx

    // 4x4 tiles for color
    return m2D_e_magicbits(x & 1023, y & 1023);
    //
    //y &= 1023;
    //x &= 1023;
    //u32 low_y = (y & 0b11)<<2;
    //u32 low_x = x & 0b11;
    //u32 high_y = ((y>>2)<<12);
    //u32 high_x = ((x>>2)<<4);
    //return low_y | low_x | high_y | high_x;

    //return (((y&1023)<<10) | (x&1023));
}

f32 pos_x = 0;
f32 pos_y = 0;
f32 dir_x = -1;
f32 dir_y = 0;
f32 plane_x = 0;
f32 plane_y = 0.66;
static f32 height = 128;
static s32 dheight = 1;

double fabs(double x);
f32 sqrtf(f32 x);
f32 atan2f(f32 y, f32 x);
f32 sinf(f32 x);
f32 cosf(f32 x);

s32 keysdown[32];
int num_keysdown = 0;
f32 dt;
f32 pitch = 0; //RENDER_HEIGHT/2;

f32 roll = 0;//-1.57;

f32 forwardbackward = 0;
f32 leftright = 0;
f32 updown = 0;
int lookup = 0;
int lookdown = 0;
f32 rollleftright = 0;

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
            lookup = 0;
            break;
        case SDL_SCANCODE_S:
            lookdown = 0;
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
            forwardbackward = 1.0;
            break;
        case SDL_SCANCODE_DOWN:
            forwardbackward = -1.0;
            break;
        case SDL_SCANCODE_LEFT: 
            leftright = +1.0;
            break;
        case SDL_SCANCODE_RIGHT: 
            leftright = -1.0;
            break;
        case SDL_SCANCODE_Z:
            updown = -1.0;
            break;
        case SDL_SCANCODE_X:
            updown = +1.0;
            break;
        case SDL_SCANCODE_A:
            lookup = 1;
            break;
        case SDL_SCANCODE_S:
            lookdown = 1;
            break;
    }
}


void handle_input(f32 dt) {


    if(leftright) {
        f32 rot_speed = dt * leftright * .9;

        f32 old_dir_x = dir_x;
        dir_x = dir_x * cos(rot_speed) - dir_y * sin(rot_speed);
        dir_y = old_dir_x * sin(rot_speed) + dir_y * cos(rot_speed);
        f32 old_plane_x = plane_x;
        plane_x = plane_x * cos(rot_speed) - plane_y * sin(rot_speed);
        plane_y = old_plane_x * sin(rot_speed) + plane_y * cos(rot_speed);
    }

    if(lookup) {
        pitch -= dt*400;
    }
    if(lookdown) {
        pitch += dt*400;
    }

    if(forwardbackward) {
        pos_x += dir_x * forwardbackward * dt * 100;
        pos_y += dir_y * forwardbackward* dt * 100;
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
        if(fabs(roll) <= .005) {
            roll = 0;
        }
    }
    //height += dt*50;
    //height -= dt*50;
    u32 cell_height = depthmap_u32s[get_map_idx((s32)pos_x, (s32)pos_y)]+10;
    if(height < cell_height) { height = cell_height; }

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

void handle_mouse_click(int mouse_x, int mouse_y) {
    mouse_is_down = 1;
    forwardbackward = 2.0;
#ifdef DOUBLE
    s32 centerX = WIDTH;///2;
    s32 centerY = HEIGHT;///2;
    s32 dx = centerX - mouse_x;
    s32 dy = centerY - mouse_y;

    leftright = (f32)dx / (WIDTH * 2) * 2;
    //rollleftright = (f32)dx / (WIDTH * 2) * 2;
    roll = -(f32)dx / (WIDTH*2) * 2;
    pitch  = (HEIGHT+200) + (f32)dy / HEIGHT * (HEIGHT);///(HEIGHT) + (f32)dy / (HEIGHT * 2);// * 300;
    //printf("new pitch: %f", pitch);
    updown    = (f32)dy / (HEIGHT * 2) * 10;
#else 
    s32 centerX = WIDTH/2;
    s32 centerY = WIDTH/2; 
    s32 dx = centerX - mouse_x;
    s32 dy = centerY - mouse_y;
    leftright = (f32)dx /  WIDTH * 2;
    pitch  = (HEIGHT+100) + (f32)dy / HEIGHT * (HEIGHT/2);
    updown    = (f32)dy / HEIGHT * 10;
#endif 
}

#include "vc.c"

double fmod(double x, double y);

#define PI 3.14159265359

//#define DIRECTIONAL_LIGHTING


#ifdef DOUBLE
static u32 pixels[WIDTH*2*HEIGHT*2];
#else
static u32 pixels[WIDTH*HEIGHT];
#endif




static void swizzle_array(u32* arr) {
    u32* tmp = malloc(1024*1024*4);
    for(int y = 0; y < 1024; y++) {
        for(int x = 0; x < 1024; x++) {
            int src_idx = (y<<10) | x;
            //tmp[dst_idx] = depthmap[src_idx];
            tmp[get_map_idx(x, y)] = arr[src_idx];
        }
    }
    //memcpy(depthmap, tmp, sizeof(depthmap));
    memcpy(arr, tmp, 1024*1024*4);
    //free(tmp);
    free(tmp);
}


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


f32 magnitude_vector(f32 x, f32 y, f32 z) {
    return sqrtf((x*x)+(y*y)+(z*z));
}


static void calculate_normals() {
    for(int y = 0; y < 512; y++) {
        for(int x = 0; x < 512; x++) {
            u32 center_depth = depthmap_u32s[get_map_idx(x,y)];
            f32 min_angle_for_point_right = 0;
            f32 max_angle_for_point_left = PI; // 180 degrees in radians

            u32 left_depth = depthmap_u32s[get_map_idx(x-1,y)];
            u32 right_depth = depthmap_u32s[get_map_idx(x+1,y)];
            u32 up_depth = depthmap_u32s[get_map_idx(x,y-1)];
            u32 down_depth = depthmap_u32s[get_map_idx(x,y+1)];

            s32 SurfaceVectorX = 2*(right_depth-left_depth);
            s32 SurfaceVectorY = -4;
            s32 SurfaceVectorZ = 2*(down_depth-up_depth);
            f32 magnitude_surf_vector = magnitude_vector(SurfaceVectorX, SurfaceVectorY, SurfaceVectorZ);
            f32 SurfaceVectorXF = SurfaceVectorX / magnitude_surf_vector;
            f32 SurfaceVectorYF = SurfaceVectorY / magnitude_surf_vector;
            f32 SurfaceVectorZF = SurfaceVectorZ / magnitude_surf_vector;
            depthmap_normal_xs[get_map_idx(x,y)] = SurfaceVectorXF;
            depthmap_normal_ys[get_map_idx(x,y)] = SurfaceVectorYF;
            depthmap_normal_zs[get_map_idx(x,y)] = SurfaceVectorZF;
        }
    }
}



static __m256i get_map_idx_256(__v8si x, __v8si y) {


    // swizzled
    __v8su wrap_mask = (__v8su)_mm256_set1_epi32(1023);
    
    __v8su two_bit_mask = (__v8su)_mm256_set1_epi32(0b11);

    __v8su wrapped_x = (__v8su)_mm256_and_si256((__m256i)x, (__m256i)wrap_mask);
    __v8su wrapped_y = (__v8su)_mm256_and_si256((__m256i)y, (__m256i)wrap_mask);


    //return _mm256_add_epi32(_mm256_slli_epi32((__m256i)wrapped_y, 10), (__m256i)wrapped_x);
    // swizzled stuff here
    __v8su low_x = (__v8su)_mm256_and_si256((__m256i)wrapped_x, (__m256i)two_bit_mask);
    __v8su low_y = (__v8su)_mm256_slli_epi32(_mm256_and_si256((__m256i)wrapped_y, (__m256i)two_bit_mask), 2);
    __v8su high_x = (__v8su)_mm256_slli_epi32(_mm256_srli_epi32((__m256i)wrapped_x, 2), 4);
    __v8su high_y = (__v8su)_mm256_slli_epi32(_mm256_srli_epi32((__m256i)wrapped_y, 2), 12);

    return _mm256_add_epi32(_mm256_add_epi32((__m256i)low_x, (__m256i)low_y), _mm256_add_epi32((__m256i)high_x, (__m256i)high_y));

}

f32 total_time;

static int bilinear = 1;
static int frame = 0;
static int swizzled = 0;
static int antialias = 1;

static u32 blend_abgrs(u32 abgr1, u32 abgr2) {
    u8 r1 = abgr1&0xFF;
    u32 half_abgr1 = (abgr1 & 0b111111101111111011111110)>>1;
    u32 half_abgr2 = (abgr2 & 0b111111101111111011111110)>>1;
    return (0xFF<<24)|(half_abgr1+half_abgr2);

}

s32 fixmul(s32 a, s32 b) {
    s64 ab = (s64)a*(s64)b; // 16.16+16.16 -> 32.32
    return (ab & 0x0000FFFFFFFF0000) >> 16;
}


f32 lerp(f32 a, f32 t, f32 b) {
    return ((1.0-t)*a)+(t*b);
}

__m256 lerp256(__m256 a, __m256 t, __m256 b) {
    const __m256 one_vec = _mm256_set1_ps(1.0);

    return _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(one_vec,t), a),
                         _mm256_mul_ps(t,b));
}

//#define VECTOR
//#define BILINEAR_DEPTH
//#define BILINEAR_COLOR


uint32_t blend_colors(uint32_t c1, uint32_t c2, uint8_t a1, uint8_t a2) {
    uint32_t r1 = OLIVEC_RED(c1);
    uint32_t g1 = OLIVEC_GREEN(c1);
    uint32_t b1 = OLIVEC_BLUE(c1);

    uint32_t r2 = OLIVEC_RED(c2);
    uint32_t g2 = OLIVEC_GREEN(c2);
    uint32_t b2 = OLIVEC_BLUE(c2);

    r1 = (r1*(255 - a2) + r2*a2)/255; if (r1 > 255) r1 = 255;
    g1 = (g1*(255 - a2) + g2*a2)/255; if (g1 > 255) g1 = 255;
    b1 = (b1*(255 - a2) + b2*a2)/255; if (b1 > 255) b1 = 255;

    return ((0xFF<<24) | (b1 << 16) | (g1 << 8) | r1);
}

f32 SRGB255ToLinear1(uint8_t c) {
    return (1.0f/255.0) * c;
}

u8 linearToSRGB255(f32 c) {
    return (u8)(255 * sqrtf(c));
}


s32 horizontal_min_epi32(__v8si v) {
    __v4si i = (__v4si)_mm256_extractf128_si256( (__m256i)v, 1 );\
    // compare lower and upper halves, get min(0,4), min(1,5), min(2,6), min(3,7)
    i = (__v4si)_mm_min_epi32( (__m128i)i, _mm256_castsi256_si128( (__m256i)v ) ); 
     // compare lower and upper 64-bit halves, get min(min(0,4), min(2,6)), min(min(1,5), min(3,7))
    i = (__v4si)_mm_min_epi32( (__m128i)i, _mm_shuffle_epi32( (__m128i)i, 0b00001110 ) ); 
    return min(i[0], i[1]);
}

s32 horizontal_max_epi32(__v8si v) {
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


__m256 bilinear_blend(__m256 A, __m256 B, __m256 C, __m256 D, __m256 sub_xs, __m256 sub_ys) {
    return lerp256(lerp256(A, sub_xs, B),
                            sub_ys,
                    lerp256(C, sub_xs, D));
}


f32 rotate_x(f32 angle, s32 input_x, s32 input_y) {
    
    f32 sinroll = sinf(roll);
    f32 cosroll = cosf(roll);            
    f32 yy = (s32)(input_y-HEIGHT/2);
    f32 xx = (s32)(input_x-WIDTH/2);
    f32 temp_xx = xx * cosroll - yy * sinroll;
    f32 temp_yy = xx * sinroll + yy*cosroll; 
    return temp_xx + WIDTH/2;
}

f32 rotate_y(f32 angle, s32 input_x, s32 input_y) {
    
    f32 sinroll = sinf(roll);
    f32 cosroll = cosf(roll);            
    f32 yy = (s32)(input_y-HEIGHT/2);
    f32 xx = (s32)(input_x-WIDTH/2);
    f32 temp_xx = xx * cosroll - yy * sinroll;
    f32 temp_yy = xx * sinroll + yy*cosroll; 
    xx = temp_xx + WIDTH/2;
    return temp_yy + HEIGHT/2;
}


static u32* inter_buffer;//[RENDER_WIDTH*RENDER_HEIGHT]; // this needs to be larger than the screen

f32 scale_height;

Olivec_Canvas vc_render(f32 dt) {
    handle_input(dt);   

    
    u32* cmap32 = (u32*)colormap;

    if(!swizzled) {
        s32 min_size = (s32) ceilf(sqrtf((WIDTH*WIDTH)+(WIDTH*WIDTH)));
        render_size = 2;
        while(render_size < min_size) {
            render_size *= 2;
        }

        scale_height = ((((16/9)*(.5))/(4/3))*render_size);
        //printf("picked render size of %i... oof\n", render_size);


        inter_buffer = malloc(sizeof(u32)*render_size*render_size);
        pitch = render_size/2;
        swizzle_array(depthmap_u32s);
        swizzle_array(cmap32);
        calculate_normals();
        swizzled = 1;
    }

#ifdef DOUBLE
    Olivec_Canvas oc = olivec_canvas(pixels, WIDTH*2, HEIGHT*2, WIDTH*2);
#else
    Olivec_Canvas oc = olivec_canvas(pixels, WIDTH, HEIGHT, WIDTH);
#endif 
    olivec_fill(oc, 0xFF000000); //BACKGROUND_COLOR);
    for(int i = 0; i < render_size*render_size; i++) {
        inter_buffer[i] = BACKGROUND_COLOR;
    }

    u32* cmap = (u32*)colormap;
  
    profile_block calc_profile; 
    
    f32 fog_r = (BACKGROUND_COLOR&0xFF);
    f32 fog_g = ((BACKGROUND_COLOR>>8)&0xFF);
    f32 fog_b = ((BACKGROUND_COLOR>>16)&0xFF);
    f32 max_z = 1400;
    f32 wrap_pos_x = pos_x;//fmod(pos_x, 1024);
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

    s32 y_buffer = ((render_size-HEIGHT)/2);
    s32 x_buffer = ((render_size-WIDTH)/2);

    // find bounding box of what needs to be drawn
    // technically we could figure this out per column
    // so that we never fill more than absolutely necessary, but i don't feel like working that out right now

    s32 x1 = rotate_x(roll, 0, 0);
    s32 x2 = rotate_x(roll, WIDTH-1, 0);
    s32 x3 = rotate_x(roll, 0, HEIGHT-1);
    s32 x4 = rotate_x(roll, WIDTH-1, HEIGHT-1);
    s32 y1 = rotate_y(roll, 0, 0);
    s32 y2 = rotate_y(roll, WIDTH-1, 0);
    s32 y3 = rotate_y(roll, 0, HEIGHT-1);
    s32 y4 = rotate_y(roll, WIDTH-1, HEIGHT-1);

    s32 min_x = min(x1, min(x2, min(x3, x4)))+x_buffer;
    s32 max_x = max(x1, max(x2, max(x3, x4)))+x_buffer;
    s32 min_y = min(y1, min(y2, min(y3, y4)))+y_buffer;
    s32 max_y = max(y1, max(y2, max(y3, y4)))+y_buffer;


    for(int x = min_x; x <= max_x; x++) {

        int prev_drawn_y = max_y+1;//render_size;

        f32 camera_space_x = 2 * x / ((f32)render_size) - 1; //x-coordinate in camera space

        f32 ray_dir_x = dir_x + plane_x * camera_space_x;
        f32 ray_dir_y = dir_y + plane_y * camera_space_x;
        //which box of the map we're in
        s32 map_x = ((s32)wrap_pos_x);
        s32 map_y = ((s32)wrap_pos_y);

        //length of ray from current position to next x or y-side
        f32 side_dist_x;
        f32 side_dist_y;

        //length of ray from one x or y-side to next x or y-side
        f32 delta_dist_x = (ray_dir_x == 0) ? 1e30 : fabs(1 / ray_dir_x);
        f32 delta_dist_y = (ray_dir_y == 0) ? 1e30 : fabs(1 / ray_dir_y);
        f32 perp_wall_dist = 0;

        //what direction to step in x or y-direction (either +1 or -1)
        int step_x;
        int step_y;

        if (ray_dir_x < 0) {
            step_x = -1;
            side_dist_x = (wrap_pos_x - map_x) * delta_dist_x;
        } else {
            step_x = 1;
            side_dist_x = (map_x + 1.0 - wrap_pos_x) * delta_dist_x;
        }
        
        if (ray_dir_y < 0) {
            step_y = -1;
            // subpixel step to grid
            side_dist_y = (wrap_pos_y - map_y) * delta_dist_y;
        } else {
            step_y = 1;
            side_dist_y = (map_y + 1.0 - wrap_pos_y) * delta_dist_y;
        }

        int side = (side_dist_x < side_dist_y) ? 0 : 1; //was a NS or a EW wall hit?
        int prev_side = 0;


        f32 prev_perp_wall_dist = 0;
 
        //printf("min_y: %i\n", min_y);
        while(perp_wall_dist <= max_z && prev_drawn_y > min_y) {   
            u32 idx = get_map_idx(map_x, map_y);               
            //prev_side = side;
            prev_perp_wall_dist = perp_wall_dist;

            perp_wall_dist = (side_dist_x < side_dist_y) ? side_dist_x : side_dist_y;
            map_x += (side_dist_x < side_dist_y) ? step_x : 0;
            map_y += (side_dist_x < side_dist_y) ? 0 : step_y;
            f32 side_dist_dx = (side_dist_x < side_dist_y) ? delta_dist_x : 0;
            f32 side_dist_dy = (side_dist_x < side_dist_y) ? 0 : delta_dist_y;
            side_dist_x += side_dist_dx;
            side_dist_y += side_dist_dy;
            /*
            if (side_dist_x < side_dist_y) {
                perp_wall_dist = side_dist_x;//(side_dist_x - delta_dist_x);
                map_x += step_x;
                side_dist_x += delta_dist_x;
                //side = 0;
            } else {
                perp_wall_dist = side_dist_y;
                side_dist_y += delta_dist_y;
                map_y += step_y;
                //side = 1;
            }
            */

            if(prev_perp_wall_dist == 0 || perp_wall_dist == 0) { continue; }
            u32 depth = depthmap_u32s[idx];
            u32 abgr = cmap[idx];

            f32 relative_height = (height-depth);
            
            //f32 project_wall_dist = (relative_depth > 0) ? prev_perp_wall_dist : perp_wall_dist;
            //if(perp_wall_dist == 0) { continue; }
            f32 front_invz = scale_height / prev_perp_wall_dist;
            f32 back_invz = scale_height / perp_wall_dist;
            //f32 invz = (1. / project_wall_dist);
            
            f32 front_float_projected_height = relative_height*front_invz;//*scale_height;
            f32 back_float_projected_height = relative_height*back_invz;//*scale_height;
            f32 avg_dist = (prev_perp_wall_dist+perp_wall_dist)/2;
            f32 z_to_1 = (avg_dist/max_z)*(avg_dist/max_z);
            f32 fog_factor = lerp(0, z_to_1, 1); //(z_to_1*1-z_to_1);
            
            f32 fog_r = fog_factor*(BACKGROUND_COLOR&0xFF);
            f32 fog_g = fog_factor*((BACKGROUND_COLOR>>8)&0xFF);
            f32 fog_b = fog_factor*((BACKGROUND_COLOR>>16)&0xFF);
   
            f32 one_minus_fog = 1-fog_factor;
            //f32 side_mult = 1;
            //if(prev_side == 1) {
            //    side_mult = 0.8;
            //}

            f32 fr  = ((abgr&0xFF) * one_minus_fog) + fog_r;
            f32 fg = (((abgr>>8)&0xFF) * one_minus_fog) + fog_g;
            f32 fb = (((abgr>>16)&0xFF) * one_minus_fog) + fog_b;
            //f32 fsr = fr * side_mult;
            //f32 fsg = fr * side_mult;
            //f32 fsb = fr * side_mult; 

            uint8_t r = (uint8_t)(fr);
            uint8_t g = (uint8_t)(fg);
            uint8_t b = (uint8_t)(fb);
            //u32 side_abgr = (0xFF<<24)|(b<<16)|(g<<8)|r;
            abgr = (0xFF<<24)|(b<<16)|(g<<8)|r;

            s32 front_int_projected_height = floor(front_float_projected_height);
            s32 back_int_projected_height = floor(back_float_projected_height);
            int front_heightonscreen = front_int_projected_height + pitch;
            int back_heightonscreen = back_int_projected_height + pitch;
           
            front_heightonscreen = max(min_y, front_heightonscreen);
            back_heightonscreen = max(min_y, back_heightonscreen);

            if(front_heightonscreen < prev_drawn_y) {
                //uint32_t* ptr = inter_buffer+(x)+(front_heightonscreen*render_size);
                for(int y = front_heightonscreen; y < prev_drawn_y; y++) {
                    u32 bb = m2D_e_magicbits(x,y);
                #ifdef DIRECTIONAL_LIGHTING
                    inter_buffer[bb] = side_abgr;
                #else
                    inter_buffer[bb] = abgr;
                #endif
                    //*ptr = side_abgr;
                    //ptr += render_size;
                }
                prev_drawn_y = front_heightonscreen;
            }

            if(back_heightonscreen < prev_drawn_y) {
                //uint32_t* ptr = inter_buffer+(x)+(back_heightonscreen*render_size);
                for(int y = back_heightonscreen; y < prev_drawn_y; y++) {
                    u32 bb = m2D_e_magicbits(x,y);
                    inter_buffer[bb] = abgr;
                    //*ptr = abgr;
                    //ptr += render_size;
                }
                prev_drawn_y = back_heightonscreen;
            }

        }

    }


    f32 sinroll = sinf(roll);
    f32 cosroll = cosf(roll);


    f32 dx_per_x = rotate_x(roll, 1, 0) - rotate_x(roll, 0, 0);
    f32 dx_per_y = rotate_x(roll, 0, 1) - rotate_x(roll, 0, 0);
    f32 dy_per_x = rotate_y(roll, 1, 0) - rotate_y(roll, 0, 0);
    f32 dy_per_y = rotate_y(roll, 0, 1) - rotate_y(roll, 0, 0);


    f32 row_y = rotate_y(roll, 0, 0);
    f32 row_x = rotate_x(roll, 0, 0);
    for(int oy = 0; oy < HEIGHT; oy++) {
        f32 yy = row_y;
        f32 xx = row_x;
        for(int ox = 0; ox < WIDTH; ox++) {


            s32 iyy = yy+y_buffer;
            s32 ixx = xx+x_buffer;
            yy += dy_per_x;
            xx += dx_per_x;
            u32 bb = m2D_e_magicbits(ixx,iyy);
            u32 pix = inter_buffer[bb];
            #ifdef DOUBLE
                oc.pixels[(oy*2)*(WIDTH*2)+(ox*2)] = pix;
                oc.pixels[(oy*2)*(WIDTH*2)+(ox*2)+1] = pix;
                oc.pixels[((oy*2)+1)*(WIDTH*2)+(ox*2)] = pix;
                oc.pixels[((oy*2)+1)*(WIDTH*2)+(ox*2)+1] = pix;
            #else
                oc.pixels[oy*WIDTH+ox] = pix;
            #endif
        }
        row_x += dx_per_y;
        row_y += dy_per_y;
    }

    f32 ms = dt*1000;
    f32 fps = 1000/ms;
    printf("ms: %f, fps: %f\n", ms, fps);
    total_time += ms;
    frame++;
    return oc;
}

/*
Olivec_Canvas vc_render_old(f32 dt) {
    handle_input(dt);
    u32* cmap32 = (u32*)colormap;

    if(!swizzled) {
        swizzle_array(depthmap_u32s);
        swizzle_array(cmap32);
        calculate_normals();
        swizzled = 1;
    }

    Olivec_Canvas oc = olivec_canvas(pixels, WIDTH, HEIGHT, WIDTH);

    olivec_fill(oc, BACKGROUND_COLOR);

    f32 sinang = sinf(ang);
    f32 cosang = cosf(ang);

    for(int i=0; i < WIDTH; i++) {
        hiddeny[i] = HEIGHT;
        prevpix[i] = 0;
        prevsuby[i] = 0.0;
        lastwasskipped[i] = 0;
    }

    f32 dz = 0.5;
    u32 dz16_16 = 1<<16;

    u32* cmap = (u32*)colormap;
    // Draw from front to back
    int rows = 0;
    f32 z, invz;
    //u32 z_16_16;
    f32 inv_z;
    //u32 invz_16_16;
    //u32 ddz_16_16 = .005*65536;
    f32 ddz = 0;//.0001;
        
    __m256i zero_vec = _mm256_set1_epi32(0);
    __m256i horizon_vec = _mm256_set1_epi32(pitch);
            
    profile_block calc_profile; 

    s64 total_fillable_pix = 0;
    s64 total_drawn_pix = 0;
    s64 fully_skipped_chunks = 0;
    s64 drawn_chunks = 0;

    const __m256 one_ps_vec = _mm256_set1_ps(1.0);
    const __m256 zero_ps_vec = _mm256_set1_ps(0.0);
    __v8si one_vec = (__v8si)_mm256_set1_epi32(1);

    __m256 TwoFiftyFiveVec = _mm256_set1_ps(255.0);
    __m256i MaskFF = _mm256_set1_epi32(0xFF);
    __m256i OpaqueAlpha = _mm256_set1_epi32(0xFF000000);
    __m256 HalfVec = _mm256_set1_ps(0.5);
    f32 background_r = (BACKGROUND_COLOR&0xFF)>>1;
    f32 background_g = ((BACKGROUND_COLOR>>8)&0xFF)>>1;
    f32 background_b = ((BACKGROUND_COLOR>>16)&0xFF)>>1;
    __m256 background_r_vec = _mm256_set1_ps(background_r);
    __m256 background_g_vec = _mm256_set1_ps(background_g);
    __m256 background_b_vec = _mm256_set1_ps(background_g);
    f32 max_z = 800;//320.0;


   
    for(z=1; z < max_z; z+=dz) { //}, dz+=ddz) {
        f32 z_to_1 = (z/max_z)*(z/max_z);
        f32 fog_factor = lerp(0, z_to_1, 1); //(z_to_1*1-z_to_1);
        
        //f32 z_falloff_factor = 1/z; //lerp(1, z_to_1, 0);
        //f32 one_minus_z_fallof = 1-z_falloff_factor;
        //__m256 one_minus_z_falloff_vec = _mm256_set1_ps(one_minus_z_fallof);

        f32 one_minus_fog = 1-fog_factor;
        __m256 one_minus_fog_vec = _mm256_set1_ps(one_minus_fog);
        f32 premult_fog_r = fog_factor*(BACKGROUND_COLOR&0xFF);
        f32 premult_fog_g = fog_factor*((BACKGROUND_COLOR>>8)&0xFF);
        f32 premult_fog_b = fog_factor*((BACKGROUND_COLOR>>16)&0xFF);

        //f32 premult_light_r = z_falloff_factor * (0xFF);
        //f32 premult_right_g = z_falloff_factor * (0xE0);
        //f32 premult_light_b = z_falloff_factor * (0x80);
        __m256 premult_fog_r_vec = _mm256_set1_ps(premult_fog_r);
        __m256 premult_fog_g_vec = _mm256_set1_ps(premult_fog_g);
        __m256 premult_fog_b_vec = _mm256_set1_ps(premult_fog_b);
        
        __m256 light_r_vec = _mm256_set1_ps(0xFF);
        __m256 light_g_vec = _mm256_set1_ps(0xE0);
        __m256 light_b_vec = _mm256_set1_ps(0x80);
        //__m256 premult_light_r_vec = _mm256_set1_ps(premult_light_r);
        //__m256 premult_light_g_vec = _mm256_set1_ps(premult_right_g);
        //__m256 premult_light_b_vec = _mm256_set1_ps(premult_light_b);
        rows++;

        // 90 degree field of view
        f32 plx = (-cosang*1.5 * z) - (sinang*1.5*z);
        f32 ply = (sinang*1.5*z) - (cosang*1.5*z);
        f32 prx = (cosang*1.5*z) - (sinang*1.5*z);
        f32 pry = (-sinang*1.5 * z) - (cosang*1.5*z);
        

        f32 dx = (prx - plx) / WIDTH;
        f32 dy = (pry - ply) / WIDTH;
        plx += pos_x;
        ply += pos_y;
        
        __v8sf plxs = (__v8sf)_mm256_set_ps(
                                              plx+(dx*7),
                                              plx+(dx*6),
                                              plx+(dx*5),
                                              plx+(dx*4),
                                              plx+(dx*3),
                                              plx+(dx*2),
                                              plx+dx,
                                              plx
        );
        

        
        __v8sf plys = (__v8sf)_mm256_set_ps(
                                                    ply+(dy*7),
                                                    ply+(dy*6),
                                                    ply+(dy*5),
                                                    ply+(dy*4),
                                                    ply+(dy*3),
                                                    ply+(dy*2),
                                                    ply+dy,
                                                    ply
        );
        
        
        __v8sf dx_vec = (__v8sf)_mm256_set1_ps(dx*8);
        __v8sf dy_vec = (__v8sf)_mm256_set1_ps(dy*8);

        invz = 1. / z * 240.;
        
        //int64_t scaled_one = 4294967296;
        //int32_t recip = scaled_one/z_16_16;

        //invz_16_16 = recip*240; // 32.0 - 16.16 -> 16.16

        __v8si xs_vector = (__v8si)_mm256_set_epi32(7,6,5,4,3,2,1,0); //0,1,2,3,4,5,6,7);
        __v8si screen_dx_vector = (__v8si)_mm256_set1_epi32(8);
        
        
        __v8sf inv_z_vec = (__v8sf)_mm256_set1_ps(invz);

        

    
    //TimeBlock(calc_profile, "calculate_coords_projection_and_clipping");

    int iteration_count;
    #ifdef VECTOR
        iteration_count = WIDTH/8;
        for(int x=continue_x; x < WIDTH; x+=8) {
    #else
        iteration_count = WIDTH;
        for(int x = 0; x < WIDTH; x++) {
    #endif



            u32 depth;
            __v8su depths;
        #ifdef VECTOR 
            __v8si ixs = (__v8si)_mm256_cvtps_epi32(_mm256_floor_ps(plxs));
            __v8si iys = (__v8si)_mm256_cvtps_epi32(_mm256_floor_ps(plys));
        #else 
            s32 ix = floor(plx);
            s32 iy = floor(ply);
        #endif 

        #if defined(BILINEAR_DEPTH) || defined(BILINEAR_COLOR)
            #ifdef VECTOR
                __m256 sub_xs = _mm256_sub_ps(plxs, _mm256_cvtepi32_ps((__m256i)depth_xs));
                __m256 sub_ys = _mm256_sub_ps(plys, _mm256_cvtepi32_ps((__m256i)depth_ys));

                __m256i idxs = get_map_idx_256(ixs, iys);
                __m256i right_idxs = get_map_idx_256((__v8si)_mm256_add_epi32((__m256i)ixs, (__m256i)one_vec),
                                                        iys);
                __m256i down_idxs = get_map_idx_256(depth_xs, (__v8si)_mm256_add_epi32((__m256i)iys, (__m256i)one_vec));
                __m256i down_right_idxs = get_map_idx_256((__v8si)_mm256_add_epi32((__m256i)ixs, (__m256i)one_vec), 
                                                           (__v8si)_mm256_add_epi32((__m256i)iys, (__m256i)one_vec));
            #else 
                f32 sub_x = plx - (f32)ix;
                f32 sub_y = ply - (f32)iy;
                s32 idx = get_map_idx(ix, iy);
                s32 right_idx = get_map_idx(ix+1, iy);
                s32 down_idx = get_map_idx(ix, (iy+1));
                s32 down_right_idx = get_map_idx(ix+1, iy+1);

            #endif 
        #else 
            #ifdef VECTOR
                __v8su idxs = (__v8su)get_map_idx_256(ixs, iys);
            #else
                s32 idx = get_map_idx(ix, iy);
            #endif
        #endif

        #ifdef VECTOR 
            depths = (__v8su)_mm256_i32gather_epi32( depthmap_u32s, (__m256i)idxs, 4);
        #else
            depth = depthmap_u32s[idx];
        #endif
            
        #ifdef VECTOR 
            __v8sf height_vec = _mm256_set1_ps(height);
            __v8sf height_diffs = (__v8sf)_mm256_sub_ps((__m256)height_vec, _mm256_cvtepi32_ps((__m256i)depths));

            __v8sf projected_heights = (__v8sf)_mm256_mul_ps((__m256)height_diffs, (__m256)inv_z_vec);
            __v8si int_projected_heights = (__v8si)_mm256_cvtps_epi32((__m256)projected_heights);
            __m256 proj_sub_ys = _mm256_sub_ps((__m256)projected_heights, _mm256_cvtepi32_ps((__m256i)int_projected_heights));
            //__v8si height_diffs = (__v8si)_mm256_sub_epi32(_mm256_set1_epi32(height), (__m256i)depths);
            //__v8si projected_heights_24_8 = (__v8si)_mm256_mullo_epi32((__m256i)height_diffs, (__m256i)shifted_inv_z_vec);
            //__v8si projected_heights = (__v8si)_mm256_srai_epi32((__m256i)projected_heights_24_8, 8);
            

            //f32 float_projected_height = (height-depth)*invz;


            __v8si heights_on_screen = (__v8si)_mm256_add_epi32((__m256i)int_projected_heights, (__m256i)horizon_vec);
            heights_on_screen = (__v8si)_mm256_max_epi32((__m256i)heights_on_screen, zero_vec);



            __v8si prev_ys = (__v8si)_mm256_load_si256((const __m256i*)&hiddeny[x]);



            __v8si heights_minus_one = (__v8si)_mm256_sub_epi32((__m256i)heights_on_screen, (__m256i)one_vec);
            __v8si shown_mask = (__v8si)_mm256_cmpgt_epi32((__m256i)prev_ys, (__m256i)heights_minus_one);


            int mask = _mm256_movemask_epi8((__m256i)shown_mask);
        #else
            
            //int32_t height_diff = height-depth;

            //int32_t projected_height = height_diff * (invz_16_16>>8);
            //int32_t int_projected_height = projected_height>>8;

            f32 float_projected_height = (height-depth)*invz;
   
            s32 int_projected_height = floor(float_projected_height);
            f32 proj_sub_y = float_projected_height - (f32)int_projected_height;
            int heightonscreen = int_projected_height + pitch;
            if(heightonscreen < 0) { 
                heightonscreen = 0;
            }
            int prevy = hiddeny[x];
        #endif 
            


        uint32_t abgr;
        __v8su abgrs;

    
        #ifdef VECTOR 
            abgrs = (__v8su)_mm256_i32gather_epi32(cmap,  (__m256i)idxs, 4);
            
            __m256 TexelR = _mm256_cvtepi32_ps(_mm256_and_si256((__m256i)abgrs, MaskFF));
            __m256 TexelG = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32((__m256i)abgrs, 8), MaskFF));
            __m256 TexelB = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32((__m256i)abgrs, 16), MaskFF));
            TexelR = _mm256_add_ps(_mm256_mul_ps(TexelR, one_minus_fog_vec), premult_fog_r_vec);
            TexelG = _mm256_add_ps(_mm256_mul_ps(TexelG, one_minus_fog_vec), premult_fog_g_vec);
            TexelB = _mm256_add_ps(_mm256_mul_ps(TexelB, one_minus_fog_vec), premult_fog_b_vec);
            __v8su Intr = (__v8su)_mm256_cvtps_epi32(TexelR);
            __v8su Intg = (__v8su)_mm256_cvtps_epi32(TexelG);
            __v8su Intb = (__v8su)_mm256_cvtps_epi32(TexelB);

            __v8su Sr = Intr;
            __v8su Sg = (__v8su)_mm256_slli_epi32((__m256i)Intg, 8);
            __v8su Sb = (__v8su)_mm256_slli_epi32((__m256i)Intb, 16);
            __v8su Sa = (__v8su)OpaqueAlpha;
            abgrs = (__v8su) _mm256_or_si256(_mm256_or_si256((__m256i)Sr, (__m256i)Sg), _mm256_or_si256((__m256i)Sb, OpaqueAlpha));
        #else

            abgr = cmap[idx];
            f32 fr  = (((f32)(abgr&0xFF)) * one_minus_fog)+premult_fog_r;
            f32 fg = (((f32)((abgr>>8)&0xFF)) * one_minus_fog)+premult_fog_g;
            f32 fb = (((f32)((abgr>>16)&0xFF)) * one_minus_fog)+premult_fog_b;
            
            uint8_t r = (uint8_t)(fr);
            uint8_t g = (uint8_t)(fg);
            uint8_t b = (uint8_t)(fb);
            abgr = (0xFF<<24)|(b<<16)|(g<<8)|r;
        #endif



  

            
        #ifdef VECTOR

            if(mask) {
                
                //s32 min_height = SDL_MAX_SINT32;
                //s32 max_prev_y = SDL_MIN_SINT32;
                s32 drawn_pix = 0;
                s32 fillable_pix = 0; 
            #if 1            
                int x_in_tile = x&0b111;
                int tile_x = x>>3;
                int outer_tile_offset = tile_x*(8*HEIGHT);
                __v8si max_int_vec = (__v8si)_mm256_set1_epi32(SDL_MAX_SINT32);
                __v8si min_shown_heights = (__v8si)_mm256_blendv_epi8((__m256i)max_int_vec, (__m256i)heights_on_screen, (__m256i)shown_mask);
                s32 min_height = horizontal_min_epi32(min_shown_heights);
                int inner_tile_offset = (x_in_tile+(min_height*8));
                __v8si min_int_vec = (__v8si)_mm256_set1_epi32(SDL_MIN_SINT32);
                __v8si max_shown_prevy = (__v8si)_mm256_blendv_epi8((__m256i)min_int_vec, (__m256i)prev_ys, (__m256i)shown_mask);
                s32 max_height = horizontal_max_epi32(max_shown_prevy);

                uint32_t* ptr = oc.pixels+(x)+(min_height*oc.stride);
                //uint32_t* ptr = oc.pixels + outer_tile_offset + inner_tile_offset; //.pixels+(x)+(min_height*oc.stride);
                __v8si y_vec = (__v8si)_mm256_set1_epi32(min_height);
                for(int y = min_height; y < max_height; y++) {
                    __v8si within_span =(__v8si) _mm256_cmpgt_epi32((__m256i)y_vec, (__m256i)heights_minus_one);
                    __v8si not_hidden = (__v8si)_mm256_cmpgt_epi32((__m256i)prev_ys, (__m256i)y_vec);
                    __v8si pixels_shown = (__v8si)_mm256_and_si256((__m256i)within_span, (__m256i)not_hidden);
                    __v8su old_abgrs = (__v8su)_mm256_load_si256((__m256i*)ptr);
                    __v8su new_pixels = (__v8su)_mm256_blendv_epi8((__m256i)old_abgrs, (__m256i)abgrs, (__m256i)pixels_shown);
                    _mm256_store_si256((__m256i*)ptr, (__m256i)new_pixels); //abgrs);
                    ptr += oc.stride;
                    //ptr += 8;
                    y_vec = (__v8si)_mm256_add_epi32((__m256i)y_vec, (__m256i)one_vec);
                }
                
            #else
                for(int i = 0; i < 8; i++) {
                    if(mask & (1<<(i*4))) {
                        // got a line to draw :)
                        min_height = min(min_height, heights_on_screen[i]);
                        max_prev_y = max(max_prev_y, prev_ys[i]);
                        int x_in_tile = x&0b111;
                        int tile_x = x>>3;
                        int outer_tile_offset = tile_x*(8*HEIGHT);
                        int inner_tile_offset = (x_in_tile+(heights_on_screen[i]*8));
                        uint32_t* ptr = oc.pixels+(x)+i+(heights_on_screen[i]*oc.stride);

                        drawn_pix += prev_ys[i] - heights_on_screen[i];

                        for(int y = heights_on_screen[i]; y < prev_ys[i]; y++) {

                            *ptr = abgrs[i];
                            //ptr++;
                            ptr+= oc.stride;
                            //ptr += 8;
                        }
                        //prevpix[x+i] = abgr;
                        //prevsuby[x] = proj_sub_y;
                        //lastwasskipped[x+i] = 0;

                    } else {
                        //lastwasskipped[x] = 1;
                        //lastwasskipped[x+i] = 1;
                    }
                }
            #endif
                
                //__v8si new_prev_ys = (__v8si )_mm256_blend_epi32((__m256i)prev_ys, (__m256i)heights_on_screen, bit_mask);
                __m256i new_prev_ys = _mm256_blendv_epi8((__m256i)prev_ys, (__m256i)heights_on_screen, (__m256i)shown_mask);
                _mm256_store_si256((__m256i*)&hiddeny[x], (__m256i)new_prev_ys);
                //fillable_pix = (max_prev_y - min_height) * 8;
                //total_drawn_pix += drawn_pix;
                //total_fillable_pix += fillable_pix;
                //drawn_chunks++;
            } else {
                //fully_skipped_chunks++;
            }
            
        #else 
            if(heightonscreen < prevy) {
                //uint32_t *ptr = &pixels[heightonscreen*WIDTH+x];
                int x_in_tile = x&0b111;
                int tile_x = x>>3;
                int outer_tile_offset = tile_x*(8*HEIGHT);
                int inner_tile_offset = (x_in_tile+(heightonscreen*8));
                uint32_t* ptr = oc.pixels+(x)+(heightonscreen*oc.stride);
                //ptr = oc.pixels + outer_tile_offset + inner_tile_offset;
                if(0) { //lastwasskipped[x]) {
                    for(int y = heightonscreen; y < prevy; y++) {
                        *ptr = abgr;
                        ptr+= 8;
                        ptr += oc.stride;
                        //ptr++;
                    }
                    // blend top pixel of crest
                    //lerp_abgr(abgr, prev )
                    //uint8_t prev_alpha = (s32)(prevsuby[x]*255);
                    //uint8_t rem_alpha = 255-prev_alpha;
                    //u32 blended = blend_colors(abgr, prevpix[x], rem_alpha, prev_alpha);
                    //*ptr = blended;//blend_abgrs(abgr, prevpix[x], prevsuby[x]);
                } else {
                    for(int y = heightonscreen; y < prevy; y++) {
                        
                        *ptr = abgr;
                        ptr += oc.stride;
                        //ptr++;
                        //ptr+= 8;
                    }                    

                }
                //prevpix[x] = abgr;
                //prevsuby[x] = proj_sub_y;
                hiddeny[x] = heightonscreen;
                //lastwasskipped[x] = 0;
            } else {
                //occluded_pixels += heightonscreen - prevy;
                //lastwasskipped[x] = 1;
            }
        #endif
            
            
            //DrawVerticalLine(i, heightonscreen, hiddeny[i], map.color[mapoffset]);
            //if (heightonscreen < hiddeny[i]) hiddeny[i] = heightonscreen;
            plx += dx;
            ply += dy;
            //plx_16_16 += dx_16_16;
            //ply_16_16 += dy_16_16;
            plxs = (__v8sf)_mm256_add_ps((__m256)dx_vec, (__m256)plxs);
            plys = (__v8sf)_mm256_add_ps((__m256)dy_vec, (__m256)plys);
            xs_vector = (__v8si)_mm256_add_epi32((__m256i)xs_vector, (__m256i)screen_dx_vector);
        }
        //end_counted_profile_block(calc_profile, iteration_count);

    }
    //printf("first z: %f, 1/z: %f\n", 1, 240.0);
    //printf("final z: %f, 1/z: %f\n", z, invz);

    printf("rows evaluated %i\n", rows);
    printf("horizon %f\n", pitch);

    printf("height: %f\n", height);


    
    //if(lightang >= PI) {
    //    lightang -= 0.01;
    //} else {
        lightang += 0.005;
    //}
    f32 ms = dt*1000;
    f32 fps = 1000/ms;
    total_time += ms;
    frame++;
    if(((frame % 100)) == 0) {
        //bilinear = !bilinear;
        //antialias = !antialias;
    }
    printf("ms: %f, fps: %f\n", ms, fps);
    printf("bilinear: %i\n", bilinear);
    printf("antialias: %i\n", antialias);
    printf("lighting: %i\n", lighting);

    return oc;
}
*/
ProfilerEndOfCompilationUnit;