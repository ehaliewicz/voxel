
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


#define DOUBLE
#define WIDTH (1920/2)
#define HEIGHT (1080/2)



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


static __m256i get_map_idx_256(__m256i x, __m256i y) {


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
        if(fabs(roll) <= dt*.3) {
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

#define VECTOR


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

__m256 abs_ps(__m256 x) {
    __m256 sign_mask = _mm256_set1_ps(-0.0); // -0.f = 1 << 31
    return _mm256_andnot_ps(sign_mask, x);
}

#define LOG2(X) ((unsigned) (8*sizeof (unsigned long long) - __builtin_clzll((X)) - 1))

s32 fb_swizzle(s32 x, s32 y) {
    
    u32 num_bits = LOG2(render_size);
    // we want 3 lower bits of x to be contiguous
    u32 low_x = x & 0b111;
    u32 high_x = x >> 3;

    return (high_x<<((11)+3))|(y<<3)|low_x;
    
    //const int row_size = 8;
    //const int row_size_shift = 3;
    //const int row_mask = 0b111;
    //s32 in_tile_x = x & row_mask;
    //s32 out_tile_x = x>>row_size_shift;
    //s32 in_tile_row_offset = (y*row_size)+in_tile_x;
    //s32 out_tile_offset = out_tile_x*(render_size*row_size);
    //return out_tile_offset+in_tile_row_offset+in_tile_x;
}

//#define VECTOR


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

        inter_buffer = malloc((sizeof(u32)*render_size*render_size)+16);
        while(((intptr_t)inter_buffer)&0b1111) {
            inter_buffer++;
        }
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
    __m256 fog_r_vec = _mm256_set1_ps(fog_r);
    __m256 fog_g_vec = _mm256_set1_ps(fog_g);
    __m256 fog_b_vec = _mm256_set1_ps(fog_b);
    f32 max_z = 1400;
    __m256 max_z_vec = _mm256_set1_ps(1400);
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
    __m256 wrap_pos_x_vec = _mm256_set1_ps(wrap_pos_x);
    __m256 wrap_pos_y_vec = _mm256_set1_ps(wrap_pos_y);

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

    __m256i min_y_vec = _mm256_set1_epi32(min_y);


    //int aligned_min_x = min_x & (~0b111);
    int aligned_min_x = min_x;
    while(aligned_min_x & 0b111) { aligned_min_x--; }
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

    __m256i MaskFF_vec = _mm256_set1_epi32(0xFF);
    __m256i OpaqueAlpha = _mm256_set1_epi32(0xFF000000);

    f32 one_over_max_z = 1/max_z;
    __m256 one_over_max_z_vec = _mm256_set1_ps(one_over_max_z);
#ifdef VECTOR
    for(int x = aligned_min_x; x <= max_x; x += 8) {
#else
    for(int x = min_x; x <= max_x; x++) {
#endif
        int prev_drawn_y = max_y+1;//render_size;
        __m256i prev_drawn_ys = _mm256_set1_epi32(max_y+1);

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
        f32 delta_dist_x = (ray_dir_x == 0) ? 1e30 : fabs(1 / ray_dir_x);
        f32 delta_dist_y = (ray_dir_y == 0) ? 1e30 : fabs(1 / ray_dir_y);

        f32 wrap_x_minus_map_x = (wrap_pos_x - map_x);
        f32 map_x_plus_one_minus_wrap_x = (map_x + (1.0 - wrap_pos_x));
        f32 wrap_y_minus_map_y = (wrap_pos_y - map_y);
        f32 map_y_plus_one_minus_wrap_y = (map_y + (1.0 - wrap_pos_y));
        __m256 wrap_x_minus_map_x_vec = _mm256_set1_ps(wrap_x_minus_map_x);
        __m256 map_x_plus_one_minus_wrap_x_vec = _mm256_set1_ps(map_x_plus_one_minus_wrap_x);
        __m256 wrap_y_minus_map_y_vec = _mm256_set1_ps(wrap_y_minus_map_y);
        __m256 map_y_plus_one_minus_wrap_y_vec = _mm256_set1_ps(map_y_plus_one_minus_wrap_y);


        //what direction to step in x or y-direction (either +1 or -1)
        int step_x = (ray_dir_x < 0) ? -1 : 1;
        int step_y = (ray_dir_y < 0) ? -1 : 1;
        f32 perp_wall_dist = 0;
        f32 prev_perp_wall_dist = 0;

        //length of ray from current position to next x or y-side
        f32 side_dist_x = (ray_dir_x < 0 ? wrap_x_minus_map_x : map_x_plus_one_minus_wrap_x) * delta_dist_x;
        f32 side_dist_y = (ray_dir_y < 0 ? wrap_y_minus_map_y : map_y_plus_one_minus_wrap_y) * delta_dist_y;


        __m256 ray_dir_xs_less_than_zero = _mm256_cmp_ps(ray_dir_xs, zero_ps_vec, _CMP_LT_OQ);
        __m256 ray_dir_ys_less_than_zero = _mm256_cmp_ps(ray_dir_ys, zero_ps_vec, _CMP_LT_OQ);
        __m256i step_xs = _mm256_blendv_epi8(one_vec, negative_one_vec, (__m256i)ray_dir_xs_less_than_zero);
        __m256i step_ys = _mm256_blendv_epi8(one_vec, negative_one_vec, (__m256i)ray_dir_ys_less_than_zero);

        __m256 side_dist_xs = _mm256_mul_ps(_mm256_blendv_ps(map_x_plus_one_minus_wrap_x_vec, wrap_x_minus_map_x_vec, ray_dir_xs_less_than_zero),
                                            delta_dist_xs);
        __m256 side_dist_ys = _mm256_mul_ps(_mm256_blendv_ps(map_y_plus_one_minus_wrap_y_vec, wrap_y_minus_map_y_vec, ray_dir_ys_less_than_zero), 
                                            delta_dist_ys);

        __m256 perp_wall_dists = _mm256_set1_ps(perp_wall_dist);
        __m256 prev_perp_wall_dists = _mm256_set1_ps(prev_perp_wall_dist);

    #ifdef VECTOR 
        int dist_lte_max_z_mask = _mm256_movemask_ps(_mm256_cmp_ps(perp_wall_dists, max_z_vec, _CMP_LE_OQ));
        int prev_draw_y_gt_min_y_mask = _mm256_movemask_epi8(_mm256_cmpgt_epi32(prev_drawn_ys, min_y_vec));

        while(dist_lte_max_z_mask && prev_draw_y_gt_min_y_mask) { 
    #else 
        while(perp_wall_dist <= max_z && prev_drawn_y > min_y) {   
    #endif
            u32 idx = get_map_idx(map_x, map_y);   
            __m256i idxs = get_map_idx_256(map_xs, map_ys);            

            prev_perp_wall_dist = perp_wall_dist;
            prev_perp_wall_dists = perp_wall_dists;

            __m256 side_dist_xs_less_than_ys = _mm256_cmp_ps(side_dist_xs, side_dist_ys, _CMP_LT_OQ);
            perp_wall_dist = (side_dist_x < side_dist_y) ? side_dist_x : side_dist_y;
            perp_wall_dists = _mm256_blendv_ps(side_dist_ys, side_dist_xs, side_dist_xs_less_than_ys);

            int map_x_dx = (side_dist_x < side_dist_y) ? step_x : 0;
            int map_y_dy = (side_dist_x < side_dist_y) ? 0 : step_y;
            map_x += map_x_dx;
            map_y += map_y_dy;

            __m256i map_xs_dxs = _mm256_blendv_epi8(zero_vec, step_xs, (__m256i)side_dist_xs_less_than_ys);
            __m256i map_ys_dys = _mm256_blendv_epi8(step_ys, zero_vec, (__m256i)side_dist_xs_less_than_ys);

            //printf("map_x_dx %i, map_xs_dxs[7], %i\n", map_x_dx, map_xs_dxs[7]);
            //assert(map_x_dx == map_xs_dxs[7]);
            //assert(map_y_dy == map_ys_dys[7]);

            map_xs = _mm256_add_epi32(map_xs, map_xs_dxs);
            map_ys = _mm256_add_epi32(map_ys, map_ys_dys);

            f32 side_dist_dx = (side_dist_x < side_dist_y) ? delta_dist_x : 0;
            f32 side_dist_dy = (side_dist_x < side_dist_y) ? 0 : delta_dist_y;

            __m256 side_dist_xs_dxs = _mm256_blendv_ps(zero_ps_vec, delta_dist_xs, side_dist_xs_less_than_ys);
            __m256 side_dist_ys_dys = _mm256_blendv_ps(delta_dist_ys, zero_ps_vec, side_dist_xs_less_than_ys);
            side_dist_x += side_dist_dx;
            side_dist_y += side_dist_dy;
            side_dist_xs = _mm256_add_ps(side_dist_xs, side_dist_xs_dxs);
            side_dist_ys = _mm256_add_ps(side_dist_ys, side_dist_ys_dys);


            __m256 prev_perp_wall_dists_zero_mask = _mm256_cmp_ps(prev_perp_wall_dists, zero_ps_vec, _CMP_EQ_UQ);
            __m256 perp_wall_dists_zero_mask = _mm256_cmp_ps(perp_wall_dists, zero_ps_vec, _CMP_EQ_UQ);

            if(perp_wall_dist == 0 || prev_perp_wall_dist == 0) {
                continue;
            }

            u32 depth = depthmap_u32s[idx];
            __m256i depths = _mm256_i32gather_epi32(depthmap_u32s, idxs, 4);
            u32 abgr = cmap[idx];
            __m256i abgrs = _mm256_i32gather_epi32(cmap, idxs, 4);

            __m256 height_vec = _mm256_set1_ps(height);
            f32 relative_height = (height-depth);
            __m256 relative_heights = _mm256_sub_ps(height_vec, _mm256_cvtepi32_ps(depths));
            
            __m256 scale_height_vec = _mm256_set1_ps(scale_height);
            __m256 swapped_prev_perp_wall_dists = _mm256_blendv_ps(prev_perp_wall_dists, one_ps_vec, prev_perp_wall_dists_zero_mask);
            __m256 swapped_perp_wall_dists = _mm256_blendv_ps(perp_wall_dists, one_ps_vec, perp_wall_dists_zero_mask);

            f32 canonical_dist = (relative_height < 0 ? prev_perp_wall_dist : perp_wall_dist);
            f32 invz = scale_height / canonical_dist;
            __m256 canonical_dists = _mm256_blendv_ps(perp_wall_dists, prev_perp_wall_dists, _mm256_cmp_ps(relative_heights, zero_ps_vec, _CMP_LT_OQ));
            __m256 invzs = _mm256_div_ps(scale_height_vec, canonical_dists);
            
            f32 float_projected_height = relative_height*invz;//*scale_height;

            __m256 float_projected_heights = _mm256_mul_ps(relative_heights, invzs);
            
        #ifdef VECTOR
            __m256 avg_dists = _mm256_div_ps(_mm256_add_ps(prev_perp_wall_dists, perp_wall_dists), two_ps_vec);
            __m256 z_to_1s = _mm256_mul_ps(avg_dists, one_over_max_z_vec);
            z_to_1s = _mm256_mul_ps(z_to_1s, z_to_1s);
            __m256 fog_factors = lerp256(zero_ps_vec, z_to_1s, one_ps_vec);
            __m256 one_minus_fogs = _mm256_sub_ps(one_ps_vec, fog_factors);
            __m256 mult_fog_rs = _mm256_mul_ps(fog_r_vec, fog_factors);
            __m256 mult_fog_gs = _mm256_mul_ps(fog_g_vec, fog_factors);
            __m256 mult_fog_bs = _mm256_mul_ps(fog_b_vec, fog_factors);

            
            __m256 rs = _mm256_cvtepi32_ps(_mm256_and_si256(abgrs, MaskFF_vec));
            __m256 gs = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(abgrs, 8), MaskFF_vec));
            __m256 bs = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(abgrs, 16), MaskFF_vec));

            rs =  _mm256_add_ps(_mm256_mul_ps(rs, one_minus_fogs), mult_fog_rs);
            gs =  _mm256_add_ps(_mm256_mul_ps(gs, one_minus_fogs), mult_fog_gs);
            bs =  _mm256_add_ps(_mm256_mul_ps(bs, one_minus_fogs), mult_fog_bs);

            __v8su Intr = (__v8su)_mm256_cvtps_epi32(rs);
            __v8su Intg = (__v8su)_mm256_cvtps_epi32(gs);
            __v8su Intb = (__v8su)_mm256_cvtps_epi32(bs);

            __v8su Sr = Intr;
            __v8su Sg = (__v8su)_mm256_slli_epi32((__m256i)Intg, 8);
            __v8su Sb = (__v8su)_mm256_slli_epi32((__m256i)Intb, 16);
            __v8su Sa = (__v8su)OpaqueAlpha;
            abgrs = _mm256_or_si256(_mm256_or_si256((__m256i)Sr, (__m256i)Sg), _mm256_or_si256((__m256i)Sb, OpaqueAlpha));
        #else

            f32 avg_dist = (prev_perp_wall_dist+perp_wall_dist)/2;
            f32 z_to_1 = (avg_dist*one_over_max_z)*(avg_dist*one_over_max_z);

            f32 fog_factor = lerp(0, z_to_1, 1);
            f32 one_minus_fog = 1-fog_factor;
            
            f32 mult_fog_r = fog_factor*fog_r;
            f32 mult_fog_g = fog_factor*fog_g;
            f32 mult_fog_b = fog_factor*fog_b;           
            f32 fr  = ((abgr&0xFF) * one_minus_fog) + mult_fog_r;
            f32 fg = (((abgr>>8)&0xFF) * one_minus_fog) + mult_fog_g;
            f32 fb = (((abgr>>16)&0xFF) * one_minus_fog) + mult_fog_b;

            uint8_t r = (uint8_t)(fr);
            uint8_t g = (uint8_t)(fg);
            uint8_t b = (uint8_t)(fb);
            abgr = (0xFF<<24)|(b<<16)|(g<<8)|r;


        #endif



        #ifdef VECTOR
            __m256i int_projected_heights = _mm256_cvtps_epi32(_mm256_floor_ps(float_projected_heights));
            __m256i heights_on_screen = _mm256_add_epi32(int_projected_heights, pitch_vec);
            heights_on_screen = _mm256_max_epi32(min_y_vec, heights_on_screen);         
            
            __m256i too_close_mask = _mm256_or_si256((__m256i)prev_perp_wall_dists_zero_mask, (__m256i)perp_wall_dists_zero_mask);
            //int too_close_mask = _mm256_movemask_ps(prev_perp_wall_dists_zero_mask) | _mm256_movemask_ps(perp_wall_dists_zero_mask);
            __m256i not_clipped_mask = _mm256_cmpgt_epi32(prev_drawn_ys, heights_on_screen);

            __m256i shown_mask = _mm256_andnot_si256(too_close_mask, not_clipped_mask);

            __v8si max_int_vec = (__v8si)_mm256_set1_epi32(SDL_MAX_SINT32);
            __v8si min_shown_heights = (__v8si)_mm256_blendv_epi8((__m256i)max_int_vec, (__m256i)heights_on_screen, (__m256i)shown_mask);
            
            s32 min_height = horizontal_min_epi32(min_shown_heights);
            __v8si min_int_vec = (__v8si)_mm256_set1_epi32(SDL_MIN_SINT32);
            __v8si max_shown_prevy = (__v8si)_mm256_blendv_epi8((__m256i)min_int_vec, (__m256i)prev_drawn_ys, (__m256i)shown_mask);
            s32 max_height = horizontal_max_epi32(max_shown_prevy);

            __v8si y_vec = (__v8si)_mm256_set1_epi32(min_height);
            __v8si heights_minus_one = (__v8si)_mm256_sub_epi32((__m256i)heights_on_screen, (__m256i)one_vec);
            u32 fb_idx = fb_swizzle(x,min_height);
            u32* ptr = &inter_buffer[fb_idx];

            for(int y = min_height; y < max_height; y++) {
                __v8si within_span =(__v8si) _mm256_cmpgt_epi32((__m256i)y_vec, (__m256i)heights_minus_one);
                __v8si not_hidden = (__v8si)_mm256_cmpgt_epi32((__m256i)prev_drawn_ys, (__m256i)y_vec);
                __v8si pixels_shown = (__v8si)_mm256_and_si256((__m256i)within_span, (__m256i)not_hidden);
                __v8su old_abgrs = (__v8su)_mm256_load_si256((__m256i*)ptr);
                __v8su new_pixels = (__v8su)_mm256_blendv_epi8((__m256i)old_abgrs, (__m256i)abgrs, (__m256i)pixels_shown);
                _mm256_store_si256((__m256i*)ptr, (__m256i)new_pixels); //abgrs);
                //ptr += oc.stride;
                ptr += 8;
                y_vec = (__v8si)_mm256_add_epi32((__m256i)y_vec, (__m256i)one_vec); 
            }      
            
            prev_drawn_ys = _mm256_blendv_epi8(prev_drawn_ys, heights_on_screen, shown_mask);
            dist_lte_max_z_mask = _mm256_movemask_ps(_mm256_cmp_ps(perp_wall_dists, max_z_vec, _CMP_LE_OQ));
            prev_draw_y_gt_min_y_mask = _mm256_movemask_epi8(_mm256_cmpgt_epi32(prev_drawn_ys, min_y_vec));      
        #else            
            
            s32 int_projected_height = floor(float_projected_height);
            s32 heightonscreen = int_projected_height + pitch;
            heightonscreen = max(min_y, heightonscreen);
            if(heightonscreen < prev_drawn_y) { // back_heightonscreen < prev_drawn_y) {
                u32 idx = fb_swizzle(x,heightonscreen); //back_heightonscreen);
                //for(int y = back_heightonscreen; y < prev_drawn_y; y++) {
                for(int y = heightonscreen; y < prev_drawn_y; y++) {
                    //u32 idx = fb_swizzle(x,y); //y*render_size+x;//m2D_e_magicbits(x,y);
                    inter_buffer[idx] = abgr;
                    idx += 8;
                }
                prev_drawn_y = heightonscreen; //back_heightonscreen;
            }
        #endif

        }
        
        xs = _mm256_add_epi32(xs, dx_vector);
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
            u32 bb = fb_swizzle(ixx, iyy);
            u32 pix = inter_buffer[bb];

            #ifdef DOUBLE
                oc.pixels[(oy*2)*(WIDTH*2)+(ox*2)] = pix;
                oc.pixels[(oy*2)*(WIDTH*2)+(ox*2)+1] = pix;
                oc.pixels[((oy*2)+1)*(WIDTH*2)+(ox*2)] = pix;
                oc.pixels[((oy*2)+1)*(WIDTH*2)+(ox*2)+1] = pix;
            #else
                oc.pixels[oy*WIDTH+(ox+i)] = pix;
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
