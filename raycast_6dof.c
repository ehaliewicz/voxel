#include "math.h"

typedef struct {
    float x,y;
} vector2;

typedef struct {
    float x,y,z,w;
} vector4;

typedef struct {
    float x,y,z;
} vector3;

typedef struct {
    float els[4][4];
} float4x4;


typedef struct {
    //float rotation;
    vector3 position;
    vector3 eulerAngles;
    vector3 forward;
    vector3 up;
    float4x4 nonJitteredProjectionMatrix;
} transform;


typedef struct {
    float near_clip_plane;
    transform transform;
    int pixel_width;
    int pixel_height;
} camera;

vector3 vec3_add(vector3 a, vector3 b) {
    return ((vector3){.x = a.x+b.x, .y = a.y+b.y, .z = a.z+b.z});
}
vector3 vec3_sub(vector3 a, vector3 b) {
    return ((vector3){.x = a.x-b.x, .y = a.y-b.y, .z = a.z-b.z});
}
vector3 vec3_scale(vector3 a, float b) {
    return ((vector3){.x = a.x*b, .y = a.y*b, .z = a.z*b});
}

vector2 vec2_add(vector2 a, vector2 b) {
    return ((vector2){.x = a.x+b.x, .y = a.y+b.y});
}
vector2 vec2_mul(vector2 a, vector2 b) {
    return ((vector2){.x = a.x*b.x, .y = a.y*b.y});
}
vector2 vec2_scale(vector2 a, float b) {
    return ((vector2){.x = a.x*b, .y = a.y*b});
}


#define DEG2RAD (3.14159f/180.0f)

vector3 vec3_up = {.x = 0.0f, .y = -1.0f, .z = 0.0f};
vector3 vec3_zero = {.x = 0.0f, .y = 0.0f, .z = 0.0f};


vector3 CalculateVanishingPointWorld (camera* cam) {
    transform t = cam->transform;
    return vec3_add(t.position, 
                    vec3_scale(vec3_up, 
                               (-cam->near_clip_plane / sinf(t.eulerAngles.x * DEG2RAD)))); // multiply pitch angle
}


float4x4 mat4x4_mul_mat4x4(float4x4* src1, float4x4* src2) {
    float4x4 dst;
    dst.els[0][0] = src1->els[0][0] * src2->els[0][0] + src1->els[0][1] * src2->els[1][0] + src1->els[0][2] * src2->els[2][0] + src1->els[0][3] * src2->els[3][0]; 
    dst.els[0][1] = src1->els[0][0] * src2->els[0][1] + src1->els[0][1] * src2->els[1][1] + src1->els[0][2] * src2->els[2][1] + src1->els[0][3] * src2->els[3][1]; 
    dst.els[0][2] = src1->els[0][0] * src2->els[0][2] + src1->els[0][1] * src2->els[1][2] + src1->els[0][2] * src2->els[2][2] + src1->els[0][3] * src2->els[3][2]; 
    dst.els[0][3] = src1->els[0][0] * src2->els[0][3] + src1->els[0][1] * src2->els[1][3] + src1->els[0][2] * src2->els[2][3] + src1->els[0][3] * src2->els[3][3]; 
    dst.els[1][0] = src1->els[1][0] * src2->els[0][0] + src1->els[1][1] * src2->els[1][0] + src1->els[1][2] * src2->els[2][0] + src1->els[1][3] * src2->els[3][0]; 
    dst.els[1][1] = src1->els[1][0] * src2->els[0][1] + src1->els[1][1] * src2->els[1][1] + src1->els[1][2] * src2->els[2][1] + src1->els[1][3] * src2->els[3][1]; 
    dst.els[1][2] = src1->els[1][0] * src2->els[0][2] + src1->els[1][1] * src2->els[1][2] + src1->els[1][2] * src2->els[2][2] + src1->els[1][3] * src2->els[3][2]; 
    dst.els[1][3] = src1->els[1][0] * src2->els[0][3] + src1->els[1][1] * src2->els[1][3] + src1->els[1][2] * src2->els[2][3] + src1->els[1][3] * src2->els[3][3]; 
    dst.els[2][0] = src1->els[2][0] * src2->els[0][0] + src1->els[2][1] * src2->els[1][0] + src1->els[2][2] * src2->els[2][0] + src1->els[2][3] * src2->els[3][0]; 
    dst.els[2][1] = src1->els[2][0] * src2->els[0][1] + src1->els[2][1] * src2->els[1][1] + src1->els[2][2] * src2->els[2][1] + src1->els[2][3] * src2->els[3][1]; 
    dst.els[2][2] = src1->els[2][0] * src2->els[0][2] + src1->els[2][1] * src2->els[1][2] + src1->els[2][2] * src2->els[2][2] + src1->els[2][3] * src2->els[3][2]; 
    dst.els[2][3] = src1->els[2][0] * src2->els[0][3] + src1->els[2][1] * src2->els[1][3] + src1->els[2][2] * src2->els[2][3] + src1->els[2][3] * src2->els[3][3]; 
    dst.els[3][0] = src1->els[3][0] * src2->els[0][0] + src1->els[3][1] * src2->els[1][0] + src1->els[3][2] * src2->els[2][0] + src1->els[3][3] * src2->els[3][0]; 
    dst.els[3][1] = src1->els[3][0] * src2->els[0][1] + src1->els[3][1] * src2->els[1][1] + src1->els[3][2] * src2->els[2][1] + src1->els[3][3] * src2->els[3][1]; 
    dst.els[3][2] = src1->els[3][0] * src2->els[0][2] + src1->els[3][1] * src2->els[1][2] + src1->els[3][2] * src2->els[2][2] + src1->els[3][3] * src2->els[3][2]; 
    dst.els[3][3] = src1->els[3][0] * src2->els[0][3] + src1->els[3][1] * src2->els[1][3] + src1->els[3][2] * src2->els[2][3] + src1->els[3][3] * src2->els[3][3]; 
    return dst;
}

vector4 mat44_mult_vec4( float4x4* src_mat, vector4* src_vec) {
    vector4 res;
    res.x = src_mat->els[0][0] * src_vec->x + src_mat->els[1][0] * src_vec->y + src_mat->els[2][0] * src_vec->z + src_mat->els[3][0] * src_vec->w;
    res.y = src_mat->els[0][1] * src_vec->x + src_mat->els[1][1] * src_vec->y + src_mat->els[2][1] * src_vec->z + src_mat->els[3][1] * src_vec->w;
    res.z = src_mat->els[0][2] * src_vec->x + src_mat->els[1][2] * src_vec->y + src_mat->els[2][2] * src_vec->z + src_mat->els[3][2] * src_vec->w;
    res.w = src_mat->els[0][3] * src_vec->x + src_mat->els[1][3] * src_vec->y + src_mat->els[2][3] * src_vec->z + src_mat->els[3][3] * src_vec->w;
    return res;
}

/*
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
*/

float4x4 mat4x4_look_at(vector3 src, vector3 dst, vector3 up) {

}


vector3 calculate_vanishing_point_world (camera cam) {
    transform t = cam.transform;
    return vec3_add(t.position, vec3_scale(vec3_up, (-cam.near_clip_plane / sinf(t.eulerAngles.x * DEG2RAD))));
}

vector2 project_vanishing_point_world_to_screen (camera cam, vector3 world_pos) {
    // does what code below does, but that one has precision issues due to the world space usage
    // set up a local space version instead
    //return ((float3)camera.WorldToScreenPoint(worldPos)).xy; < -precision issues

    float4x4 lookMatrix = mat4x4_look_at(vec3_zero, cam.transform.forward, cam.transform.up);
    float4x4 view_matrix = mat4x4_mul_mat4x4(mat4x4_scale(float3(1, 1, 1)), 
                                            inverse(lookMatrix)); // -1*z because unity
    float4x4 local_to_screen_matrix = mat4x4_mul_mat4x4(&cam.non_jittered_projection_matrix, &view_matrix);

    vector3 load_pos = vec3_sub(world_pos, cam.transform.position);
    vector4 load_pos4 = {.x=load_pos.x, .y=load_pos.y, .z=load_pos.z, .w=1.0f};
    vector4 cam_pos = mat44_mult_vec4(&local_to_screen_matrix, &load_pos4);

    vector2 screen = ((vector2){.x = cam_pos.x/cam_pos.w, .y = cam_pos.y/cam_pos.w});

    vector2 screen_size = {.x = cam.pixel_width, .y = cam.pixel_height};
    return vec2_mul(
            vec2_add(
                vec2_scale(screen, 0.5f), 
                0.5f), 
            screen_size);
}



void raycast_6dof(camera cam) {
    // Setup vanishing point


    //Profiler.BeginSample("Setup VP");
    vector3 vanishingPointWorldSpace = calculate_vanishing_point_world(cam);
    vector2 vanishingPointScreenSpace = project_vanishing_point_world_to_screen(cam, vanishingPointWorldSpace);
    //Profiler.EndSample();
}