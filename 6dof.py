#import PIL
import typing

#import PIL.PngImagePlugin
import pygame
import pygame.image

import math
import numpy as np
from dataclasses import dataclass
import pyglet


rgb_triple = typing.Tuple[int, int, int]


T = typing.TypeVar('T')
def __load_transformed_img(file: str, pix_func: typing.Callable[[int],  T]) -> typing.Generator[T, typing.Any, typing.Any]:
    #img: PIL.PngImagePlugin.PngImageFile = PIL.Image.open(f"./comanche_maps/{file}")
    img = pygame.image.load(f"./comanche_maps/{file}")
    #data = pygame.PixelArray(img) #.load()

    #print(data[0,0])
    #palette_lut = img.palette.palette if img.palette is not None else None
    palette_lut = img.get_palette()
    print(img.get_at((0,0)))
    w = img.get_width()
    h = img.get_height()
    for y in range(h):
        for x in range(w):
            idx = img.get_at((x, y)) #[x,y]
            yield pix_func(idx)



load_paletted_img = lambda f: __load_transformed_img(f, lambda x: x)
load_grayscale_img = lambda f: __load_transformed_img(f, lambda x: x[0])


OUTPUT_WIDTH = 400
OUTPUT_HEIGHT = 400
RENDER_WIDTH = 200
RENDER_HEIGHT = 200
SHADE_SIZE = 200

OUTPUT_HEIGHT = OUTPUT_HEIGHT*2


FULL_CIRCLE = 2.0 * math.pi        # 360 degrees
HALF_CIRCLE = FULL_CIRCLE/2.0      # 180 degrees
QUARTER_CIRCLE = HALF_CIRCLE/2.0   # 90 degrees
EIGHTH_CIRCLE = QUARTER_CIRCLE/2.0 # 45 degrees
THREE_QUARTERS_CIRCLE = 3.0 * QUARTER_CIRCLE
MOVE_SPEED = 7

    

sky_blue = (135, 206, 235)
hot_pink = (0xFF, 0x69, 0xB4)

type float2 = tuple[float, float]
type float3 = tuple[float, float, float]
type float4 = tuple[float, float, float, float]
type int2 = tuple[int, int]
type float4x4 = tuple[float4, float4, float4, float4]


def float3_add(f1: float3, f2: float3) -> float3:
    return (f1[0] + f2[0], f1[1] + f2[1], f1[2] + f2[2])

def float3_scale(f1: float3, f2: float) -> float3:
    return (f1[0] * f2, f1[1] * f2, f1[2] * f2)


def frac(f: float) -> float: 
    return f - math.floor(f)

def cmax(f: float2) -> float:
    return max(f[0], f[1])

def cmin(f: float2) -> float:
    return min(f[0], f[1])


def clamp(a,mi,ma):
    return min(max(a,mi), ma)

def yaw_3d(x,y,z, siny, cosy):
    
    # working versions!
    # supposed to be cosy, siny, then siny, cosy, but this works instead lol
    ytx = x*cosy + z*siny
    ytz = -x*siny + z*cosy

    return ytx, y, ytz

def yaw_matrix(siny, cosy):
    return np.array([
         [siny, 0, cosy, 0],
         [0, 1, 0, 0],
         [-cosy, 0, siny, 0],
         [0, 0, 0, 1]
    ])

def pitch_3d(x,y,z, sinp, cosp):
    pty = y * cosp - z*sinp
    ptz = y * sinp + z*cosp
    return x, pty, ptz


def pitch_matrix(sinp, cosp):
    return np.array([
        [1, 0, 0, 0],
        [0, cosp, -sinp, 0],
        [0, sinp, cosp, 0],
        [0, 0, 0, 1]
    ])

def rot_matrix(sinp, cosp, siny, cosy):
    return np.matmul( yaw_matrix(siny, cosy), pitch_matrix(sinp, cosp))

def scale_matrix(x, y, z):
    return np.array([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]
    ])

def translate_matrix(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])


def proj_matrix(fov, front, back):
    aspect = RENDER_WIDTH/RENDER_HEIGHT
    scale = 1/math.tan(fov/2)
    right = front * scale
    top = right / aspect
    return np.transpose(np.array([
        [front/right,0,0,0],
        [0,front/top,0,0],
        [0,0,-(back + front) / (back - front),-1],
        [0,0,-(2 * back * front) / (back - front),0]
        #[scale, 0, 0, 0],
        #[0, scale, 0, 0],
        #[0, 0, -(back/(back-front)), -1],
        #[0, 0, -((back*front)/(back-front)), 0]
    ]))


def world_to_camera_matrix(pitch, yaw, roll, px, py, pz):
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    Ry = np.array([
        [cos_yaw, 0, -sin_yaw, 0],
        [0, 1, 0, 0],
        [sin_yaw, 0, cos_yaw, 0],
        [0, 0, 0, 1]
    ])
    #Ry = pyglet.math.Mat4.from_rotation(
    #    yaw
    #    pyglet.math.Vec3(0,1,0)
    #)
    #Ry = yaw_matrix(sin_yaw, cos_yaw)
    
    # Pitch (X-axis rotation)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    Rx = np.array([
        [1, 0, 0, 0],
        [0, cos_pitch, sin_pitch, 0],
        [0, -sin_pitch, cos_pitch, 0],
        [0, 0, 0, 1]
    ])
    
    # Roll (Z-axis rotation)
    #cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    #Rz = np.array([
    #    [cos_roll, -sin_roll, 0, 0],
    #    [sin_roll, cos_roll, 0, 0],
    #    [0, 0, 1, 0],
    #    [0, 0, 0, 1]
    #])
    
    #R = Ry # @ Rx # @ Rz
    R = Rx @ Ry

    T = np.array([
        [1, 0, 0, -px],  # Note the negation
        [0, 1, 0, -py],  # These are negated because we're
        [0, 0, 1, -pz],  # moving the world relative to camera
        [0, 0, 0, 1]
    ])

    mat = R @ T
    # needed for yaw?
    for i in range(4):
        mat[2][i] = -mat[2][i]
    return mat




@dataclass
class Ray:
    position: int2
    step: int2
    start: float2
    dir: float2
    t_delta: float2
    t_max: float2
    intersection_distances: float2


def make_ray(start: float2, dir: float2) -> Ray:
    position = [math.floor(start[0]), math.floor(start[1])]
    eps = .0000001
    t_delta = ((1 / max(eps, abs(dir[0]))), (1 / max(eps, abs(dir[1]))))
    sign_dir = (sign(dir[0]), sign(dir[1]))
    step = (int(sign_dir[0]), int(sign_dir[1]))
    t_max_x = (sign_dir[0] * -frac(start[0]) + (sign_dir[0] * 0.5) + 0.5) * t_delta[0]
    t_max_y = (sign_dir[1] * -frac(start[1]) + (sign_dir[1] * 0.5) + 0.5) * t_delta[1]
    t_max = [t_max_x, t_max_y]
    intersection_distances = (cmax((t_max[0] - t_delta[0], t_max[1] - t_delta[1])), cmin((t_max[0], t_max[1])))
    return Ray(position=position, step=step, start=start, dir=dir, t_delta=t_delta, t_max=t_max, intersection_distances=intersection_distances)



@dataclass
class CameraData:
    world_to_screen_matrix: float4x4
    position_xz: float2
    position_y: float
    inverse_element_iteration_direction: bool
    far_clip: float
    screen: int2

def make_camera(farClip, pitch, yaw, roll, pos_x, pos_y, pos_z, screen_dims):

    world_to_cam_matrix = world_to_camera_matrix(pitch, yaw, roll, pos_x, pos_y, pos_z)
    cam_to_screen_matrix = proj_matrix(QUARTER_CIRCLE, NEAR_CLIP_PLANE, FAR_CLIP_PLANE)

    world_to_screen_matrix = np.dot(cam_to_screen_matrix, world_to_cam_matrix)
    world_to_screen_matrix = np.dot(
        scale_matrix(.5, .5, 1), world_to_screen_matrix) # scale from -1,1 to -0.5,.5
    world_to_screen_matrix = np.dot(
        translate_matrix(0.5, 0.5, 1), world_to_screen_matrix) # translate from -0.5,.5 to 0,1
    world_to_screen_matrix = np.dot(
        scale_matrix(screen_dims[0], screen_dims[1], 1), world_to_screen_matrix); # scale from 0,1 to 0,screen


    return CameraData(
        world_to_screen_matrix=tuple([tuple([float(f) for f in row]) for row in world_to_screen_matrix]),
        position_xz=(pos_x,pos_z),
        position_y=pos_y,#
        inverse_element_iteration_direction = pitch_ang <= 0.0,
        far_clip = farClip,
        screen=[RENDER_WIDTH,RENDER_HEIGHT]
    )


def project_to_homogeneous_camera_space(camera: CameraData, world_a: float3) -> float4:
        ((m00,m01,m02,m03),
         (m10,m11,m12,m13),
         (m20,m21,m22,m23),
         (m30,m31,m32,m33)) = camera.world_to_screen_matrix
        
        x,y,z = world_a 
        nx = m00*x + m01*y + m02*z + m03
        ny = m10*x + m11*y + m12*z + m13
        nz = m20*x + m21*y + m22*z + m23
        nw = m30*x + m31*y + m32*z + m33 
        return (nx,ny,nz,nw)
        #return np.dot(camera.world_to_screen_matrix, (world_a[0], world_a[1], world_a[2], 1))


def project_vector_to_homogeneous_camera_space(camera: CameraData, world_a: float3) -> float4:
        ((m00,m01,m02,m03),
         (m10,m11,m12,m13),
         (m20,m21,m22,m23),
         (m30,m31,m32,m33)) = camera.world_to_screen_matrix
        
        x,y,z = world_a 
        nx = m00*x + m01*y + m02*z# + m03
        ny = m10*x + m11*y + m12*z# + m13
        nz = m20*x + m21*y + m22*z# + m23
        nw = m30*x + m31*y + m32*z# + m33 
        return (nx,ny,nz,nw)
        #return np.dot(camera.world_to_screen_matrix, (world_a[0], world_a[1], world_a[2], 0))



def setup_projected_plane_params(
        camera: CameraData,
        ray: Ray,
        world_max_y: float,
        voxel_scale: int,
        y_axis: int):
    start = ray.start
    plane_start_bottom = (start[0], 0.0, start[1])
    plane_start_top = (start[0], world_max_y, start[1])
    plane_ray_direction = (ray.dir[0], 0.0, ray.dir[1])

    full_plane_start_top_projected = project_to_homogeneous_camera_space(camera, plane_start_top)
    full_plane_start_bot_projected = project_to_homogeneous_camera_space(camera, plane_start_bottom)
    full_plane_ray_direction_projected = project_vector_to_homogeneous_camera_space(camera, plane_ray_direction)
    if y_axis == 0:
        return (
             (full_plane_start_bot_projected[0], full_plane_start_bot_projected[2], full_plane_start_bot_projected[3]),
             (full_plane_start_top_projected[0], full_plane_start_top_projected[2], full_plane_start_top_projected[3]),
             (full_plane_ray_direction_projected[0], full_plane_ray_direction_projected[2], full_plane_ray_direction_projected[3])
        )
    else:
        return (
             (full_plane_start_bot_projected[1], full_plane_start_bot_projected[2], full_plane_start_bot_projected[3]),
             (full_plane_start_top_projected[1], full_plane_start_top_projected[2], full_plane_start_top_projected[3]),
             (full_plane_ray_direction_projected[1], full_plane_ray_direction_projected[2], full_plane_ray_direction_projected[3])
        )


def lerp(a:float,b:float,i:float):
    return a + (b-a)*i

def unlerp(a:float,b:float,v:float):
    return (v-a) / (b-a)

def clip_homogeneous_camera_space_line(a: float3, b: float3) -> tuple[bool, float3, float3]:
    # near-plane clipping
    (ax,ay,az) = a
    (bx,by,bz) = b

    if (ay < 0):
        if (by < 0):
            return False, a, b
        v = b[1] / (b[1] - a[1])
        
        ax = lerp(b[0], a[0], v)
        az = lerp(b[2], a[2], v)
        return True, (ax,ay,az), b
    elif (b[1] < 0):

        v = a[1] / (a[1] - b[1])
        bx = lerp(a[0], b[0], v)
        bz = lerp(a[2], b[2], v)
        return True, a, (bx,by,bz)
    else:
        return True, a, b


@dataclass
class Segment:
    min_screen: float
    max_screen: float
    cam_local_plane_ray_min: float
    cam_local_plane_ray_max: float
    ray_count: int


                        # B G R A
skybox_col_bytes = bytes([235,206,135,0xFF])
def draw_skybox(next_free_pix_min, next_free_pix_max, seen_pixel_cache, ray_buf, ray_column_off):
    for y in range(next_free_pix_min, next_free_pix_max):
        pix_off = (ray_column_off+y)*4
        if seen_pixel_cache[y] == 0:
            ray_buf.write(skybox_col_bytes, pix_off)

def execute_rays(
        px, py, pz, 
        pitch, roll, yaw, camera: CameraData, axis_mapped_to_y,
        segment: Segment, segment_ray_index_offset,
        next_free_pix_min, next_free_pix_max, seen_pixel_cache_length,
        height_map, color_map, pix_arr: pygame.BufferProxy, surf_width: int):
    
    #pass
    ray_draw_height = next_free_pix_max - next_free_pix_min
    if segment.ray_count == 0:
        return

    #print(pix_arr.shape)
    #surf_width,surf_height = pix_arr.shape
    seen_pixel_cache = [0 for _ in range(seen_pixel_cache_length)]

    for i in range(segment.ray_count):

        cur_next_free_pix_min = next_free_pix_min
        cur_next_free_pix_max = next_free_pix_max

        end_ray_lerp = i / segment.ray_count
        
        cam_local_plane_ray_direction_x = lerp(
            segment.cam_local_plane_ray_min[0],
            segment.cam_local_plane_ray_max[0],
            end_ray_lerp
        )
        cam_local_plane_ray_direction_y = lerp(
            segment.cam_local_plane_ray_min[1],
            segment.cam_local_plane_ray_max[1],
            end_ray_lerp
        )
        cam_local_plane_ray_direction = ( cam_local_plane_ray_direction_x, cam_local_plane_ray_direction_y )
        norm_ray_dir = normalize_float2(cam_local_plane_ray_direction)
        ray = make_ray(camera.position_xz, (norm_ray_dir[0], norm_ray_dir[1]))

        #int voxelScale = 1 << lod;
        far_clip = 128  #FAR_CLIP_PLANE
        voxel_scale = 1

        world_max_y = WORLD_MAX_Y
        camera_pos_y_normalized = camera.position_y / world_max_y

        # small offset to the frustums to prevent a division by zero in the clipping algorithm
        frustum_bounds_min = cur_next_free_pix_min - .501
        frustum_bounds_max = cur_next_free_pix_max + .501

        (plane_start_bottom_projected,
        plane_start_top_projected,
        plane_ray_direction_projected) = setup_projected_plane_params(
            camera, ray, world_max_y, voxel_scale, axis_mapped_to_y
        )


        base_pix_off = surf_width*i

        intersection_distances = ray.intersection_distances
        intersection_distances_x, intersection_distances_y = ray.intersection_distances

        step_x, step_y = ray.step
        t_delta_x, t_delta_y = ray.t_delta
        t_max_x, t_max_y = ray.t_max
        position_x, position_y = ray.position
        
        cam_inv = camera.inverse_element_iteration_direction
        while True:
            if(intersection_distances_x >= far_clip):
            #if cur_next_free_pix_min < cur_next_free_pix_max:
                draw_skybox(cur_next_free_pix_min, cur_next_free_pix_max, seen_pixel_cache, pix_arr, base_pix_off)
                
                break # no lod stuff :)
            
            #if cam_inv:
            #    element_bounds_min = world_max_y
            #    element_bounds_max = world_max_y
            #else:
            element_bounds_min = 0
            element_bounds_max = 0

            #world_column = get_voxel_column(ray.position, height_map, color_map)
            index = (position_y&1023)*1024+(1024-(position_x&1023))
            height = world_max_y - height_map[index]
            color = color_map[index]
            color_bytes = bytes((color[2], color[1], color[0], color[3]))
            cam_space_min_last = float3_add(
                plane_start_bottom_projected, 
                float3_scale(plane_ray_direction_projected, intersection_distances_x)
            )
			#cam_space_min_next = plane_start_bottom_projected + plane_ray_direction_projected * ray.intersection_distances[1];

            cam_space_max_last = float3_add(
                plane_start_top_projected, 
                float3_scale(plane_ray_direction_projected, intersection_distances_x)
            )
			#cam_space_max_next = plane_start_top_projected + plane_ray_direction_projected * ray.IntersectionDistances[1];


            #if world_column.column_runs == -1:
            #    draw_skybox(cur_next_free_pix_min, cur_next_free_pix_max, seen_pixel_cache)
            #    return
            
            #if world_column.column_runs == 0:
            #    step_ray(ray, far_clip)

            #if cam_inv: #camera.inverse_element_iteration_direction:
                # max will be 256
            #    element_bounds_max = element_bounds_min
            #    element_bounds_min = element_bounds_min - height
            #else:
                # min will be 0
                # max will be 0 + 256-height
                #element_bounds_min = element_bounds_max
                #element_bounds_max = element_bounds_min + (world_max_y - height)
            element_bounds_min = element_bounds_max
            element_bounds_max = element_bounds_min + (world_max_y - height)
            

            #portion_bottom = unlerp(0, world_max_y, element_bounds_min)
            #portion_top = unlerp(0, world_max_y, element_bounds_max)
            #(v-a) / (b-a)
            one_over_world_max_y = 1/world_max_y 
            portion_top = element_bounds_max * one_over_world_max_y
            portion_bottom = element_bounds_min * one_over_world_max_y


            cam_space_front_bottom_x = lerp(cam_space_min_last[0], cam_space_max_last[0], portion_bottom)
            cam_space_front_bottom_y = lerp(cam_space_min_last[1], cam_space_max_last[1], portion_bottom)
            cam_space_front_bottom_z = lerp(cam_space_min_last[2], cam_space_max_last[2], portion_bottom)
            cam_space_front_top_x = lerp(cam_space_min_last[0], cam_space_max_last[0], portion_top)
            cam_space_front_top_y = lerp(cam_space_min_last[1], cam_space_max_last[1], portion_top)
            cam_space_front_top_z = lerp(cam_space_min_last[2], cam_space_max_last[2], portion_top)
            
            onscreen, cam_space_front_bottom, cam_space_front_top = clip_homogeneous_camera_space_line(
                (cam_space_front_bottom_x, cam_space_front_bottom_y, cam_space_front_bottom_z),
                (cam_space_front_top_x, cam_space_front_top_y, cam_space_front_top_z)
            )

            if onscreen:
                ray_buffer_bounds_float_min = cam_space_front_bottom[0] / cam_space_front_bottom[2]
                ray_buffer_bounds_float_max = cam_space_front_top[0] / cam_space_front_top[2]
                #ray_buffer_bounds_float_min = cam_space_front_bottom_x / cam_space_front_bottom_z
                #ray_buffer_bounds_float_max = cam_space_front_top_x / cam_space_front_top_z

                # clip homogeneous camera space line


                if ray_buffer_bounds_float_max < ray_buffer_bounds_float_min:
                    ray_buffer_bounds_float_min, ray_buffer_bounds_float_max = ray_buffer_bounds_float_max, ray_buffer_bounds_float_min

                ray_buffer_bounds_min = int(ray_buffer_bounds_float_min)
                ray_buffer_bounds_max = int(ray_buffer_bounds_float_max)

                if (ray_buffer_bounds_max >= next_free_pix_min and ray_buffer_bounds_float_min <= next_free_pix_max):
                    
                    # shrink top of frustum as much as possible
                    if ray_buffer_bounds_min <= cur_next_free_pix_min:
                        ray_buffer_bounds_min = cur_next_free_pix_min
                        
                        if ray_buffer_bounds_max >= cur_next_free_pix_min:
                            cur_next_free_pix_min = ray_buffer_bounds_max+1

                            while cur_next_free_pix_min <= next_free_pix_max and seen_pixel_cache[cur_next_free_pix_min] > 0:
                                cur_next_free_pix_min += 1

                    # shrink bottom frustum as much as possible
                    if ray_buffer_bounds_max >= cur_next_free_pix_max:
                        ray_buffer_bounds_max = cur_next_free_pix_max
                        if ray_buffer_bounds_min <= cur_next_free_pix_max:
                            cur_next_free_pix_max = ray_buffer_bounds_min-1
                            while cur_next_free_pix_max >= next_free_pix_min and seen_pixel_cache[cur_next_free_pix_max] > 0:
                                cur_next_free_pix_max -= 1


                    for y in range(ray_buffer_bounds_min, ray_buffer_bounds_max+1):
                        if seen_pixel_cache[y] == 0:
                            seen_pixel_cache[y] = 1
                            pix_off = base_pix_off + y
                            pix_arr.write(color_bytes, pix_off*4)

                    


                
                if cur_next_free_pix_min > cur_next_free_pix_max:
                    break
            

            if (t_max_x < t_max_y):
                crossed_boundary_distance = t_max_x
                t_max_x += t_delta_x
                position_x += step_x
            else:
                crossed_boundary_distance = t_max_y
                t_max_y += t_delta_y
                position_y += step_y

            intersection_distances_x = crossed_boundary_distance
            #intersection_distances_y = cmin(t_max)

            #intersection_distances[1] = cmin(t_max)
            #if step_ray(t_max, position, t_delta, step, intersection_distances, far_clip):
            #    break
            
        for y in range(seen_pixel_cache_length):
            seen_pixel_cache[y] = 0
        #seen_pixel_cache = [0 for _ in range(seen_pixel_cache_length)]

def rads_to_degrees(r):
    return r * 57.2

def step(edge, x):
    if x < edge:
        return 0.0
    return 1.0

def shade_screen(screen_draw_surf: pygame.Surface, seg_draw_surf: pygame.Surface, seg_offsets, rot_x_greater_zero, shader_vp):
    shader_vpx, shader_vpy = shader_vp
    screen_pix = pygame.PixelArray(screen_draw_surf)
    #seg_draw_surf_pix = pygame.PixelArray(seg_draw_surf)
    shader_vpx /= SHADE_SIZE
    shader_vpy /= SHADE_SIZE
    try:
        
        for y in range(SHADE_SIZE):
            scy = y/SHADE_SIZE
            dy = scy - shader_vpy
            #upper = step(scy1, 0.0)

            for x in range(SHADE_SIZE):
                scx = x/SHADE_SIZE

                #border = (SHADE_SIZE-SHADE_SIZE) / (SHADE_SIZE*2)
                dx = scx - shader_vpx

                if dy > 0:
                    if dx > dy: 
                        quad = 0
                        #col = (0,0,255)
                    else:
                        if -dx > dy:
                            # quadrant 3
                            quad = 2
                            #col = (0,128, 128)
                        else:
                            # quadrant 4
                            quad = 3
                            #col = (128,128,0)
                    # below the vanishing point
                else:
                    if dy > dx:
                        # quadrant
                        quad = 2
                        #col = (0, 128, 128)
                    else:
                        if -dy > dx:
                            # quadrant 2
                            quad = 1
                            #col = (0,255,0)
                        else:
                            quad = 0
                            #col = (255,0,0)

                col = (0,0,0)
                
                if quad == 1:
                    pass
                    u = clamp(.5 + (scx - shader_vpx) * abs(scy-shader_vpy), 0, .99999) * SHADE_SIZE + seg_offsets[quad]
                    v = clamp(2*((1 - scy - shader_vpy)), 0, .99999) * SHADE_SIZE
                    #col = seg_draw_surf.get_at((int(u), int(v)))
                elif quad == 3:
                    u = clamp(0.5 + (scx - shader_vpx) * abs(scy-shader_vpy), 0, .99999) * SHADE_SIZE + seg_offsets[quad]
                    v = clamp((scy - shader_vpy), 0, .99999) * SHADE_SIZE
                    col = seg_draw_surf.get_at((int(u), int(v)))
                    #col = (255,0,0)
                    #col = (0,0,0)
                elif quad == 0:
                    u = clamp((scy-shader_vpy) * abs(scx-shader_vpx), 0, .99999) * SHADE_SIZE + seg_offsets[quad]
                    v =  clamp((scx - shader_vpx), 0, .99999) * SHADE_SIZE
                    col = (0,0,0)
                elif quad == 2:
                    u = clamp((scy-shader_vpy) * abs(scx-shader_vpx), 0, .99999) * SHADE_SIZE + seg_offsets[quad]
                    v =  clamp(1 - (scx - shader_vpx), 0, .99999) * SHADE_SIZE
                    col = (0,0,0)
                    #col = (128, 0, 128)
                else:
                    pass
                

                #screen_draw_surf.set_at((x,y), col) # col)
                #screen_draw_surf.set_at((x,y), col) # col)
                screen_pix[x,y] = col #seg_draw_surf_pix[int(u), int(v)]
    finally:
        #pass
        screen_pix.close()
        #seg_draw_surf_pix.close()      
        #
        #      


pitch_ang = 1.53589 #-QUARTER_CIRCLE #QUARTER_CIRCLE
yaw_ang = 2.843782 #.1619
roll_ang = 0


pos_x = 512
pos_z = 512
pos_y = 153.6

    
def handle_input(keys_down):
    global pitch_ang, yaw_ang, roll_ang
    global pos_x, pos_z, pos_y
    if pygame.K_z in keys_down:
        pitch_ang -= .06 #min(pitch_ang+.06, QUARTER_CIRCLE)
        pitch_ang %= FULL_CIRCLE
    elif pygame.K_x in keys_down:
        pitch_ang += .06 #max(-(QUARTER_CIRCLE+EIGHTH_CIRCLE), pitch_ang - .06)
        pitch_ang %= FULL_CIRCLE


    #if pitch_ang == 0:
    #    pitch_ang += 0.01
    forward_y_for_pitch = math.sin(pitch_ang)
    if abs(forward_y_for_pitch) < 0.001:
        forward_y_for_pitch = sign(pitch_ang) * .001
        pitch_ang = math.atan2(forward_y_for_pitch, 1.0)
        pitch_ang %= FULL_CIRCLE


    if pygame.K_LEFT in keys_down:
        yaw_ang += .08
        yaw_ang %= FULL_CIRCLE
    elif pygame.K_RIGHT in keys_down:
        yaw_ang -= .08
        yaw_ang %= FULL_CIRCLE

    if pygame.K_q in keys_down:
        roll_ang += .08
        roll_ang %= FULL_CIRCLE
    elif pygame.K_f in keys_down:
        roll_ang -= .08
        roll_ang %= FULL_CIRCLE
    #print(roll_ang*math.pi*2/360)
    if pygame.K_c in keys_down:
        pos_y += MOVE_SPEED
    elif pygame.K_v in keys_down:
        pos_y -= MOVE_SPEED

    if pygame.K_UP in keys_down:
        pos_x += math.cos(yaw_ang)*MOVE_SPEED
        pos_z += math.sin(yaw_ang)*MOVE_SPEED
        #pos_z += math.sin(pitch_ang)*MOVE_SPEED
    elif pygame.K_DOWN in keys_down:
        pos_x -= math.cos(yaw_ang)*MOVE_SPEED
        pos_z -= math.sin(yaw_ang)*MOVE_SPEED
        #pos_z -= math.sin(pitch_ang)*MOVE_SPEED

NEAR_CLIP_PLANE = .05
FAR_CLIP_PLANE = 2048 # 1024 * 2

def calc_vanishing_point_world():
    global pos_x, pos_z, pos_y
    global pitch_ang
    pos_x, pos_z + 1, pos_y
    rot_y = 1 * (-NEAR_CLIP_PLANE / math.sin(pitch_ang))

    return pos_x, pos_y + rot_y, pos_z

def project_vanishing_point_world_to_screen(vp_world):
    global pitch_ang
    (wx, wy, wz) = vp_world
    #rot_matrix = rot_mat(pitch_ang, yaw_ang, roll_ang)
    lx = wx - pos_x
    ly = wy - pos_y
    lz = wz - pos_z

    (cx, cy, cz) = pitch_3d(lx, ly, lz, math.sin(-pitch_ang), math.cos(-pitch_ang))
    sx = (cx / cz * .5 + .5) * RENDER_WIDTH
    sy = (cy / cz * .5 + .5) * RENDER_HEIGHT
    return sx, sy

WORLD_MAX_Y = 256

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def vec2_dot(lhs, rhs): 
    return lhs[0] * rhs[0] + lhs[1] * rhs[1];

k_epsilon_normal_sqrt = .000000000000001
def vec2_angle(frm, to):
    frm_x, frm_y = frm
    to_x, to_y = to
    frmSqrMag = frm_x*frm_x + frm_y*frm_y
    toSqrMag = to_x*to_x + to_y*to_y
    denom = math.sqrt(frmSqrMag * toSqrMag)
    if denom < k_epsilon_normal_sqrt:
        return 0
    dot = clamp(vec2_dot(frm, to) / denom, -1, 1)
    return math.acos(dot) * 57.29
    

def vec2_signed_angle(frm, to):
    frm_x, frm_y = frm
    to_x, to_y = to
    unsigned_angle = vec2_angle(frm, to)
    sgn = sign(frm_x * to_y - frm_y * to_x)
    return unsigned_angle * sgn

def lookAt(center, forward, up, right):

    return np.array([
        [right[0], right[1], right[2], -center[0] ],
        [up[0], up[1], -up[2], -center[1]],
        [forward[0], forward[1], forward[2], -center[2]],
        [0, 0, 0, 1]
    ])
    return m

def normalize(v):

    
    norm=np.linalg.norm(np.array(v))
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def normalize_float2(v: float2) -> float2:
    
    norm=np.linalg.norm(np.array(v))
    if norm==0:
        norm=np.finfo(v.dtype).eps
    one_over_norm = 1/float(norm)
    return (v[0]*one_over_norm, v[1]*one_over_norm)

def transform_pixel(pix):
    (x, y) = pix
    scaled_pix = (x / RENDER_WIDTH), (y / RENDER_HEIGHT)
    off_pix = scaled_pix[0]-0.5, scaled_pix[1]-0.5
    homogenoeous_pix = (off_pix[0]*2, off_pix[1]*2, 1, 1)


    matrix = np.linalg.inv(
        proj_matrix(QUARTER_CIRCLE, NEAR_CLIP_PLANE, FAR_CLIP_PLANE)
    )
    inverse_proj_pix = matrix.dot(homogenoeous_pix)
    
    forward = normalize(np.array([
        math.sin(yaw_ang)*math.cos(-pitch_ang), 
        math.sin(-pitch_ang),
        math.cos(yaw_ang)*math.cos(-pitch_ang), 
    ]))
    #if pitch_ang > QUARTER_CIRCLE:
    #    forward[1] *= -1
    temp_up = np.array([0,1,0])
    #right = (math.sin(QUARTER_CIRCLE-yaw_ang), 0, math.cos(QUARTER_CIRCLE-yaw_ang))
    right = normalize(np.cross(temp_up, forward))


    up = normalize(np.cross(forward, right))

    inverse_scale_pix = np.dot(np.linalg.inv(np.array(
        [[1,0,0,0],
         [0,1,0,0],
         [0,0,-1,0],
         [0,0,0,1]])), inverse_proj_pix)
    lookat_matrix = lookAt(np.array([0,0,0]), forward, up, right)
    world_pix = lookat_matrix.dot(inverse_scale_pix)

    world_pix_div_w = np.array([world_pix[0],world_pix[2]]) / world_pix[3]
    
    return (float(world_pix[0]/world_pix[3]), float(world_pix[2]/world_pix[3]))



def dist2d(x1,y1, x2,y2):
    dx = abs(x2-x1)
    dy = abs(y2-y1)
    return math.sqrt(dx*dx+dy*dy)



def get_segment_parameters(screen_dims: typing.Tuple[float, float],
                           screen_vp: typing.Tuple[float, float], dist_to_other_end: float, 
                           neutral: typing.Tuple[float, float], primary_axis: int, world_y_max: int,
                           screen_draw_surf: pygame.Surface):
    secondary_axis = 1 - primary_axis

    segment = Segment(0.0, 0.0, 0.0, 0.0, 0)

    # setup the end points for the 2 45 degree angle rays
    scmin = screen_vp[secondary_axis] - dist_to_other_end
    simple_case_min = [scmin, scmin]
    scmax = screen_vp[secondary_axis] + dist_to_other_end
    simple_case_max = [scmax, scmax]
    a = screen_vp[primary_axis] + dist_to_other_end * sign(neutral[primary_axis])
    simple_case_min[primary_axis] = a
    simple_case_max[primary_axis] = a


    if simple_case_max[secondary_axis] <= 0 or simple_case_min[secondary_axis] >= screen_dims[secondary_axis]:
        return segment
    screen_vpx, screen_vpy = screen_vp
    screen_w, screen_h = screen_dims
    if (screen_vpx >= 0 and screen_vpy >= 0 and screen_vpx <= screen_w and screen_vpy <= screen_h):
        # vp within bounds, so nothing to clamp angle wise
        segment.min_screen = simple_case_min
        segment.max_screen = simple_case_max
    else:
        # vp outside of bounds, so we want to check if we can clamp the segment to the screen area
        # to prevent wasting buffer space

        dir_simple_middle_x = lerp(simple_case_min[0], simple_case_max[0], 0.5) - screen_vpx
        dir_simple_middle_y = lerp(simple_case_min[1], simple_case_max[1], 0.5) - screen_vpy

        angle_left = 90
        angle_right = -90
        dir_left = [0,0]
        dir_right = [0,0]

        vectors = [(0,0), (0, screen_dims[1]), (screen_dims[0], 0), screen_dims]

        for i in range(4):
            dirvec = [vectors[i][0] - screen_vpx, vectors[i][1] - screen_vpy]
            scaled_end_x = dirvec[0] * (dist_to_other_end / abs(dirvec[primary_axis]))
            scaled_end_y = dirvec[1] * (dist_to_other_end / abs(dirvec[primary_axis]))
            angle = vec2_signed_angle(neutral, dirvec)
            if angle < angle_left:
                angle_left = angle
                dir_left[0] = scaled_end_x
                dir_left[1] = scaled_end_y
            if angle > angle_right:
                angle_right = angle
                dir_right[0] = scaled_end_x
                dir_right[1] = scaled_end_y

        corner_left = [dir_left[0] + screen_vpx, dir_left[1] + screen_vpy]

        corner_right = [dir_right[0] + screen_vpx, dir_right[1] + screen_vpy]

        if angle_left < -45:
            # fallback to whatever the simple case left corner was
            sgn = vec2_signed_angle((dir_simple_middle_x, dir_simple_middle_y), simple_case_max)
            if sgn > 0:
                corner_left = simple_case_min
            else:
                corner_left = simple_case_max
            
        if angle_right > 45:
            sgn = vec2_signed_angle((dir_simple_middle_x, dir_simple_middle_y), simple_case_max)
            if sgn < 0:
                corner_right = simple_case_min 
            else:
                corner_right = simple_case_max

        pygame.draw.line(screen_draw_surf, (255, 0, 0), screen_vp, corner_left)
        pygame.draw.line(screen_draw_surf, (255, 0, 0), screen_vp, corner_right)
    
        swap = corner_left[secondary_axis] > corner_right[secondary_axis]

        # todo is this correct?
        segment.min_screen = corner_right if swap else corner_left
        segment.max_screen = corner_left if swap else corner_right

    segment.cam_local_plane_ray_min = transform_pixel(segment.min_screen)
    segment.cam_local_plane_ray_max = transform_pixel(segment.max_screen)

    segment.ray_count = round(segment.max_screen[secondary_axis] - segment.min_screen[secondary_axis])
    segment.ray_count = max(segment.ray_count, 0)


    return segment



def raycast_segments(
    segments, vanishing_point_screen_space, camera,
    heights, colors, top_down_draw_surf, left_right_draw_surf):
    
    vp_x, vp_y = vanishing_point_screen_space

    for segment_index in range(4):
        if segments[segment_index].ray_count == 0:
            continue
        segment_ray_index_offset = 0
        if segment_index == 1:
            segment_ray_index_offset = segments[0].ray_count
        if segment_index == 3:
            segment_ray_index_offset = segments[1].ray_count
        
        next_free_pixel_min = None
        next_free_pixel_max = None
        # 0,1 are mapped to x, 2,3 are mapped to y
        axis_mapped_to_y = 0 if segment_index > 1 else 1

        if segment_index < 2:
            if segment_index == 0:
                next_free_pixel_min = clamp(round(vp_y), 0, RENDER_HEIGHT-1)
                next_free_pixel_max = RENDER_HEIGHT - 1
            else:
                next_free_pixel_min = 0
                next_free_pixel_max = clamp(round(vp_y), 0, RENDER_HEIGHT-1)
        else:
            if segment_index == 3:
                next_free_pixel_min = 0
                next_free_pixel_max = clamp(round(vp_x), 0, RENDER_WIDTH-1)
            else:
                next_free_pixel_min = clamp(round(vp_x), 0, RENDER_WIDTH-1)
                next_free_pixel_max = RENDER_WIDTH-1
        
        seen_pixel_cache_length = [RENDER_WIDTH, RENDER_HEIGHT][axis_mapped_to_y]


        if segment_index < 2:
            draw_surf = top_down_draw_surf
        else:
            draw_surf = left_right_draw_surf

    
        #pix_arr = pygame.PixelArray(draw_surf)
        try:
            view = draw_surf.get_view('1')
            execute_rays(
                pos_x, pos_y, pos_z,
                pitch_ang, roll_ang, yaw_ang, camera, axis_mapped_to_y,
                segments[segment_index], segment_ray_index_offset,
                next_free_pixel_min, next_free_pixel_max, seen_pixel_cache_length,
                heights, colors, view, draw_surf.get_width(),
            )
        finally:
            
            pass
            #pix_arr.close()



def main():
    color_pix = load_paletted_img("C1W.png")
    height_pix = load_grayscale_img("D1.png")

    #print(next(color_pix))
    colors = [x for x in color_pix]
    heights = [x for x in height_pix]

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    running = True



    #top_down_draw_surf = Surface((RENDER_SIZE))
    left_right_draw_surf = pygame.Surface((RENDER_WIDTH, 2*RENDER_WIDTH + RENDER_HEIGHT))
    top_down_draw_surf = pygame.Surface((RENDER_HEIGHT*2, RENDER_WIDTH + 2*RENDER_HEIGHT))
    seg1_draw_surf = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT))

    last_fps = 0
    keys_down = set()
    blit_surf = pygame.display.get_surface()
    while running:
            # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                keys_down.add(event.key)

            elif event.type == pygame.KEYUP:
                if event.key in keys_down:
                    keys_down.remove(event.key)

        handle_input(keys_down)



        top_down_draw_surf.fill(hot_pink)
        left_right_draw_surf.fill(hot_pink)


        world_vpx, world_vpy, world_vpz = calc_vanishing_point_world()
        screen_vp = project_vanishing_point_world_to_screen((world_vpx, world_vpy, world_vpz))
        shader_vp  = screen_vp


        default_seg = Segment(0, 0, 0, 0, 0)
        segments = [default_seg, default_seg, default_seg, default_seg]
        
        screen_vpx, screen_vpy = screen_vp
        segment_vpy = screen_vpy
        segment_vpx = screen_vpx 
        segment_vp = (segment_vpx, segment_vpy)
        screen_dims = (float(RENDER_WIDTH), float(RENDER_HEIGHT))



        if segment_vpy < RENDER_HEIGHT:
        #    # top segment
            segments[0] = get_segment_parameters(
                screen_dims, 
                segment_vp, RENDER_HEIGHT - segment_vpy, 
                (0, 1), 1, WORLD_MAX_Y, blit_surf)

        if segment_vpy > 0:
            segments[1] = get_segment_parameters(
                screen_dims, 
                segment_vp, segment_vpy, 
                (0, -1), 1, WORLD_MAX_Y, blit_surf)

        if segment_vpx < RENDER_WIDTH:
        #    # right segment
            segments[2] = get_segment_parameters(
                screen_dims, 
                screen_vp,  RENDER_WIDTH - screen_vpx, 
                (1, 0), 0, WORLD_MAX_Y, blit_surf)

        if segment_vpx > 0:
        #    # left segment
            segments[3] = get_segment_parameters(
                screen_dims, 
                screen_vp, screen_vpx, 
                (-1, 0), 0, WORLD_MAX_Y, blit_surf)


        top_down_ray_count = segments[0].ray_count + segments[1].ray_count
        print(f"top_down_ray_count {top_down_ray_count}")
        left_right_ray_count = segments[2].ray_count + segments[3].ray_count
        print(f"left_right_ray_count {left_right_ray_count}")

        camera = make_camera(FAR_CLIP_PLANE, pitch_ang, yaw_ang, roll_ang, pos_x, pos_y, pos_z, [RENDER_WIDTH, RENDER_HEIGHT])
        raycast_segments(
            segments, segment_vp, camera,
            heights, colors, top_down_draw_surf, left_right_draw_surf
        )
        
        tpds = pygame.transform.rotate(top_down_draw_surf, 90)
        #blit_surf.blit(top_down_draw_surf, (0, 0))
        blit_surf.blit(tpds, (0, 0))
        lrds = pygame.transform.rotate(left_right_draw_surf, 90)
        #blit_surf.blit(left_right_draw_surf, (RENDER_WIDTH + 2*RENDER_HEIGHT, 0))
        blit_surf.blit(lrds, (RENDER_WIDTH + 2*RENDER_HEIGHT, 0))
        
        print(f"x: {pos_x} y: {pos_y} z: {pos_z} ", end="")
        print(f"p: {round(rads_to_degrees(pitch_ang), 2)} r: {round(rads_to_degrees(roll_ang), 2)} y: {round(rads_to_degrees(yaw_ang), 2)}")


        # RENDER YOUR GAME HERE

        # flip() the display to put your work on screen
        pygame.display.flip()
        fps = clock.get_fps()
        print(f"fps: {fps}")
        last_fps = fps
        clock.tick(60)  # limits FPS to 60

    pygame.quit()

import cProfile
import pstats
import sys
if __name__ == '__main__':
    #print(sys.version)
    #profiler = cProfile.Profile()
    #profiler.enable()
    main()
    #profiler.disable()

    #stats = pstats.Stats(profiler).sort_stats('cumulative')
    #stats.print_stats()
    #stats.print_stats(20)