import typing

import pygame
import pygame.image

import math
from dataclasses import dataclass
import pyglet
import numpy as np
import numba

# CONSTANTS 

OUTPUT_WIDTH = 800
OUTPUT_HEIGHT = 800
RENDER_WIDTH = 200
RENDER_HEIGHT = 200
SHADE_SIZE = 400


FULL_CIRCLE = 2.0 * math.pi        # 360 degrees
HALF_CIRCLE = FULL_CIRCLE/2.0      # 180 degrees
QUARTER_CIRCLE = HALF_CIRCLE/2.0   # 90 degrees
EIGHTH_CIRCLE = QUARTER_CIRCLE/2.0 # 45 degrees
THREE_QUARTERS_CIRCLE = 3.0 * QUARTER_CIRCLE
MOVE_SPEED = 3



NEAR_CLIP_PLANE = .05
FAR_CLIP_PLANE = 2048 

# FILE IO, loading maps images, etc


T = typing.TypeVar('T')
def __load_transformed_img(file: str) -> typing.Generator[pygame.Color, typing.Any, typing.Any]:
    img = pygame.image.load(f"./comanche_maps/{file}")

    print(img.get_at((0,0)))
    w = img.get_width()
    h = img.get_height()
    for y in range(h):
        for x in range(w):
            yield img.get_at((1023-y, 1023-x))



load_color_img = __load_transformed_img
load_grayscale_img = lambda f: (x[0] for x in __load_transformed_img(f))


# data types 

type float2 = tuple[float, float]
type float3 = tuple[float, float, float]
type float4 = tuple[float, float, float, float]
type int2 = tuple[int, int]
type float4x4 = tuple[float4, float4, float4, float4]


@dataclass
class DDARay:
    position: int2
    step: int2
    start: float2
    dir: float2
    t_delta: float2
    t_max: float2
    intersection_distances: float2

@dataclass
class Segment:
    index: int
    min_screen: float
    max_screen: float
    cam_local_plane_ray_min: float
    cam_local_plane_ray_max: float
    ray_count: int

@dataclass
class SegmentContext:
    ray_buffer: np.ndarray
    segment: Segment
    original_next_free_pixel_min: int
    original_next_free_pixel_max: int
    axis_mapped_to_y: int
    segment_ray_index_offset: int
    seen_pixel_cache_length: int

@dataclass
class RaySegmentContext:
    context: SegmentContext
    plane_ray_index: int

@dataclass
class RayDDAContext:
    context: SegmentContext
    plane_ray_index: int
    dda_ray: DDARay

@dataclass
class RayContinuation:
    context: SegmentContext
    dda_ray: DDARay
    plane_ray_index: int
    ray_column: np.ndarray
    lod: int

def cmax(f: pyglet.math.Vec2) -> float:
    return max(f.x, f.y)

def cmin(f: pyglet.math.Vec2) -> float:
    return min(f.x, f.y)

def clamp(a,mi,ma):
    return min(max(a,mi), ma)


def frac(f2: pyglet.math.Vec2) -> pyglet.math.Vec2:
    return f2 - f2.__floor__()



def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


class Camera:
    def __init__(self):
        self.pos = pyglet.math.Vec3(512, 153.6, 512)
        self.forward = pyglet.math.Vec3(0, -.707, .707) #pyglet.math.Vec3(0, 0, 1.0)
        self.right = pyglet.math.Vec3(1.0, 0, 0)
        #self.up = pyglet.math.Vec3(0, 0, 1) 
        self.up = self.forward.cross(self.right)
        self.dims = pyglet.math.Vec2(RENDER_WIDTH, RENDER_HEIGHT)
        self.near_clip = NEAR_CLIP_PLANE
        self.far_clip = FAR_CLIP_PLANE
        self.fov = 90
        self.inverse_element_iteration_direction = self.forward.y >= 0

        self.world_to_screen_matrix_dirty = True
        self.world_to_camera_matrix_dirty = True
        self.projection_matrix_dirty = True

        self.world_to_screen_matrix = None
        self.world_to_camera_matrix = None
        self.projection_matrix = None

    def set_forward(self, new_forward: pyglet.math.Vec3):
        self.forward = new_forward
        self.right = self.up.cross(self.forward).normalize() #pyglet.math.Vec3(1.0, 0, 0)
        #self.right = pyglet.math.Vec3(self.forward.z, self.right.y, self.forward.x) #self.forward.cross(pyglet.math.Vec3(0, 1.0, 0))
        self.up = self.forward.cross(self.right).normalize()
        self.inverse_element_iteration_direction = self.forward.y >= 0
        self.set_dirty()

    def set_dirty(self):
        self.world_to_camera_matrix_dirty = True
        self.world_to_screen_matrix_dirty = True
        self.projection_matrix_dirty = True

    def euler_angles(self) -> pyglet.math.Vec3:
        z_axis = self.forward.normalize()
        x_axis = self.up.cross(z_axis).normalize()
        y_axis = z_axis.cross(x_axis).normalize()
        
        rot_matrix = pyglet.math.Mat4(
            x_axis.x, x_axis.y, x_axis.z, 0,
            y_axis.x, y_axis.y, y_axis.z, 0,
            z_axis.x, z_axis.y, z_axis.z, 0,
            0, 0, 0, 1
        )
        pitch = 0
        yaw = 0
        roll = 0

        # Only if the forward vector is not exactly (0,0,1), we need these calculations:
        pitch = math.atan2(rot_matrix[2*4+1], rot_matrix[2*4+2])
        yaw = math.atan2(-rot_matrix[2*4+0], math.sqrt(rot_matrix[2*4+1]**2 + rot_matrix[2*4+2]**2))
        roll = math.atan2(rot_matrix[1*4+0], rot_matrix[0*4+0])

        if z_axis.z < 0:
            yaw -= HALF_CIRCLE
            pitch -= HALF_CIRCLE
        
        yaw = normalize_radians(yaw)
        pitch = normalize_radians(pitch)
        roll = normalize_radians(roll)

        return pyglet.math.Vec3(pitch, yaw, roll)


    def get_translation_matrix(self) -> pyglet.math.Mat4:
        return pyglet.math.Mat4.from_translation(-self.pos)


    def get_world_to_camera_matrix(self) -> pyglet.math.Mat4:
        if self.world_to_camera_matrix_dirty:
            self.world_to_camera_matrix_dirty = False
            self.world_to_camera_matrix = pyglet.math.Mat4(
                self.right[0], self.up[0], -self.forward[0], 0,
                self.right[1], self.up[1], -self.forward[1], 0,
                self.right[2], self.up[2], -self.forward[2], 0,
                -self.right.dot(self.pos), -self.up.dot(self.pos), self.forward.dot(self.pos), 1
            )
        return self.world_to_camera_matrix


    def get_projection_matrix(self) -> pyglet.math.Mat4:
        if self.projection_matrix_dirty:
            self.projection_matrix_dirty = False
            self.projection_matrix = pyglet.math.Mat4.perspective_projection(
                self.dims.x/self.dims.y, self.near_clip, self.far_clip,
                self.fov
            )
        return self.projection_matrix
    
    def get_world_to_screen_matrix(self) -> pyglet.math.Mat4:
        if self.world_to_screen_matrix_dirty:
            self.world_to_screen_matrix_dirty = False
            world_to_cam_matrix = self.get_world_to_camera_matrix()
            cam_to_screen_matrix = self.get_projection_matrix()

            world_to_screen_matrix = cam_to_screen_matrix @ world_to_cam_matrix
            world_to_screen_matrix = (
                pyglet.math.Mat4.from_scale(pyglet.math.Vec3(.5, .5, 1)) @ world_to_screen_matrix # scale from -1,1 to -0.5,.5
            )
            world_to_screen_matrix = (
                pyglet.math.Mat4.from_translation(pyglet.math.Vec3(0.5, 0.5, 1)) @ world_to_screen_matrix # translate from -0.5,.5 to 0,1
            )

            world_to_screen_matrix = (
                pyglet.math.Mat4.from_scale(pyglet.math.Vec3(self.dims.x, self.dims.y, 1)) @ world_to_screen_matrix # scale from 0,1 to 0,screen
            )
            self.world_to_screen_matrix = world_to_screen_matrix

        return self.world_to_screen_matrix
    

def project_to_homogeneous_camera_space(camera: Camera, world_a: pyglet.math.Vec3) -> pyglet.math.Vec4: 
        return camera.world_to_screen_matrix @ world_a

def setup_projected_plane_params(
        camera: Camera,
        ray: DDARay,
        world_max_y: float,
        voxel_scale: int,
        y_axis: int):
    start = ray.start
    plane_start_bottom = pyglet.math.Vec4(start[0], 0.0, start[1], 1)
    plane_start_top = pyglet.math.Vec4(start[0], world_max_y, start[1], 1)
    plane_ray_direction = pyglet.math.Vec4(ray.dir[0], 0.0, ray.dir[1], 0)

    world_to_screen_mat = camera.get_world_to_screen_matrix()
    full_plane_start_top_projected = world_to_screen_mat @ plane_start_top
    full_plane_start_bot_projected = world_to_screen_mat @ plane_start_bottom
    full_plane_ray_direction_projected = world_to_screen_mat @ plane_ray_direction
    if y_axis == 0:
        return (
             pyglet.math.Vec3(full_plane_start_bot_projected.x, full_plane_start_bot_projected.z, full_plane_start_bot_projected.w),
             pyglet.math.Vec3(full_plane_start_top_projected.x, full_plane_start_top_projected.z, full_plane_start_top_projected.w),
             pyglet.math.Vec3(full_plane_ray_direction_projected.x, full_plane_ray_direction_projected.z, full_plane_ray_direction_projected.w)
        )
    else:
        return (
             pyglet.math.Vec3(full_plane_start_bot_projected.y, full_plane_start_bot_projected.z, full_plane_start_bot_projected.w),
             pyglet.math.Vec3(full_plane_start_top_projected.y, full_plane_start_top_projected.z, full_plane_start_top_projected.w),
             pyglet.math.Vec3(full_plane_ray_direction_projected.y, full_plane_ray_direction_projected.z, full_plane_ray_direction_projected.w)
        )

# MATH UTILITIES FOR VECTORS 

@numba.njit
def lerp3(a:float3,b:float3,f:float):
    #a + (b-a) * f
    return add3(a, scale3(sub3(b,a), f))

@numba.njit
def lerp2(a:float2,b:float2,f:float):
    return add2(a, scale2(sub2(b,a), f))

@numba.njit
def scale3(a:float3,f:float):
    return (a[0]*f,a[1]*f,a[2]*f)

@numba.njit
def scale2(a:float2,f:float):
    return (a[0]*f,a[1]*f)

@numba.njit
def offset3(a:float3,f:float):
    return (a[0]+f,a[1]+f,a[2]+f)

@numba.njit
def add3(a:float3,b:float3):
    return (a[0]+b[0],a[1]+b[1],a[2]+b[2])

@numba.njit
def add2(a:float2,b:float2):
    return (a[0]+b[0],a[1]+b[1])

@numba.njit
def sub3(a:float3,b:float3):
    return (a[0]-b[0],a[1]-b[1],a[2]-b[2])

@numba.njit
def sub2(a:float2,b:float2):
    return (a[0]-b[0], a[1]-b[1],)

@numba.njit
def clip_homogeneous_camera_space_line(a: float3, b: float3) -> tuple[bool, float2, float2]:
    # clip to camera space
    (ax,ay,az) = a
    (bx,by,bz) = b

    if (ay <= 0):
        if (by <= 0):
            return False, (ax,az),(bx,bz)
        
        v = by / (by - ay)
        ax,az = lerp2((bx,bz), (ax,az), v)
        return True, (ax,az),(bx,bz)
    elif (by <= 0):
        v = ay / (ay - by)
        (bx,bz) = lerp2((ax,az), (bx,bz), v)
        return True, (ax,az),(bx,bz)
    else:
        return True, (ax,az),(bx,bz)



# unused!
# we just fill the entire buffer with blue every frame instead of calling this
                        # B G R A
skybox_col = pygame.Color(135,206,235,0xFF)
def draw_skybox(next_free_pix_min, next_free_pix_max, seen_pixel_cache, pix_arr, ray_column_off):
    #pass
    for y in range(next_free_pix_min, next_free_pix_max):
        if seen_pixel_cache[y] == 0:
            pix_arr[ray_column_off, y] = skybox_col

def color_to_int(color: pygame.Color) -> int:
    return (color.b | (color.g<<8) | (color.r<<16) | (color.a<<24))


def execute_rays(total_rays, camera: Camera, ray_continuations: typing.List[RayContinuation],
        height_map, color_map
        ):
    

    # these are used to track which pixels have been filled while raycasting
    # NOTE: for regular heightmaps, these are not strictly required
    # but since we want to (eventually) support arbitrary voxel data per column
    # these are necessary

    # here generate fixed sized caches, one of each size of the camera dimensions,
    # before iterating over rays
    # whichever cache is used will be re-filled with zeros after each ray is executed
    seen_pixel_caches = [
        np.zeros([camera.dims[0]], dtype=np.uint8),
        np.zeros([camera.dims[1]], dtype=np.uint8)
    ]

    far_clip = camera.far_clip
    voxel_scale = 1

    world_max_y = WORLD_MAX_Y
    one_over_world_max_y = 1/world_max_y 
    
    for i in range(total_rays):
        ray_cont = ray_continuations[i]
        dda_ray = ray_cont.dda_ray
        segment_ctx = ray_cont.context
        pix_arr = ray_cont.context.ray_buffer 
        col = ray_cont.ray_column

        seen_pixel_cache = seen_pixel_caches[ray_cont.context.axis_mapped_to_y]

        #pix_arr_col = pix_arr[i]

        #int voxelScale = 1 << lod;
        #camera_pos_y_normalized = camera.pos.y / world_max_y

        # small offset to the frustums to prevent a division by zero in the clipping algorithm
        #frustum_bounds_min = cur_next_free_pix_min - .501
        #frustum_bounds_max = cur_next_free_pix_max + .501

        (plane_start_bottom_projected,
        plane_start_top_projected,
        plane_ray_direction_projected) = setup_projected_plane_params(
            camera, dda_ray, world_max_y, voxel_scale, segment_ctx.axis_mapped_to_y
        )


        intersection_distances = dda_ray.intersection_distances
        intersection_distances_x, intersection_distances_y = dda_ray.intersection_distances

        step_x, step_y = dda_ray.step
        t_delta_x, t_delta_y = dda_ray.t_delta
        t_max_x, t_max_y = dda_ray.t_max
        position_x, position_y = dda_ray.position
        
        # this is used to control the iteration order of elements within a voxel column
        # we don't actually handle more than just a single height value per column right now
        # so it's unused
        cam_inv = camera.inverse_element_iteration_direction

        original_next_free_pix_min = segment_ctx.original_next_free_pixel_min
        original_next_free_pix_max = segment_ctx.original_next_free_pixel_max

        ray_loop(far_clip, world_max_y, height_map, color_map,
                 plane_start_bottom_projected, plane_start_top_projected, plane_ray_direction_projected,
                 one_over_world_max_y, seen_pixel_cache, pix_arr, col,
                 original_next_free_pix_min, original_next_free_pix_max,
                 intersection_distances_x, t_max_x, t_max_y,
                 t_delta_x, t_delta_y, position_x, position_y, step_x, step_y)
        
        seen_pixel_cache.fill(0) 



@numba.njit
def ray_loop(far_clip: float, world_max_y: int, 
             height_map: np.ndarray[typing.Any, np.dtype[np.uint16]], color_map: np.ndarray[typing.Any, np.dtype[np.uint32]],
              plane_start_bot: float3, plane_start_top: float3, plane_ray_dir: float3, 
              one_over_world_max_y: float, 
              seen_pixel_cache: np.ndarray[typing.Any, np.dtype[np.uint8]], 
              pix_arr: np.ndarray[typing.Any, np.dtype[np.uint32]], col: int,
              original_next_free_pix_min: int, original_next_free_pix_max: int,
              intersection_distances_x: float, t_max_x: float, t_max_y: float,
              t_delta_x: float, t_delta_y: float, position_x: int, position_y: int,
              step_x: int, step_y: int):
    

    cur_next_free_pix_min, cur_next_free_pix_max = original_next_free_pix_min, original_next_free_pix_max
    while True:
        if(intersection_distances_x >= far_clip):
            #draw_skybox(cur_next_free_pix_min, cur_next_free_pix_max, seen_pixel_cache, pix_arr, i)
            break # no lod stuff :)

        element_bounds_min = 0
        element_bounds_max = 0

        index = (1023-(position_y&1023))*1024+((position_x&1023))
        height = world_max_y - height_map[index]
        color = color_map[index]
        # B G R A
        
        # plane_start_bot, plane_start_top are the camera space positions of the start of the
        # ray plane for this raycast

        # calculate the position along the ray via adding the ray_direction * dist to these values
        plane_ray_dir_times_dist = scale3(plane_ray_dir, intersection_distances_x)
        cam_space_min_last = add3(plane_start_bot, plane_ray_dir_times_dist)
        cam_space_max_last = add3(plane_start_top, plane_ray_dir_times_dist)

        element_bounds_min = element_bounds_max
        element_bounds_max = element_bounds_min + (world_max_y - height)
        
        # calculate the position, between 0 and 1, in world space
        # of the top and bottom of the solid chunk of voxels for this column
        portion_top = element_bounds_max * one_over_world_max_y
        portion_bottom = element_bounds_min * one_over_world_max_y

        # now lerp the camera space top and bottom ray positions with the portions that
        # correspond to the bottom and top of the solid voxel chunk

        # this gives us a camera space position for the voxel chunk
        cam_space_front_bottom = lerp3(cam_space_min_last, cam_space_max_last,portion_bottom)
        cam_space_front_top = lerp3(cam_space_min_last, cam_space_max_last, portion_top)


        (onscreen, (cam_space_front_top_x, cam_space_front_top_z), 
        (cam_space_front_bot_x, cam_space_front_bot_z)) = clip_homogeneous_camera_space_line(
            cam_space_front_top,
            cam_space_front_bottom,
        )

        # near clipping
        
        # 
        if onscreen:
            # calculate the position in the ray buffer
            ray_buffer_bounds_float_min = cam_space_front_top_x / cam_space_front_top_z
            ray_buffer_bounds_float_max = cam_space_front_bot_x / cam_space_front_bot_z

            # flip min and max if necessary
            if ray_buffer_bounds_float_max < ray_buffer_bounds_float_min:
                ray_buffer_bounds_float_min, ray_buffer_bounds_float_max = ray_buffer_bounds_float_max, ray_buffer_bounds_float_min

            # round to an integer position
            ray_buffer_bounds_min = round(ray_buffer_bounds_float_min)
            ray_buffer_bounds_max = round(ray_buffer_bounds_float_max)

            # check if within screen-space drawable bounds
            if (ray_buffer_bounds_max >= original_next_free_pix_min and ray_buffer_bounds_float_min <= original_next_free_pix_max):
                
                # if this visible chunk touches the top screen bound
                # shrink top of frustum as much as possible
                if ray_buffer_bounds_min <= cur_next_free_pix_min:
                    ray_buffer_bounds_min = cur_next_free_pix_min
                    
                    if ray_buffer_bounds_max >= cur_next_free_pix_min:
                        cur_next_free_pix_min = ray_buffer_bounds_max+1

                        while cur_next_free_pix_min <= original_next_free_pix_max and seen_pixel_cache[cur_next_free_pix_min] > 0:
                            cur_next_free_pix_min += 1

                # if this visible chunk touches the bottom screen bound
                # shrink bottom frustum as much as possible
                if ray_buffer_bounds_max >= cur_next_free_pix_max:
                    ray_buffer_bounds_max = cur_next_free_pix_max
                    if ray_buffer_bounds_min <= cur_next_free_pix_max:
                        cur_next_free_pix_max = ray_buffer_bounds_min-1
                        while cur_next_free_pix_max >= original_next_free_pix_min and seen_pixel_cache[cur_next_free_pix_max] > 0:
                            cur_next_free_pix_max -= 1

                # now draw visible portion of the chunk
                for y in range(ray_buffer_bounds_min, ray_buffer_bounds_max+1):
                    if seen_pixel_cache[y] == 0:
                        seen_pixel_cache[y] = 1
                        pix_arr[col][y] = color


            # if the frustum doesn't cover a pixel, break out of the loop
            if cur_next_free_pix_min > cur_next_free_pix_max:
                break
        
        # step the ray to the next grid intersection
        if (t_max_x < t_max_y):
            crossed_boundary_distance = t_max_x
            t_max_x += t_delta_x
            position_x += step_x
        else:
            crossed_boundary_distance = t_max_y
            t_max_y += t_delta_y
            position_y += step_y

        intersection_distances_x = crossed_boundary_distance


def rads_to_degrees(r):
    return r * 57.2

def step(edge, x):
    if x < edge:
        return 0.0
    return 1.0

# TODO: do correctly :)
def shade_screen(screen_draw_surf: pygame.Surface, seg_draw_surf: pygame.Surface, seg_offsets, rot_x_greater_zero, shader_vp):
    raise Exception("This is unimplemented :)")

def normalize_radians(f):
    while f < 0:
        f += math.radians(360)
    while f > math.radians(360):
        f -= math.radians(360)
    return f

    
def handle_input(camera: Camera, keys_down):

    views = [
        (
            pygame.K_KP_0, # looking forward
            [0,0,1],    # forward
            [1,0,0],    # right
            [0,1,0]     # up
        ),
        (
            pygame.K_KP_1, # looking right
            [1,0,0],
            [0,0,-1],
            [0,1,0]
        ),
        (
            pygame.K_KP_2, # looking back
            [0,0,-1],
            [-1,0,0],
            [0,1,0]
        ),
        (
            pygame.K_KP_3, # looking left
            [-1,0,0],
            [0,0,1],
            [0,1,0]
        ),
        (
            pygame.K_KP_4,
            [0,-1,0],
            [1,0,0],
            [0,0,1]
        )
    ]
    for (key, (fx,fy,fz), (rx,ry,rz), (ux,uy,uz)) in views:
        if key in keys_down:
            camera.forward = pyglet.math.Vec3(fx,fy,fz).normalize()
            camera.right = pyglet.math.Vec3(rx,ry,rz).normalize()
            camera.up = pyglet.math.Vec3(ux,uy,uz).normalize()
            camera.set_dirty()
            break


    if abs(camera.forward.y) < 0.0001:
        newForward = pyglet.math.Vec3(camera.forward.x, sign(camera.forward.y)*.001, camera.forward.z).normalize()
        camera.set_forward(newForward)

    
    dpitch = 0
    if pygame.K_z in keys_down:
        dpitch = -.03
    elif pygame.K_x in keys_down:
        dpitch = .03
    
    if dpitch != 0:
        sp,cp = math.sin(dpitch),math.cos(dpitch)
        rot_mat = pyglet.math.Mat3(
            1,0,0,
            0,cp,sp,
            0,-sp,cp
        )
        camera.forward = (rot_mat @ camera.forward).normalize()
        camera.right = (rot_mat @ camera.right).normalize()
        camera.up = (rot_mat @ camera.up).normalize()
        camera.set_dirty()

    dyaw = 0
    if pygame.K_LEFT in keys_down:
        dyaw = .03
    elif pygame.K_RIGHT in keys_down:
        dyaw = -.03

    if dyaw != 0:
        sy,cy = math.sin(dyaw),math.cos(dyaw)

        rot_mat = pyglet.math.Mat3(
            cy,0,sy,
            0,1,0,
            -sy,0,cy
        )
        camera.forward = (rot_mat @ camera.forward).normalize()
        camera.right = (rot_mat @ camera.right).normalize()
        camera.up = (rot_mat @ camera.up).normalize()
        camera.set_dirty()

    droll = 0
    if pygame.K_q in keys_down:
        droll = -0.3
    elif pygame.K_f in keys_down:
        droll = .3

    if droll != 0:
        sr,cr = math.sin(droll),math.cos(droll)
        rot_mat = pyglet.math.Mat3(
            cr, -sr, 0,
            -sr, cr, 0,
            0, 0, 1
        )
        camera.forward = (rot_mat @ camera.forward).normalize()
        camera.right = (rot_mat @ camera.right).normalize()
        camera.up = (rot_mat @ camera.up).normalize()
        camera.set_dirty()

    
    if abs(camera.forward.y) < 0.001:
        newForward = pyglet.math.Vec3(camera.forward.x, sign(camera.forward.y)*.001, camera.forward.z).normalize()
        camera.set_forward(newForward)
        
    new_pos = camera.pos
    if pygame.K_c in keys_down:
        new_pos = new_pos + camera.up * MOVE_SPEED
    elif pygame.K_v in keys_down:
        new_pos = new_pos + camera.up * -MOVE_SPEED

    if pygame.K_UP in keys_down:
        new_pos = new_pos + camera.forward * MOVE_SPEED
    elif pygame.K_DOWN in keys_down:
        new_pos = new_pos + camera.forward * -MOVE_SPEED
    
    camera.pos = new_pos
    

def calc_vanishing_point_world(camera: Camera):
    rot_y = 1 * (-NEAR_CLIP_PLANE / -camera.forward.y)
    return camera.pos + pyglet.math.Vec3(0, rot_y, 0)


def project_vanishing_point_world_to_screen(camera: Camera, vp_world: pyglet.math.Vec3):
    forward = camera.forward
    right = camera.right
    up = camera.forward.cross(right).normalize()
    # forward.x neg
    # right.z neg?
    look_matrix = pyglet.math.Mat4(
        right.x, right.y, right.z, 0,
        up.x, up.y, up.z, 0,
        forward.x, forward.y, forward.z, 0,
        0, 0, 0, 1
    )


    view_matrix = pyglet.math.Mat4.from_scale(pyglet.math.Vec3(1,1,-1)) @ look_matrix.__invert__()
    proj_matrix = camera.get_projection_matrix()
    local_to_screen_matrix = proj_matrix @ view_matrix

    local_pos = vp_world - camera.pos
    cam_pos = local_to_screen_matrix @ pyglet.math.Vec4(local_pos.x, local_pos.y, local_pos.z, 1.0)
    if abs(cam_pos.w) < 0.001:
        cam_pos = pyglet.math.Vec4(
            cam_pos.x, cam_pos.y, cam_pos.z, sign(cam_pos.w) * .001,
        )
    return (((pyglet.math.Vec2(cam_pos.x, cam_pos.y) / cam_pos.w) * .5 + .5) *
            camera.dims)

WORLD_MAX_Y = 256


k_epsilon_normal_sqrt = .000000000000001
def vec2_angle(frm: pyglet.math.Vec2, to: pyglet.math.Vec2) -> float:
    frmSqrMag = frm.x*frm.x + frm.y*frm.y
    toSqrMag = to.x*to.x + to.y*to.y
    denom = math.sqrt(frmSqrMag * toSqrMag)
    if denom < k_epsilon_normal_sqrt:
        return 0
    
    dot = frm.dot(to) / denom
    dot = clamp(dot, -1, 1)
    return math.acos(dot) * 57.29
    

def vec2_signed_angle(frm: pyglet.math.Vec2, to: pyglet.math.Vec2) -> float:
    unsigned_angle = vec2_angle(frm, to)
    sgn = sign(frm.x * to.y - frm.y * to.x)
    return unsigned_angle * sgn


def transform_pixel(camera: Camera, pix: pyglet.math.Vec2):
    scaled_pix = pix / camera.dims
    off_pix = scaled_pix[0]-0.5, scaled_pix[1]-0.5
    homogenoeous_pix = pyglet.math.Vec4(off_pix[0]*2, off_pix[1]*2, 1, 1)


    matrix = camera.get_projection_matrix().__invert__()
    inverse_proj_pix = matrix @ homogenoeous_pix

    inverse_scale_pix = pyglet.math.Mat4.from_scale(pyglet.math.Vec3(1,1,-1)) @ inverse_proj_pix

    forward = camera.forward
    right = camera.right
    up = camera.forward.cross(right)


    lookat_matrix = pyglet.math.Mat4(
        right.x, right.y, right.z, 0,
        up.x, up.y, up.z, 0,
        forward.x, forward.y, forward.z, 0,
        0, 0, 0, 1
    )


    world_pix = lookat_matrix @ inverse_scale_pix
    unprojected_world_pix = pygame.math.Vector2(world_pix.x / world_pix.w, world_pix.z / world_pix.w)  
    return unprojected_world_pix


def dist2d(x1,y1, x2,y2):
    dx = abs(x2-x1)
    dy = abs(y2-y1)
    return math.sqrt(dx*dx+dy*dy)


def vec2_set_axis(v: pyglet.math.Vec2, axis: int, val: float) -> pyglet.math.Vec2:
    if axis == 0:
        return pyglet.math.Vec2(val, v.y)
    else:
        return pyglet.math.Vec2(v.x, val)


# get parameters for each segment, including 
# number of rays, 
# world-space position for min and max coordinates on screen
def get_segment_parameters(segment_index: int, camera: Camera,
                           screen_vp: pyglet.math.Vec2, dist_to_other_end: float, 
                           neutral: pyglet.math.Vec2, primary_axis: int, world_y_max: int,
                           screen_draw_surf: pygame.Surface):
    secondary_axis = 1 - primary_axis

    segment = Segment(segment_index, 0.0, 0.0, 0.0, 0.0, 0)
    max_ray_count = screen_draw_surf.get_width()

    # setup the end points for the 2 45 degree angle rays
    scmin = screen_vp[secondary_axis] - dist_to_other_end
    simple_case_min = pyglet.math.Vec2(scmin, scmin)
    scmax = screen_vp[secondary_axis] + dist_to_other_end
    simple_case_max = pyglet.math.Vec2(scmax, scmax)
    a = screen_vp[primary_axis] + dist_to_other_end * sign(neutral[primary_axis])

    simple_case_min = vec2_set_axis(simple_case_min, primary_axis, a)
    simple_case_max = vec2_set_axis(simple_case_max, primary_axis, a)


    if simple_case_max[secondary_axis] <= 0 or simple_case_min[secondary_axis] >= camera.dims[secondary_axis]:
        return segment
    if (screen_vp.x >= 0 and screen_vp.y >= 0 and screen_vp.x <= camera.dims.x and screen_vp.y <= camera.dims.y):
        # vp within bounds, so nothing to clamp angle wise
        segment.min_screen = simple_case_min
        segment.max_screen = simple_case_max
    else:
        # vp outside of bounds, so we want to check if we can clamp the segment to the screen area
        # to prevent wasting buffer space
        dir_simple_middle = pyglet.math.Vec2.lerp(simple_case_min, simple_case_max, .5) - screen_vp
        angle_left = 90
        angle_right = -90
        dir_left = [0,0]
        dir_right = [0,0]

        vectors = [(0,0), (0, camera.dims.y), (camera.dims.x, 0), camera.dims]

        for i in range(4):
            dirvec = vectors[i] - screen_vp
            scaled_end = dirvec * (dist_to_other_end / abs(dirvec[primary_axis]))
            angle = vec2_signed_angle(neutral, dirvec)
            if angle < angle_left:
                angle_left = angle
                dir_left = scaled_end
            if angle > angle_right:
                angle_right = angle
                dir_right = scaled_end

        corner_left = dir_left + screen_vp
        corner_right = dir_right + screen_vp

        if angle_left < -45:
            # fallback to whatever the simple case left corner was
            sgn = vec2_signed_angle(dir_simple_middle, simple_case_max)
            if sgn > 0:
                corner_left = simple_case_min
            else:
                corner_left = simple_case_max
            
        if angle_right > 45:
            sgn = vec2_signed_angle(dir_simple_middle, simple_case_max)
            if sgn < 0:
                corner_right = simple_case_min 
            else:
                corner_right = simple_case_max


        swap = corner_left[secondary_axis] > corner_right[secondary_axis]

        # todo is this correct?
        segment.min_screen = corner_right if swap else corner_left
        segment.max_screen = corner_left if swap else corner_right

    segment.cam_local_plane_ray_min = transform_pixel(camera, segment.min_screen)
    segment.cam_local_plane_ray_max = transform_pixel(camera, segment.max_screen)

    ray_count = round(segment.max_screen[secondary_axis] - segment.min_screen[secondary_axis])
    segment.ray_count = max(min(ray_count, max_ray_count), 0)

    return segment


# map each ray to it's parent segment
def setup_ray_segment_mapping(total_rays, segmentContexts: typing.List[SegmentContext]) -> typing.List[RaySegmentContext]:
    rays: typing.List[RaySegmentContext] = []
    for i in range(total_rays):
        plane_index = i
        for j in range(4):

            segment_rays = segmentContexts[j].segment.ray_count
            if segment_rays <= 0:
                continue
            if plane_index >= segment_rays:
                plane_index -= segment_rays
                continue
                
            rays.append(
                RaySegmentContext(context=segmentContexts[j], plane_ray_index=plane_index)
            )
            break
    return rays


# calculate DDA information for a given start position and direction
def make_ray(start: pyglet.math.Vec2, dir: pyglet.math.Vec2) -> DDARay:
    position = [math.floor(start.x), math.floor(start.y)]
    eps = .0000001
    absDir = dir.__abs__()
    t_delta = 1 / pyglet.math.Vec2(max(eps, absDir.x), max(eps, absDir.y))

    sign_dir = pyglet.math.Vec2(sign(dir.x), sign(dir.y))
    step = pyglet.math.Vec2(int(sign_dir.x), int(sign_dir.y))
    t_max_vec = ((sign_dir * -frac(start)) + (sign_dir * 0.5) + 0.5) * t_delta
    t_max = pyglet.math.Vec2(t_max_vec.x, t_max_vec.y)
    intersection_distances = pyglet.math.Vec2(cmax(t_max - t_delta), cmin(t_max))
    return DDARay(position=position, step=step, start=start, dir=dir, t_delta=t_delta, t_max=t_max, intersection_distances=intersection_distances)


# stash stuff here for debugging
segment_start_end_rays  = None


# sets up the DDA details for each ray
def setup_ray_dda_contexts(total_rays: int, camera: Camera, ray_segment_contexts: typing.List[RaySegmentContext]) -> typing.List[RayDDAContext]:
    res: typing.List[RayDDAContext] = []
    global segment_start_end_rays

    segment_start_end_rays = [
        [None,None],
        [None,None],
        [None,None],
        [None,None]
    ]

    for i in range(total_rays):
        ctx = ray_segment_contexts[i].context
        segment = ctx.segment
        segment_idx = segment.index
        end_ray_lerp = ray_segment_contexts[i].plane_ray_index / segment.ray_count
        
        cam_local_plane_ray_direction = pyglet.math.Vec2.lerp(
            segment.cam_local_plane_ray_min, segment.cam_local_plane_ray_max, end_ray_lerp
        )
        
        norm_ray_dir = cam_local_plane_ray_direction.normalize()
        ray = make_ray(pyglet.math.Vec2(camera.pos.x, camera.pos.z), norm_ray_dir)

        res.append(RayDDAContext(
            context=ctx, plane_ray_index=ray_segment_contexts[i].plane_ray_index,
            dda_ray=ray
        ))


    return res

# sets up the remaining information for each ray
def setup_ray_continuations(total_rays: int, ray_dda_contexts: typing.List[RayDDAContext]) -> typing.List[RayContinuation]:
    res: typing.List[RayContinuation] = []
    for i in range(total_rays):
        ray_context = ray_dda_contexts[i]
        segment_context = ray_context.context
        segment_idx = segment_context.segment.index
        total_index = ray_context.plane_ray_index + segment_context.segment_ray_index_offset
        #print(f"index: {total_index}")
        cont = RayContinuation(
            context=ray_context.context,
            dda_ray=ray_context.dda_ray,
            plane_ray_index=ray_context.plane_ray_index,
            ray_column=total_index, #, segment_context.ray_buffer[total_index]),
            lod=0
        )
        res.append(cont)
        if segment_start_end_rays[segment_idx][0] is None:
            segment_start_end_rays[segment_idx][0] = cont

        segment_start_end_rays[segment_idx][1] = cont
    return res





def raycast_segments(
    segments, vanishing_point_screen_space, camera: Camera,
    heights, colors, top_down_pix_arr: np.ndarray, left_right_pix_arr: np.ndarray):
    
    vp_x, vp_y = vanishing_point_screen_space

    segment_contexts: typing.List[SegmentContext] = [
        SegmentContext(
            ray_buffer=top_down_pix_arr, segment=segments[0],
            original_next_free_pixel_min=0, original_next_free_pixel_max=0,
            axis_mapped_to_y=1, segment_ray_index_offset=0,
            seen_pixel_cache_length=0
        ),
        SegmentContext(
            ray_buffer=top_down_pix_arr, segment=segments[1],
            original_next_free_pixel_min=0, original_next_free_pixel_max=0,
            axis_mapped_to_y=1, segment_ray_index_offset=0,
            seen_pixel_cache_length=0
        ),
        SegmentContext(
            ray_buffer=left_right_pix_arr, segment=segments[2],
            original_next_free_pixel_min=0, original_next_free_pixel_max=0,
            axis_mapped_to_y=0, segment_ray_index_offset=0,
            seen_pixel_cache_length=0
        ),
        SegmentContext(
            ray_buffer=left_right_pix_arr, segment=segments[3],
            original_next_free_pixel_min=0, original_next_free_pixel_max=0,
            axis_mapped_to_y=0, segment_ray_index_offset=0,
            seen_pixel_cache_length=0
        )
    ]

    total_rays = 0

    top_down_pix_arr.fill(color_to_int(skybox_col))
    left_right_pix_arr.fill(color_to_int(skybox_col))

    for segment_index in range(4):
        total_rays += segments[segment_index].ray_count
        if segments[segment_index].ray_count == 0:
            continue

        segment_ray_index_offset = 0
        if segment_index == 1:
            segment_ray_index_offset = segments[0].ray_count
        if segment_index == 3:
            segment_ray_index_offset = segments[2].ray_count
        
        # 0,1 are mapped to y, 2,3 are mapped to x
        axis_mapped_to_y = 0 if segment_index > 1 else 1

        screen_width, screen_height = camera.dims.x, camera.dims.y
        if segment_index < 2:
            pix_arr = top_down_pix_arr
            if segment_index == 0:
                next_free_pixel = pyglet.math.Vec2(pyglet.math.clamp(round(vp_y), 0, screen_width-1), screen_height-1)
            else:
                next_free_pixel = pyglet.math.Vec2(0, pyglet.math.clamp(round(vp_y), 0, screen_height-1))
        else:
            pix_arr = left_right_pix_arr
            if segment_index == 3:
                next_free_pixel = pyglet.math.Vec2(0, clamp(round(vp_x), 0, screen_width-1))
            else:
                next_free_pixel = pyglet.math.Vec2(clamp(round(vp_x), 0, camera.dims.x-1), screen_width-1)
        
        seen_pixel_cache_length = camera.dims[axis_mapped_to_y]

        segment_contexts[segment_index] = SegmentContext(
            ray_buffer=pix_arr, segment=segments[segment_index],
            original_next_free_pixel_min=next_free_pixel.x, original_next_free_pixel_max=next_free_pixel.y,
            axis_mapped_to_y=axis_mapped_to_y,
            segment_ray_index_offset=segment_ray_index_offset,
            seen_pixel_cache_length=seen_pixel_cache_length
        )
        

    ray_segment_contexts = setup_ray_segment_mapping(total_rays, segment_contexts)
    ray_dda_contexts = setup_ray_dda_contexts(total_rays, camera, ray_segment_contexts)
    ray_continuations = setup_ray_continuations(total_rays, ray_dda_contexts)

    execute_rays(total_rays, camera, ray_continuations, heights, colors)

        


def main():
    color_pix = load_color_img("C1W.png")
    height_pix = load_grayscale_img("D1.png")

    colors = np.array([color_to_int(x) for x in color_pix]) 
    heights = np.array([x for x in height_pix], dtype=np.uint16)

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((OUTPUT_WIDTH, OUTPUT_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    camera = Camera()

    # we have to create these surfaces with reverse dimensions,
    # because pygame.surf array expects this
    max_rays_left_right = 2*RENDER_WIDTH + RENDER_HEIGHT
    max_rays_up_down = RENDER_WIDTH + 2*RENDER_HEIGHT
    left_right_draw_surf = pygame.Surface((max_rays_left_right, RENDER_WIDTH))
    top_down_draw_surf = pygame.Surface((max_rays_up_down, RENDER_HEIGHT))

    left_right_pix_arr = np.full([max_rays_left_right, RENDER_WIDTH], 
                                 color_to_int(skybox_col), dtype=np.uint32)
    top_down_pix_arr = np.full([max_rays_up_down, RENDER_HEIGHT], 
                               color_to_int(skybox_col), dtype=np.uint32)
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

        handle_input(camera, keys_down)

        camera.set_dirty()

        world_vp = calc_vanishing_point_world(camera)
        screen_vp = project_vanishing_point_world_to_screen(camera, world_vp)

        segments = [
            Segment(0, 0, 0, 0, 0, 0), Segment(1, 0, 0, 0, 0, 0), 
            Segment(2, 0, 0, 0, 0, 0), Segment(3, 0, 0, 0, 0, 0)
        ]
        

        if screen_vp.y < camera.dims.y:
        #    # top segment
            segments[0] = get_segment_parameters(
                0,
                camera, screen_vp, camera.dims.y - screen_vp.y, 
                pyglet.math.Vec2(0, 1), 1, WORLD_MAX_Y, top_down_draw_surf)

        if screen_vp.y > 0:
            # down segment
            segments[1] = get_segment_parameters(
                1,
                camera, screen_vp, screen_vp.y, 
                pyglet.math.Vec2(0, -1), 1, WORLD_MAX_Y, top_down_draw_surf)

        if screen_vp.x < camera.dims.x:
        #    # right segment
            segments[2] = get_segment_parameters(
                2,
                camera, screen_vp,  camera.dims.x - screen_vp.x, 
                pyglet.math.Vec2(1, 0), 0, WORLD_MAX_Y, left_right_draw_surf)

        if screen_vp.x > 0:
        #    # left segment
            segments[3] = get_segment_parameters(
                3,
                camera, screen_vp, screen_vp.x, 
                pyglet.math.Vec2(-1, 0), 0, WORLD_MAX_Y, left_right_draw_surf)
        
        raycast_segments(
            segments, screen_vp, camera,
            heights, colors, top_down_pix_arr, left_right_pix_arr
        )

        pygame.surfarray.blit_array(top_down_draw_surf, top_down_pix_arr)
        pygame.surfarray.blit_array(left_right_draw_surf, left_right_pix_arr)
        
        #tpds = pygame.transform.rotate(top_down_draw_surf, 90)
        blit_surf.blit(top_down_draw_surf, (0, 0))
        #lrds = pygame.transform.rotate(left_right_draw_surf, 90)

        blit_surf.blit(left_right_draw_surf, (0, RENDER_WIDTH))
        print(f"pos: {camera.pos}")
        print(f"fx: {camera.forward.x} fy: {camera.forward.y} fz: {camera.forward.z}")
        print(f"ux: {camera.up.x} uy: {camera.up.y} uz: {camera.up.z}")
        print(f"rx: {camera.right.x} ry: {camera.right.y} rz: {camera.right.z}")


        # RENDER YOUR GAME HERE

        # flip() the display to put your work on screen
        pygame.display.flip()
        fps = clock.get_fps()
        print(f"fps: {fps}")
        clock.tick(200)  # limits FPS to 60

    pygame.quit()




def floats_close(f1, f2):
    return abs(f2-f1) < 0.017

def vecs_close(f1: pyglet.math.Vec3, f2: pyglet.math.Vec3):
    return floats_close(f1.x, f2.x) and floats_close(f1.y, f2.y) and floats_close(f1.z, f2.z)

import cProfile
import pstats
import sys
PROFILING = 0

if __name__ == '__main__':
    if PROFILING:
        print(sys.version)
        profiler = cProfile.Profile()
        profiler.enable()

    main()

    if PROFILING:
        profiler.disable()

        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats()
        stats.print_stats(20)