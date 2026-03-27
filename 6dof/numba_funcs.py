import numpy as np
import math
import numba 

import typing
from vectypes import int4, float2, float3, float4, float5, float44, RayTuple
from utils import clampf

@numba.njit(fastmath=True, nogil=True, cache=True)
def cmax2(f: float2) -> float:
    return max(f[0], f[1])

@numba.njit(fastmath=True, nogil=True, cache=True)
def cmin2(f: float2) -> float:
    return min(f[0], f[1])

@numba.njit(fastmath=True, nogil=True, cache=True)
def lerp3(a:float3,b:float3,f:float):
    #a + (b-a) * f
    return add3(a, scale3(sub3(b,a), f))

@numba.njit(fastmath=True, nogil=True, cache=True)
def lerp4(a:float4,b:float4,f:float) -> float4:
    #a + (b-a) * f
    return add4(a, scale4(sub4(b,a), f))

@numba.njit(fastmath=True, nogil=True, cache=True)
def lerp(a:float,b:float,f:float):
    return (a+((b-a)*f))


@numba.njit(fastmath=True, nogil=True, cache=True)
def lerp2(a:float2,b:float2,f:float):
    return add2(a, scale2(sub2(b,a), f))

@numba.njit(fastmath=True, nogil=True, cache=True)
def scale3(a:float3,f:float):
    return (a[0]*f,a[1]*f,a[2]*f)

@numba.njit(fastmath=True, nogil=True, cache=True)
def scale4(a:float4,f:float4):
    return (a[0]*f,a[1]*f,a[2]*f,a[3]*f)

@numba.njit(fastmath=True, nogil=True, cache=True)
def scale2(a:float2,f:float):
    return (a[0]*f,a[1]*f)

@numba.njit(fastmath=True, nogil=True, cache=True)
def mul2(a:float2,b:float2):
    return (a[0]*b[0], a[1]*b[1])

@numba.njit(fastmath=True, nogil=True, cache=True)
def div2(a:float2,b:float2):
    return (a[0]/b[0], a[1]/b[1])

@numba.njit(fastmath=True, nogil=True, cache=True)
def offset2(a:float2,f:float):
    return (a[0]+f,a[1]+f)

@numba.njit(fastmath=True, nogil=True, cache=True)
def offset3(a:float3,f:float):
    return (a[0]+f,a[1]+f,a[2]+f)

@numba.njit(fastmath=True, nogil=True, cache=True)
def add4(a:float4,b:float4):
    return (a[0]+b[0],a[1]+b[1],a[2]+b[2],a[3]+b[3])

@numba.njit(fastmath=True, nogil=True, cache=True)
def add3(a:float3,b:float3):
    return (a[0]+b[0],a[1]+b[1],a[2]+b[2])

@numba.njit(fastmath=True, nogil=True, cache=True)
def add2(a:float2,b:float2):
    return (a[0]+b[0],a[1]+b[1])

@numba.njit(fastmath=True, nogil=True, cache=True)
def neg2(a:float2):
    return (-a[0],-a[1])

@numba.njit(fastmath=True, nogil=True, cache=True)
def sub3(a:float3,b:float3):
    return (a[0]-b[0],a[1]-b[1],a[2]-b[2])

@numba.njit(fastmath=True, nogil=True, cache=True)
def sub4(a:float4,b:float4):
    return (a[0]-b[0],a[1]-b[1],a[2]-b[2],a[3]-b[3])


@numba.njit(fastmath=True, nogil=True, cache=True)
def sub2(a:float2,b:float2):
    return (a[0]-b[0], a[1]-b[1],)


@numba.njit(fastmath=True, nogil=True, cache=True)
def frac2(f2: float2) -> float2:
    return sub2(f2, (math.floor(f2[0]), math.floor(f2[1])))

@numba.njit(fastmath=True, nogil=True, cache=True)
def len2(vec):
    """ Calculate the norm of a 3d vector. """
    return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])

@numba.njit(forceinline=True, nogil=True, cache=True)
def normalize(vec):
    norm_ = len2(vec)
    if norm_ < 1e-6:
        return (0, 0)
    else:
        return (vec[0] / norm_, vec[1] / norm_)


@numba.njit(fastmath=True, nogil=True, cache=True)
def sign(x):
    if x >= 0:
        return 1
    else:
        return -1
    
@numba.njit(fastmath=True, nogil=True, cache=True)
def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]

@numba.njit(fastmath=True, nogil=True, cache=True)
def clip_min(p_min, p_max, frustum):
    frustum_inv = 1.0 / frustum
    c0 = cross2((1.0, frustum_inv), (p_max[0], p_max[2]))
    c1 = cross2((1.0, frustum_inv), (p_min[0], p_min[2]))
    return 1.0 - (c0 / (c0 - c1))

@numba.njit(fastmath=True, nogil=True, cache=True)
def clip_max(p_min, p_max, frustum):
    frustum_inv = 1.0 / frustum
    c0 = cross2((1.0, frustum_inv), (p_max[0], p_max[2]))
    c1 = cross2((1.0, frustum_inv), (p_min[0], p_min[2]))
    return c1 / (c1 - c0)

@numba.njit(fastmath=True, nogil=True, cache=True)
def get_world_bounds_clipping_cam_space(p_min, p_max, frustum_bound_min, frustum_bound_max):
    # returns (clipped: bool, min_lerp: float, max_lerp: float)
    if p_min[0] > p_min[2] * frustum_bound_max:
        if p_max[0] > p_max[2] * frustum_bound_max:
            return True, 0.0, 1.0  # both above frustum
        min_lerp = clip_min(p_min, p_max, frustum_bound_max)
        if p_max[0] < p_max[2] * frustum_bound_min:
            max_lerp = clip_max(p_min, p_max, frustum_bound_min)
        else:
            max_lerp = 1.0
    elif p_max[0] > p_max[2] * frustum_bound_max:
        max_lerp = clip_max(p_min, p_max, frustum_bound_max)
        if p_min[0] < p_min[2] * frustum_bound_min:
            min_lerp = clip_min(p_min, p_max, frustum_bound_min)
        else:
            min_lerp = 0.0
    else:
        if p_min[0] < p_min[2] * frustum_bound_min:
            if p_max[0] < p_max[2] * frustum_bound_min:
                return True, 0.0, 1.0  # both below frustum
            min_lerp = clip_min(p_min, p_max, frustum_bound_min)
            max_lerp = 1.0
        elif p_max[0] < p_max[2] * frustum_bound_min:
            max_lerp = clip_max(p_min, p_max, frustum_bound_min)
            min_lerp = 0.0
        else:
            min_lerp = 0.0
            max_lerp = 1.0
    return False, min_lerp, max_lerp


@numba.njit(forceinline=True, nogil=True, cache=True)
def matmult(mat: float44, vec: float4):
    x, y, z, w = vec
    # extract the elements in row-column form. (matrix is stored column first)
    a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, a41, a42, a43, a44 = mat
    return (
        x * a11 + y * a21 + z * a31 + w * a41,
        x * a12 + y * a22 + z * a32 + w * a42,
        x * a13 + y * a23 + z * a33 + w * a43,
        x * a14 + y * a24 + z * a34 + w * a44,
    )


#@numba.njit(fastmath=True, nogil=True, cache=True)
def setup_projected_plane_params(
        world_to_screen_mat: float44,
        ray_start: float2,
        ray_dir: float2,
        world_max_y: float,
        #voxel_scale: int,
        y_axis: int) -> typing.Tuple[float3, float3, float3]:
    plane_start_bottom = (ray_start[0], 0.0, ray_start[1], 1)
    plane_start_top = (ray_start[0], world_max_y, ray_start[1], 1)
    plane_ray_direction = (ray_dir[0], 0.0, ray_dir[1], 0)

    full_plane_start_top_projected = matmult(world_to_screen_mat, plane_start_top)
    full_plane_start_bot_projected = matmult(world_to_screen_mat, plane_start_bottom)
    full_plane_ray_direction_projected = matmult(world_to_screen_mat, plane_ray_direction)
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


@numba.njit(fastmath=True, nogil=True, cache=True)
def clip_homogeneous_camera_space_line(near: float, a: float5, b: float5) -> tuple[bool, float4, float4]:
    # clip to camera space
    (ax,ay,az,au,av) = a
    (bx,by,bz,bu,bv) = b

    if (ay < near):
        if (by < near):
            return False, (ax,az,au,av),(bx,bz,bu,bv)
        
        v = (by-near) / (by - ay)
        (cx,cz,cu,cv) = lerp4((bx,bz,bu,bv), (ax,az,au,av), v)
        return True, (cx,cz,cu,cv),(bx,bz,bu,bv)
    elif (by < near):
        v = (ay-near) / (ay - by)
        (cx,cz,cu,cv) = lerp4((ax,az,au,av), (bx,bz,bu,bv), v)
        return True, (ax,az,au,av),(cx,cz,cu,cv)
    else:
        return True, (ax,az,au,av),(bx,bz,bu,bv)

@numba.njit(fastmath=True, nogil=True, cache=True)
def is_pixel_set(seen_pixel_cache: np.ndarray[typing.Any, np.uint8], y: int) -> int:
    byte_idx = y >> 3
    bit_idx = y & 0b111
    return seen_pixel_cache[byte_idx] & (1<<bit_idx)
    #return seen_pixel_cache[y] 

@numba.njit(fastmath=True, nogil=True, cache=True)
def mark_pixel(seen_pixel_cache: np.ndarray[typing.Any, np.uint8], y: int):
    byte_idx = y >> 3
    bit_idx = y & 0b111
    seen_pixel_cache[byte_idx] |= (1<<bit_idx)
    
    #seen_pixel_cache[y] = 1

@numba.njit(fastmath=True, nogil=True, cache=True)
def get_texel_for_v(colors, base_idx, num_colors, color_v):
    return colors[base_idx + int(color_v * num_colors)]

@numba.njit(fastmath=True, nogil=True, cache=True)
def scale_color(col: int, depth: float):
    #depth = depth*32.0
    #depth = depth/32.0
    #depth = math.sqrt(depth)
    depth *= 4

    #tot = 1.0-(depth/FAR_CLIP_PLANE)

    r,g,b = ((col>>16)&0xFF)/255.0, ((col>>8)&0xFF)/255.0, ((col>>0)&0xFF)/255.0
    r = clampf(r*depth, 0.0, 1.0)
    g = clampf(g*depth, 0.0, 1.0)
    b = clampf(b*depth, 0.0, 1.0)
    #g *= depth
    #b *= depth
    ir = int(r*255.0)
    ig = int(g*255.0)
    ib = int(b*255.0)
    return (0xFF<<24)|(ir<<16)|(ig<<8)|(ib<<0)

#@numba.njit(fastmath=True, nogil=True, cache=True)
def fill_raybuffer_col(cam_space_top: float4, cam_space_bot: float4,
                       cur_next_free_pix_min: int, cur_next_free_pix_max: int, 
                       original_next_free_pix_min: int, original_next_free_pix_max: int,
                       seen_pixel_cache: np.ndarray[typing.Any, np.dtype[np.int8]], 
                       pix_arr_col: np.ndarray[typing.Any, np.dtype[np.uint32]],
                       texture: np.ndarray[1024, np.dtype[np.uint32]],
                       use_flat_col: bool, flat_col: np.uint32
                       ) -> typing.Tuple[int, int]:
    

    #color = colors[color_idx_in_span]
    # calculate the position in the ray buffer
    one_over_z_top = 1.0 / cam_space_top[1]
    one_over_z_bot = 1.0 / cam_space_bot[1]
    u_top = cam_space_top[2]
    v_top = cam_space_top[3]
    u_bot = cam_space_bot[2]
    v_bot = cam_space_bot[3]
    u_over_z_top = u_top * one_over_z_top
    v_over_z_top = v_top * one_over_z_top
    u_over_z_bot = u_bot * one_over_z_bot
    v_over_z_bot = v_bot * one_over_z_bot
    ray_buffer_bounds_float_min = cam_space_top[0] * one_over_z_top
    ray_buffer_bounds_float_max = cam_space_bot[0] * one_over_z_bot


    # flip min and max if necessary
    if ray_buffer_bounds_float_max < ray_buffer_bounds_float_min:
        ray_buffer_bounds_float_min, ray_buffer_bounds_float_max = ray_buffer_bounds_float_max, ray_buffer_bounds_float_min
        one_over_z_top,one_over_z_bot = one_over_z_bot,one_over_z_top
        u_over_z_top,u_over_z_bot = u_over_z_bot,u_over_z_top
        v_over_z_top,v_over_z_bot = v_over_z_bot,v_over_z_top

    original_ray_buffer_bounds_min = ray_buffer_bounds_float_min
    #original_ray_buffer_bounds_max = ray_buffer_bounds_float_max

    num_pix = (ray_buffer_bounds_float_max+1) - ray_buffer_bounds_float_min
    one_over_z_per_pix = (one_over_z_bot-one_over_z_top) / num_pix
    du_over_z_per_pix = (u_over_z_bot-u_over_z_top)/num_pix
    dv_over_z_per_pix = (v_over_z_bot-v_over_z_top)/num_pix


    # round to an integer position
    ray_buffer_bounds_min = round(ray_buffer_bounds_float_min)
    ray_buffer_bounds_max = round(ray_buffer_bounds_float_max)

    # check if within screen-space drawable bounds
    if (ray_buffer_bounds_max >= original_next_free_pix_min and ray_buffer_bounds_float_min <= original_next_free_pix_max):
        # if this visible chunk touches the top screen bound
        # shrink top of frustum as much as possible
        if ray_buffer_bounds_min <= cur_next_free_pix_min:
            # simple and works, but doesn't continuously shrink the top
            ray_buffer_bounds_min = cur_next_free_pix_min


            cur_next_free_pix_min = ray_buffer_bounds_max+1
        
            if ray_buffer_bounds_max >= cur_next_free_pix_min:
                cur_next_free_pix_min = ray_buffer_bounds_max+1
                while cur_next_free_pix_min <= original_next_free_pix_max and is_pixel_set(seen_pixel_cache, cur_next_free_pix_min) > 0: #seen_pixel_cache[cur_next_free_pix_min] > 0:
                    cur_next_free_pix_min += 1

        # if this visible chunk touches the bottom screen bound
        # shrink bottom frustum as much as possible
        if ray_buffer_bounds_max >= cur_next_free_pix_max:
            ray_buffer_bounds_max = cur_next_free_pix_max
            cur_next_free_pix_max = ray_buffer_bounds_min - 1
            if ray_buffer_bounds_min <= cur_next_free_pix_max:
                cur_next_free_pix_max = ray_buffer_bounds_min-1
                while cur_next_free_pix_max >= original_next_free_pix_min and is_pixel_set(seen_pixel_cache, cur_next_free_pix_max) > 0: #seen_pixel_cache[cur_next_free_pix_max] > 0:
                    cur_next_free_pix_max -= 1


        # now draw visible portion of the chunk
        dy = len(pix_arr_col)-1
        for y in range(ray_buffer_bounds_min, ray_buffer_bounds_max+1):
            if is_pixel_set(seen_pixel_cache, y) == 0: #seen_pixel_cache[y] == 0:
                mark_pixel(seen_pixel_cache, y)
                dy_for_depth = y - original_ray_buffer_bounds_min
                inv_z = one_over_z_top + one_over_z_per_pix * dy_for_depth
                z = 1.0/inv_z
                u_over_z = u_over_z_top + (dy_for_depth * du_over_z_per_pix)
                v_over_z = v_over_z_top + (dy_for_depth * dv_over_z_per_pix)

                u = int(u_over_z * z * 32) & 31
                v = int(v_over_z * z * 32) & 31


                col = texture[u*32+v]
                if use_flat_col:
                    col = flat_col

                pix_arr_col[(dy-y)] = col
                
    return cur_next_free_pix_min, cur_next_free_pix_max

# calculate DDA information for a given start position and direction
@numba.njit(fastmath=True, nogil=True, cache=True)
def make_ray(start, dir) -> tuple:
    position = (math.floor(start[0]), math.floor(start[1]))
    eps = .0000001
    absDir = (abs(dir[0]), abs(dir[1]))
    t_delta = (1 / max(eps, absDir[0]), 1 / max(eps, absDir[1]))

    sign_dir = (sign(dir[0]), sign(dir[1]))
    step = (int(sign_dir[0]), int(sign_dir[1]))

    t_max = mul2(t_delta, 
                   offset2(
                        add2(
                            mul2(sign_dir, neg2(frac2(start))),
                            scale2(sign_dir,  0.5)),
                        0.5)) # * t_delta
    
    if (t_max[0] - t_delta[0]) > (t_max[1] - t_delta[1]):
        # entered via X boundary
        enter_side_flag = X_SIDE
    else:
        # entered via Y boundary
        enter_side_flag = Z_SIDE
    

    intersection_distances = (cmax2(sub2(t_max, t_delta)), cmin2(t_max))
    return (
        position, 
        step, start, 
        dir, 
        t_delta, 
        t_max, 
        intersection_distances,
    ),  enter_side_flag

# returns a new ray tuple, plus the remaining x and y steps
@numba.njit(forceinline=True, fastmath=True, nogil=True, cache=True)
def step_ray(ray: RayTuple, ray_origin: float2, ray_dir:float2, rem_x_steps: int, rem_y_steps: int) -> tuple[RayTuple, int, int, float]:
    #(intersection_distances, t_max, t_delta, position, step) = ray
    (position, step, start, dir, t_delta, t_max, intersection_distances) = ray
     
        
    if t_max[0] < t_max[1]:
        crossed_boundary_distance = t_max[0]
        t_max = add2(t_max, (t_delta[0], 0))
        position = add2(position, (step[0], 0))
        rem_x_steps -= 1
        hit = ray_origin[1] + ray_dir[1] * crossed_boundary_distance
    else:
        crossed_boundary_distance = t_max[1]
        t_max = add2(t_max, (0, t_delta[1]))
        position = add2(position, (0, step[1]))
        rem_y_steps -= 1
        hit = ray_origin[0] + ray_dir[0] * crossed_boundary_distance
    
    intersection_distances_cur = crossed_boundary_distance

    if t_max[0] < t_max[1]:
        intersection_distances_next = t_max[0]

    else:
        intersection_distances_next = t_max[1]
    
    intersection_distances = (intersection_distances_cur, intersection_distances_next)
    new_ray = (position, step, start, dir, t_delta, t_max, intersection_distances)
    return new_ray, rem_x_steps, rem_y_steps


FLOAT_EPS = np.finfo(np.float32).eps

X_SIDE = 0
Z_SIDE = 1

frustum_cull = False

MAP_SIZE = 32 # 512
#@numba.njit(fastmath=True, nogil=True, cache=False)
def ray_loop(ray_origin:float2, ray_dir:float2,
             near_clip: float, far_clip: float, world_max_y: int, 
             plane_start_bot: float3, plane_start_top: float3, plane_ray_dir: float3, 
             one_over_world_max_y: float, 
             seen_pixel_cache: np.ndarray[typing.Any, np.dtype[np.uint8]], 
             ray_buffer_col: np.ndarray[typing.Any, np.dtype[np.uint32]],
             original_next_free_pix_min: int, original_next_free_pix_max: int,
             ray: RayTuple, prev_enter_side,
             camera_pos_y: float,
             wall_tex: np.ndarray[1024, np.uint32],
             flat_tex: np.ndarray[1024, np.uint32],
             iteration_direction: int, use_flat_col: bool
):
    

    camera_pos_y_normalized = camera_pos_y/world_max_y
    in_start_cell = True
    
    ((position_x, position_z), 
     _, 
     _,
     _, _, _, 
     (cur_intersection_distance, next_intersection_distance)) = ray
    
    (ray_dir_x, ray_dir_z) = ray_dir
    (ray_origin_x, ray_origin_z) = ray_origin

    default_exit_u =  1.0 if (ray_dir_x >= 0) else 0.0
    default_exit_v =  1.0 if (ray_dir_z >= 0) else 0.0

    default_start_u = 1.0 - default_exit_u
    default_start_v = 1.0 - default_exit_v

    
    x_steps = 32
    y_steps = 32


    cur_next_free_pix_min, cur_next_free_pix_max = original_next_free_pix_min, original_next_free_pix_max

    wall_u = 0
    side = prev_enter_side

    frustum_bounds_min = original_next_free_pix_min - 0.501
    frustum_bounds_max = original_next_free_pix_max + 0.501

    frustum_dir_max_world = FLOAT_EPS
    frustum_dir_min_world = FLOAT_EPS
    while True:
        if(cur_intersection_distance >= far_clip):
            #draw_skybox(cur_next_free_pix_min, cur_next_free_pix_max, seen_pixel_cache, ray_buffer, i)
            break # no lod stuff :)
        if x_steps < 0 or y_steps < 0:
            break

        
        has_ceil = ((position_x^position_z)&1) == 1
        has_floor = ((position_x^position_z)&1) == 0

        col_spans = [[16,14], [2,0]]
        if has_floor:
            col_spans[1][0] = 1
        if has_ceil:
            col_spans[0][1] = 15
        world_col_min = col_spans[1][1]
        world_col_max = col_spans[0][0]


        plane_ray_dir_times_dist_x = scale3(plane_ray_dir, cur_intersection_distance)
        cam_space_min_last = add3(plane_start_bot, plane_ray_dir_times_dist_x)
        cam_space_max_last = add3(plane_start_top, plane_ray_dir_times_dist_x)
        plane_ray_dir_times_dist_y = scale3(plane_ray_dir, next_intersection_distance)
        cam_space_min_next = add3(plane_start_bot, plane_ray_dir_times_dist_y)
        cam_space_max_next = add3(plane_start_top, plane_ray_dir_times_dist_y)

        if frustum_cull:
            world_bounds_min = 0.0
            world_bounds_max = world_max_y

            if frustum_dir_max_world != FLOAT_EPS:
                # distance to top of frustum?
                dist_top = next_intersection_distance if frustum_dir_max_world > 0.0 else cur_intersection_distance
                dist_bot = cur_intersection_distance if frustum_dir_min_world < 0.0 else next_intersection_distance
                new_max = camera_pos_y + frustum_dir_max_world * dist_top
                new_min = camera_pos_y + frustum_dir_min_world * dist_bot
                if new_min > world_bounds_max or new_max < world_bounds_min:
                    return  # frustum went out of world entirely
                if world_col_min > new_max or world_col_max < new_min:
                    p_x_steps = x_steps 
                    ray, x_steps, y_steps = step_ray(ray, ray_origin, ray_dir, x_steps, y_steps)
                    ((position_x, position_z),_,_,_,_,_,(cur_intersection_distance, next_intersection_distance)) = ray
                    if x_steps != p_x_steps:
                        side = X_SIDE
                    else:
                        side = Z_SIDE

                    continue  # column doesn't overlap writable world bounds
                world_bounds_min = new_min
                world_bounds_max = new_max

            if cur_intersection_distance > 8.0 and frustum_dir_max_world == FLOAT_EPS:
                clipped_last, clip_last_min_lerp, clip_last_max_lerp = get_world_bounds_clipping_cam_space(
                    cam_space_min_last, cam_space_max_last, frustum_bounds_min, frustum_bounds_max)
                clipped_next, clip_next_min_lerp, clip_next_max_lerp = get_world_bounds_clipping_cam_space(
                    cam_space_min_next, cam_space_max_next, frustum_bounds_min, frustum_bounds_max)

                if clipped_last:
                    if clipped_next:
                        return  # skybox
                    else:
                        world_bounds_min = lerp(0.0, world_max_y, clip_next_min_lerp)
                        world_bounds_max = lerp(0.0, world_max_y, clip_next_max_lerp)
                        frustum_dir_max_world = (world_bounds_max - camera_pos_y) / next_intersection_distance
                        frustum_dir_min_world = (world_bounds_min - camera_pos_y) / next_intersection_distance
                        min_clip = lerp3(cam_space_min_next, cam_space_max_next, clip_next_min_lerp)
                        max_clip = lerp3(cam_space_min_next, cam_space_max_next, clip_next_max_lerp)
                        cam_space_clipped_min = min_clip[0] / min_clip[2]
                        cam_space_clipped_max = max_clip[0] / max_clip[2]
                        if cam_space_clipped_max < cam_space_clipped_min:
                            cam_space_clipped_min, cam_space_clipped_max = cam_space_clipped_max, cam_space_clipped_min
                else:
                    if clipped_next:
                        world_bounds_min = lerp(0.0, world_max_y, clip_last_min_lerp)
                        world_bounds_max = lerp(0.0, world_max_y, clip_last_max_lerp)
                        frustum_dir_max_world = (world_bounds_max - camera_pos_y) / cur_intersection_distance
                        frustum_dir_min_world = (world_bounds_min - camera_pos_y) / cur_intersection_distance
                        min_clip = lerp3(cam_space_min_last, cam_space_max_last, clip_last_min_lerp)
                        max_clip = lerp3(cam_space_min_last, cam_space_max_last, clip_last_max_lerp)
                        cam_space_clipped_min = min_clip[0] / min_clip[2]
                        cam_space_clipped_max = max_clip[0] / max_clip[2]
                        if cam_space_clipped_max < cam_space_clipped_min:
                            cam_space_clipped_min, cam_space_clipped_max = cam_space_clipped_max, cam_space_clipped_min
                    else:
                        if clip_last_min_lerp < clip_next_min_lerp:
                            world_bounds_min = lerp(0.0, world_max_y, clip_last_min_lerp)
                            frustum_dir_min_world = (world_bounds_min - camera_pos_y) / cur_intersection_distance
                        else:
                            world_bounds_min = lerp(0.0, world_max_y, clip_next_min_lerp)
                            frustum_dir_min_world = (world_bounds_min - camera_pos_y) / next_intersection_distance

                        if clip_last_max_lerp > clip_next_max_lerp:
                            world_bounds_max = lerp(0.0, world_max_y, clip_last_max_lerp)
                            frustum_dir_max_world = (world_bounds_max - camera_pos_y) / cur_intersection_distance
                        else:
                            world_bounds_max = lerp(0.0, world_max_y, clip_next_max_lerp)
                            frustum_dir_max_world = (world_bounds_max - camera_pos_y) / next_intersection_distance

                        min_clip_a = lerp3(cam_space_min_last, cam_space_max_last, clip_last_min_lerp)
                        max_clip_a = lerp3(cam_space_min_last, cam_space_max_last, clip_last_max_lerp)
                        min_clip_b = lerp3(cam_space_min_next, cam_space_max_next, clip_next_min_lerp)
                        max_clip_b = lerp3(cam_space_min_next, cam_space_max_next, clip_next_max_lerp)

                        min_last = min_clip_a[0] / min_clip_a[2]
                        max_last = max_clip_a[0] / max_clip_a[2]
                        min_next = min_clip_b[0] / min_clip_b[2]
                        max_next = max_clip_b[0] / max_clip_b[2]

                        if max_next < min_next: min_next, max_next = max_next, min_next
                        if max_last < min_last: min_last, max_last = max_last, min_last

                        cam_space_clipped_min = min(min_last, min_next)
                        cam_space_clipped_max = max(max_last, max_next)

                world_bounds_min = math.floor(world_bounds_min)
                world_bounds_max = math.ceil(world_bounds_max)

                writable_min_pixel = int(math.floor(cam_space_clipped_min))
                writable_max_pixel = int(math.ceil(cam_space_clipped_max))

                if writable_max_pixel < cur_next_free_pix_min or writable_min_pixel > cur_next_free_pix_max:
                    return  # skybox

                if writable_min_pixel > cur_next_free_pix_min:
                    cur_next_free_pix_min = writable_min_pixel
                    while cur_next_free_pix_min <= original_next_free_pix_max and is_pixel_set(seen_pixel_cache, cur_next_free_pix_min):
                        cur_next_free_pix_min += 1
                if writable_max_pixel < cur_next_free_pix_max:
                    cur_next_free_pix_max = writable_max_pixel
                    while cur_next_free_pix_max >= original_next_free_pix_min and is_pixel_set(seen_pixel_cache, cur_next_free_pix_max):
                        cur_next_free_pix_max -= 1

                if cur_next_free_pix_min > cur_next_free_pix_max:
                    return  # skybox


        
        hit_x = ray_origin_x + ray_dir_x*cur_intersection_distance
        hit_z = ray_origin_z + ray_dir_z*cur_intersection_distance
        next_hit_x = ray_origin_x + ray_dir_x*next_intersection_distance
        next_hit_z = ray_origin_z + ray_dir_z*next_intersection_distance

        break_ray_loop = False
        
        top_down = iteration_direction == 1

        span_idx_start = 0 if top_down else len(col_spans)-1
        span_idx_end = len(col_spans) if top_down else -1

        if side == X_SIDE:
            wall_u = hit_z - math.floor(hit_z)
            flat_u = default_start_u
            flat_v = wall_u
        else:
            wall_u = hit_x - math.floor(hit_x)
            flat_u = wall_u
            flat_v = default_start_v



        p_x_steps = x_steps 
        _, nx_steps, _ = step_ray(ray, ray_origin, ray_dir, x_steps, y_steps)
        if nx_steps != p_x_steps:
            next_side = X_SIDE
        else:
            next_side = Z_SIDE
        

        if next_side == X_SIDE:
            exit_flat_u = default_exit_u 
            exit_flat_v = next_hit_z - math.floor(next_hit_z)
        else:
            exit_flat_u = next_hit_x - math.floor(next_hit_x)
            exit_flat_v = default_exit_v



        for span_idx in range(span_idx_start, span_idx_end, iteration_direction): #span_idx in range(num_spans):
            
            #if in_start_cell:
            #    continue
            element_bounds_max = col_spans[span_idx][0]
            element_bounds_min = col_spans[span_idx][1]


            # calculate the position, between 0 and 1, in world space
            # of the top and bottom of the solid chunk of voxels for this column
            portion_top = element_bounds_max * one_over_world_max_y
            portion_bottom = element_bounds_min * one_over_world_max_y 

            # now lerp the camera space top and bottom ray positions with the portions that
            # correspond to the bottom and top of the solid voxel chunk

            # this gives us a camera space position for the voxel chunk

            cam_space_front_top = lerp3(cam_space_min_last, cam_space_max_last, portion_top)
            (xt,yt,zt) = cam_space_front_top

            cam_space_front_bottom = lerp3(cam_space_min_last, cam_space_max_last, portion_bottom)
            (xb,yb,zb) = cam_space_front_bottom

            total_steps = (32-x_steps)+(32-y_steps)
            step_shade = int((total_steps/64)*255)
            step_color = (0xFF<<24)|(step_shade<<16)|(step_shade<<8)|(step_shade<<0)





            if not in_start_cell:
                (onscreen, cam_space_clipped_front_top, cam_space_clipped_front_bot) = clip_homogeneous_camera_space_line(
                    near_clip,
                    (xt,yt,zt,wall_u,0.0),
                    (xb,yb,zb,wall_u,1.0)
                )

                if onscreen:

                    (cur_next_free_pix_min, cur_next_free_pix_max) = fill_raybuffer_col(
                        cam_space_clipped_front_top, cam_space_clipped_front_bot,
                        cur_next_free_pix_min, cur_next_free_pix_max,
                        original_next_free_pix_min, original_next_free_pix_max,
                        seen_pixel_cache, 
                        ray_buffer_col, wall_tex, use_flat_col, step_color
                    )


                    # if the frustum doesn't cover a pixel, break out of the loop
                    if cur_next_free_pix_min > cur_next_free_pix_max:
                        break_ray_loop = True
                        break

            
            if (portion_top < camera_pos_y_normalized):
                flat_exit_cam_space = lerp3(cam_space_min_next, cam_space_max_next, portion_top)
                flat_enter_cam_space = cam_space_front_top
            else:
                flat_exit_cam_space = lerp3(cam_space_min_next, cam_space_max_next, portion_bottom)
                flat_enter_cam_space = cam_space_front_bottom
            
            (ax,ay,az) = flat_exit_cam_space  # far side of a flat, use exit_flat_u/exit_flat_v
            (bx,by,bz) = flat_enter_cam_space  # near side of a flat, use flat_u/flat_v
            
            (onscreen, cam_space_clipped_secondary_a, cam_space_clipped_secondary_b) = clip_homogeneous_camera_space_line(
                near_clip,
                (ax,ay,az, exit_flat_u, exit_flat_v),
                (bx,by,bz, flat_u, flat_v)
            )
            if onscreen:
                (cur_next_free_pix_min, cur_next_free_pix_max) = fill_raybuffer_col(
                    cam_space_clipped_secondary_a, cam_space_clipped_secondary_b,
                    cur_next_free_pix_min, cur_next_free_pix_max,
                    original_next_free_pix_min, original_next_free_pix_max,
                    seen_pixel_cache, 
                    ray_buffer_col, flat_tex, use_flat_col, step_color
                )
                if cur_next_free_pix_min > cur_next_free_pix_max:
                    break_ray_loop = True
                    break
                    
            if break_ray_loop:
                break
        if break_ray_loop:
            break


            #print(floor_enter_transformed, floor_exit_transformed)

        p_x_steps = x_steps 
        ray, x_steps, y_steps = step_ray(ray, ray_origin, ray_dir, x_steps, y_steps)
        ((position_x, position_z),_,_,_,_,_,(cur_intersection_distance, next_intersection_distance)) = ray
        if x_steps != p_x_steps:
            side = X_SIDE
        else:
            side = Z_SIDE
        

        in_start_cell = False




# sets up the remaining information for each ray
#@numba.njit(parallel=True, fastmath=True, nogil=True, cache=False)
def execute_rays_in_segment(
    rays_in_segment: int,
    ray_buffer_base_offset: int,
    cam_local_plane_ray_min: float2,
    cam_local_plane_ray_max: float2,
    axis_mapped_to_y: int,
    original_next_free_pix_min: int,
    original_next_free_pix_max: int,
    world_to_screen_mat: float44,
    camera_position: float3,
    near_clip: float,
    far_clip: float, 
    iteration_direction: int,
    seen_pixel_cache: np.ndarray[typing.Any, np.dtype[np.uint8]],
    ray_buffer: np.ndarray[(typing.Any, typing.Any), np.dtype[np.uint32]],
    world_max_y: int,
    wall_tex: np.ndarray[1024, np.uint32],
    flat_tex: np.ndarray[1024, np.uint32], use_flat_col: bool
):
    
    voxel_scale = 1
    #world_max_y = 512

    one_over_world_max_y = 1/world_max_y 


    cam_pos_xz = (camera_position[0], camera_position[2])
    for ray_in_segment_idx in numba.prange(rays_in_segment):

        end_ray_lerp = ray_in_segment_idx / rays_in_segment
        
        cam_local_plane_ray_direction = lerp2(cam_local_plane_ray_min, cam_local_plane_ray_max, end_ray_lerp)

        norm_ray_dir = normalize(cam_local_plane_ray_direction)
        ray, prev_side = make_ray(cam_pos_xz, norm_ray_dir)

        ray_column = ray_in_segment_idx + ray_buffer_base_offset

        seen_pixel_col = seen_pixel_cache[ray_in_segment_idx]


        (plane_start_bottom_projected,
        plane_start_top_projected,
        plane_ray_direction_projected) = setup_projected_plane_params(
            world_to_screen_mat,
            cam_pos_xz, norm_ray_dir,
            world_max_y, 
            axis_mapped_to_y
        )

        ray_loop(cam_pos_xz, norm_ray_dir,
                 near_clip, far_clip, world_max_y,
                 plane_start_bottom_projected, plane_start_top_projected, plane_ray_direction_projected,
                 one_over_world_max_y,
                 seen_pixel_col, ray_buffer[ray_column],
                 original_next_free_pix_min, original_next_free_pix_max,
                 ray, prev_side, camera_position.y, wall_tex, flat_tex, iteration_direction, use_flat_col)
    #return res



#@numba.njit(fastmath=True, nogil=True, cache=False)
def raycast_segments(
    segment_ray_counts: int4,
    segment_next_free_pixel_mins: int4,
    segment_next_free_pixel_maxs: int4,
    segment_cam_local_plane_ray_mins: float4,
    segment_cam_local_plane_ray_maxs: float4,
    camera_pos: float3, 
    camera_near_clip: float,
    camera_far_clip: float,
    iteration_direction: int,
    world_to_screen_mat: float44,
    top_down_pix_arr: np.ndarray, 
    left_right_pix_arr: np.ndarray, 
    world_max_y: int,
    full_seen_pixel_cache: np.ndarray[(typing.Any, typing.Any), np.uint8],
    skybox_col_int: int,
    wall_tex: np.ndarray[(32,32), np.uint32],
    flat_tex: np.ndarray[(32,32), np.uint32], 
    use_flat_col: bool
    ):

    top_down_pix_arr.fill(skybox_col_int)
    left_right_pix_arr.fill(skybox_col_int)
    
    
    #
    total_rays = 0 #sum([s.ray_count for s in segments])
    for s in segment_ray_counts:
        total_rays += s


    for segment_index in range(4):
        if segment_index != 0:
            continue
        #segment = segments[segment_index]
        segment_ray_count = segment_ray_counts[segment_index]
        if segment_ray_count == 0:
            continue

        segment_ray_index_offset = 0
        if segment_index == 1:
            segment_ray_index_offset = segment_ray_counts[0]
        if segment_index == 3:
            segment_ray_index_offset = segment_ray_counts[2]
            
        if segment_index < 2:
            pix_arr = top_down_pix_arr
        else:
            pix_arr = left_right_pix_arr

        # 0,1 are mapped to y, 2,3 are mapped to x
        axis_mapped_to_y = 0 if segment_index > 1 else 1

        next_free_pixel_min = segment_next_free_pixel_mins[segment_index]
        next_free_pixel_max = segment_next_free_pixel_maxs[segment_index]
        cam_local_plane_ray_min = segment_cam_local_plane_ray_mins[segment_index]
        cam_local_plane_ray_max = segment_cam_local_plane_ray_maxs[segment_index]

        local_seen_pixel_cache = full_seen_pixel_cache[0:segment_ray_count]
        local_seen_pixel_cache.fill(0)

        execute_rays_in_segment(
            segment_ray_count, segment_ray_index_offset, 
            cam_local_plane_ray_min, cam_local_plane_ray_max,
            axis_mapped_to_y, next_free_pixel_min, next_free_pixel_max,
            world_to_screen_mat, camera_pos, camera_near_clip, camera_far_clip, iteration_direction,
            local_seen_pixel_cache, pix_arr, world_max_y, wall_tex, flat_tex, use_flat_col)


@numba.njit(parallel=True, fastmath=True, nogil=True, cache=False)
def transpose_and_create_bytes(np_arr: np.ndarray, output_arr: np.ndarray, x1,y1,w,h):
    for x in numba.prange(w):
        col = np_arr[x+x1] # get col in src array

        output_arr[x*h:x*h+h] = col[y1:y1+h]
        #continue
        #for y in range(h):
        #    rgba = col[y+y1] # get pixel in column
        #    #output_arr[(y*w+x)] = rgba # write out into rows..
        #    output_arr[x*h+y] = rgba # write out into cols..


@numba.njit(parallel=True, fastmath=True, nogil=True, cache=False)
def transpose_buffers(np_arrs, output_arrs, dims):
    for idx in numba.prange(4):
        (x1,y1,w,h) = dims[idx]
        transpose_and_create_bytes(np_arrs[idx], output_arrs[idx], x1,y1,w,h)

@numba.njit(parallel=False, fastmath=True, nogil=True)
def rasterize_seg01(src_pix_arr: np.ndarray, output_arr: np.ndarray, seg_verts, screen_width, screen_height, offset_x, offset_y, seg_w, seg_h):
    # offset into the source array

    min_y = seg_verts[0].y
    max_y = seg_verts[1].y
    vp_x = seg_verts[0].x

    left_x_at_top = vp_x
    right_x_at_top = vp_x
    left_x_at_bot = seg_verts[2].x
    right_x_at_bot = seg_verts[1].x
    if min_y > max_y:
        max_y,min_y = min_y,max_y
        left_x_at_top,left_x_at_bot = left_x_at_bot,left_x_at_top
        right_x_at_top,right_x_at_bot = right_x_at_bot,right_x_at_top

    # we have two edges, one that goes up/down left, one that goes up/down right


    left_dx = left_x_at_bot - left_x_at_top
    right_dx = right_x_at_bot - right_x_at_top

    dy = max_y - min_y
    left_dx_per_y = left_dx / dy
    right_dx_per_y = right_dx / dy

    left_x = left_x_at_top
    right_x = right_x_at_top

    screen_min_y = max(math.floor(min_y), 0)
    screen_max_y = min(math.floor(max_y)+1, screen_height)
    #screen_min_y 
    for y in range(screen_min_y, screen_max_y):#math.floor(min_y), math.floor(max_y)+1):
        y_portion = (1 - ((y-min_y) / (max_y+1-min_y)))
        #y_int = int(y_portion * 255)
        y_coord = int(y_portion*seg_h) + offset_y
        if y_coord < 0:
            y_coord = 0
        elif y_coord >= len(src_pix_arr[0]):
            y_coord = len(src_pix_arr[0])-1

        screen_min_x = max(math.floor(left_x), 0)
        screen_max_x = min(math.floor(right_x)+1, screen_width)
        for x in range(screen_min_x, screen_max_x):#math.floor(left_x), math.floor(right_x)+1):
            x_portion = (x-left_x) / (right_x+1-left_x)
            #x_int = int(x_portion * 255)
            x_coord = int(x_portion*seg_w)+offset_x
            if x_coord < 0:
                x_coord = 0
            elif x_coord >= len(src_pix_arr):
                x_coord = len(src_pix_arr)-1
            texel = src_pix_arr[x_coord][y_coord]

            output_arr[x][y] = texel #(0xFF<<24)|(x_int<<16)|(y_int<<8)|(0<<0)
            #output_arr[x*2+1][y*2] = texel #(0xFF<<24)|(x_int<<16)|(y_int<<8)|(0<<0)
            #output_arr[x*2][y*2+1] = texel #(0xFF<<24)|(x_int<<16)|(y_int<<8)|(0<<0)
            #output_arr[x*2+1][y*2+1] = texel #(0xFF<<24)|(x_int<<16)|(y_int<<8)|(0<<0)

        left_x += left_dx_per_y
        right_x += right_dx_per_y

@numba.njit(parallel=False, fastmath=True, nogil=True)
def rasterize_seg23(src_pix_arr: np.ndarray, output_arr: np.ndarray, seg_verts, screen_width, screen_height, offset_x, offset_y, seg_w, seg_h):
    # offset into the source array
    min_x = seg_verts[0].x
    max_x = seg_verts[1].x
    vp_y = seg_verts[0].y


    top_y_at_left = vp_y
    bot_y_at_left = vp_y
    top_y_at_right = seg_verts[2].y
    bot_y_at_right = seg_verts[1].y
    if min_x > max_x:
        max_x,min_x = min_x,max_x
        top_y_at_left,top_y_at_right = top_y_at_right,top_y_at_left
        bot_y_at_left,bot_y_at_right = bot_y_at_right,bot_y_at_left

    # we have two edges, one that goes up/down left, one that goes up/down right


    top_dy = top_y_at_right - top_y_at_left
    bot_dy = bot_y_at_right - bot_y_at_left

    dx = max_x - min_x
    top_dy_per_x = top_dy / dx
    bot_dy_per_x = bot_dy / dx

    top_y = top_y_at_left
    bot_y = bot_y_at_left

    screen_min_x = max(math.floor(min_x), 0)
    screen_max_x = min(math.floor(max_x)+1, screen_width)
    #screen_min_y 
    for x in range(screen_min_x, screen_max_x): #math.floor(min_x), math.floor(max_x)+1):
        y_portion = (1 - ((x-min_x) / (max_x+1-min_x)))
        #y_int = int(y_portion * 255)
        y_coord = int(y_portion*seg_h) + offset_y
        if y_coord < 0:
            y_coord = 0
        elif y_coord >= len(src_pix_arr[0]):
            y_coord = len(src_pix_arr[0])-1

        screen_min_y = max(math.floor(top_y), 0)
        screen_max_y = min(math.floor(bot_y)+1, screen_height)
        for y in range(screen_min_y, screen_max_y):#math.floor(top_y), math.floor(bot_y)+1):
            x_portion = (y-top_y) / (bot_y+1-top_y)
            #x_int = int(x_portion * 255)
            x_coord = int(x_portion*seg_w)+offset_x
            if x_coord < 0:
                x_coord = 0
            elif x_coord >= len(src_pix_arr):
                x_coord = len(src_pix_arr)-1
            texel = src_pix_arr[x_coord][y_coord]

            output_arr[x][y] = texel #(0xFF<<24)|(x_int<<16)|(y_int<<8)|(0<<0)

        top_y += top_dy_per_x
        bot_y += bot_dy_per_x
