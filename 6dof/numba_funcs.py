import numpy as np
import math
import numba 

import typing
from vectypes import int4, float2, float3, float4, float44, RayTuple

@numba.njit(cache=True)
def cmax2(f: float2) -> float:
    return max(f[0], f[1])

@numba.njit(cache=True)
def cmin2(f: float2) -> float:
    return min(f[0], f[1])

@numba.njit(cache=True)
def lerp3(a:float3,b:float3,f:float):
    #a + (b-a) * f
    return add3(a, scale3(sub3(b,a), f))

@numba.njit(cache=True)
def lerp(a:float,b:float,f:float):
    return (a+((b-a)*f))


@numba.njit(cache=True)
def lerp2(a:float2,b:float2,f:float):
    return add2(a, scale2(sub2(b,a), f))

@numba.njit(cache=True)
def scale3(a:float3,f:float):
    return (a[0]*f,a[1]*f,a[2]*f)

@numba.njit(cache=True)
def scale2(a:float2,f:float):
    return (a[0]*f,a[1]*f)

@numba.njit(cache=True)
def mul2(a:float2,b:float2):
    return (a[0]*b[0], a[1]*b[1])

@numba.njit(cache=True)
def div2(a:float2,b:float2):
    return (a[0]/b[0], a[1]/b[1])

@numba.njit(cache=True)
def offset2(a:float2,f:float):
    return (a[0]+f,a[1]+f)

@numba.njit(cache=True)
def offset3(a:float3,f:float):
    return (a[0]+f,a[1]+f,a[2]+f)

@numba.njit(cache=True)
def add3(a:float3,b:float3):
    return (a[0]+b[0],a[1]+b[1],a[2]+b[2])

@numba.njit(cache=True)
def add2(a:float2,b:float2):
    return (a[0]+b[0],a[1]+b[1])

@numba.njit(cache=True)
def neg2(a:float2):
    return (-a[0],-a[1])

@numba.njit(cache=True)
def sub3(a:float3,b:float3):
    return (a[0]-b[0],a[1]-b[1],a[2]-b[2])

@numba.njit(cache=True)
def sub2(a:float2,b:float2):
    return (a[0]-b[0], a[1]-b[1],)


@numba.njit(cache=True)
def frac2(f2: float2) -> float2:
    return sub2(f2, (math.floor(f2[0]), math.floor(f2[1])))

@numba.njit(cache=True)
def len2(vec):
    """ Calculate the norm of a 3d vector. """
    return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])

@numba.njit(cache=True)
def normalize2(vec, result):
    norm_ = len2(vec)
    if norm_ < 1e-6:
        result[0] = 0.
        result[1] = 0.
    else:
        result[0] = vec[0] / norm_
        result[1] = vec[1] / norm_


@numba.njit(cache=True)
def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


@numba.njit(forceinline=True, cache=True)
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


@numba.njit(cache=True)
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


@numba.njit(cache=True)
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

@numba.njit(cache=True)
def is_pixel_set(seen_pixel_cache: np.ndarray[typing.Any, np.uint8], y: int) -> int:
    byte_idx = y >> 3
    bit_idx = y & 0b111
    return seen_pixel_cache[byte_idx] & (1<<bit_idx)
    #return seen_pixel_cache[y] 

@numba.njit(cache=True)
def mark_pixel(seen_pixel_cache: np.ndarray[typing.Any, np.uint8], y: int):
    byte_idx = y >> 3
    bit_idx = y & 0b111
    seen_pixel_cache[byte_idx] |= (1<<bit_idx)
    
    #seen_pixel_cache[y] = 1


@numba.njit(forceinline=True, cache=True)
def fill_raybuffer_col(cam_space_top: float2, cam_space_bot: float2,
                       cur_next_free_pix_min: int, cur_next_free_pix_max: int, 
                       original_next_free_pix_min: int, original_next_free_pix_max: int,
                       seen_pixel_cache: np.ndarray[typing.Any, np.dtype[np.int8]], 
                       pix_arr: np.ndarray[typing.Any, np.dtype[np.uint32]], col: int,
                       color: np.uint32) -> typing.Tuple[int, int]:
    # calculate the position in the ray buffer
    ray_buffer_bounds_float_min = cam_space_top[0] / cam_space_top[1]
    ray_buffer_bounds_float_max = cam_space_bot[0] / cam_space_bot[1]

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
        pix_arr_col = pix_arr[col]
        dy = len(pix_arr_col)-1
        for y in range(ray_buffer_bounds_min, ray_buffer_bounds_max+1):
            if is_pixel_set(seen_pixel_cache, y) == 0: #seen_pixel_cache[y] == 0:
                #seen_pixel_cache[y] = 1
                mark_pixel(seen_pixel_cache, y)

                pix_arr_col[dy-y] = color

    return cur_next_free_pix_min, cur_next_free_pix_max



# returns a new ray tuple, plus the remaining x and y steps
@numba.njit(forceinline=True, cache=True)
def step_ray(ray: RayTuple, rem_x_steps: int, rem_y_steps: int) -> tuple[RayTuple, int, int]:
    (intersection_distances, t_max, t_delta, position, step) = ray
    if t_max[0] < t_max[1]:
        crossed_boundary_distance = t_max[0]
        t_max = add2(t_max, (t_delta[0], 0))
        position = add2(position, (step[0], 0))
        rem_x_steps -= 1
    else:
        crossed_boundary_distance = t_max[1]
        t_max = add2(t_max, (0, t_delta[1]))
        position = add2(position, (0, step[1]))
        rem_y_steps -= 1
    
    intersection_distances_x = crossed_boundary_distance
    if t_max[0] < t_max[1]:
        intersection_distances_y = t_max[0]
    else:
        intersection_distances_y = t_max[1]
    
    intersection_distances = (intersection_distances_x, intersection_distances_y)
    new_ray = (intersection_distances, t_max, t_delta, position, step)
    return new_ray, rem_x_steps, rem_y_steps


@numba.njit(cache=True)
def ray_loop(far_clip: float, world_max_y: int, 
             height_map: np.ndarray[typing.Any, np.dtype[np.uint16]], color_map: np.ndarray[typing.Any, np.dtype[np.uint32]],
              plane_start_bot: float3, plane_start_top: float3, plane_ray_dir: float3, 
              one_over_world_max_y: float, 
              seen_pixel_cache: np.ndarray[typing.Any, np.dtype[np.uint8]], 
              pix_arr: np.ndarray[typing.Any, np.dtype[np.uint32]], col: int,
              original_next_free_pix_min: int, original_next_free_pix_max: int,
              ray: RayTuple,
              camera_pos_y_normalized: float):
    
    FLOAT_EPS = np.finfo(np.float32).eps
    
    ((intersection_distances_x,intersection_distances_y),
     _, _, 
     (position_x, position_y), 
     (step_x, step_y)) = ray
    


    while position_x < 0 or position_x > 1023 or position_y < 0 or position_y > 1023:
        if(intersection_distances_x >= far_clip):
            #draw_skybox(cur_next_free_pix_min, cur_next_free_pix_max, seen_pixel_cache, pix_arr, i)
            break # no lod stuff :)

        ray, _, _ = step_ray(ray, 1, 1)
        ((intersection_distances_x, intersection_distances_y), _, _,
         (position_x, position_y), _) = ray

    if step_x > 0:
        # bounding is end of map, 1023
        x_steps = max(0, 1023-position_x) # if position x is 1024, 0 steps.  if position_x = -10, 1033 steps
    else:
        x_steps = max(0, position_x)

    if step_y > 0:
        y_steps = max(0, 1023-position_y)
    else:
        y_steps = max(0, position_y)


    cur_next_free_pix_min, cur_next_free_pix_max = original_next_free_pix_min, original_next_free_pix_max


    frustum_bounds_min = cur_next_free_pix_min - 0.501
    frustum_bounds_max = cur_next_free_pix_max + 0.501

    frustum_dir_max_world = FLOAT_EPS
    frustum_dir_min_world = FLOAT_EPS

    while True:
        if(intersection_distances_x >= far_clip):
            #draw_skybox(cur_next_free_pix_min, cur_next_free_pix_max, seen_pixel_cache, pix_arr, i)
            break # no lod stuff :)
        if x_steps < 0 or y_steps < 0:
            break

        index = position_y*1024+position_x
        height = height_map[index]
        color = color_map[index]
        element_bounds_min = 0
        element_bounds_max = max(1, height)

        column_top = element_bounds_max
        column_bot = element_bounds_min

        # world-space frustum culling
        world_bounds_min = 0
        world_bounds_max = world_max_y

        if frustum_dir_max_world != FLOAT_EPS:
            dist_top = intersection_distances_y if frustum_dir_max_world > 0 else intersection_distances_x
            dist_bot = intersection_distances_y if frustum_dir_min_world < 0 else intersection_distances_x
            new_max = position_y + frustum_dir_max_world * dist_top
            new_min = position_y + frustum_dir_min_world * dist_bot

            if new_min > world_bounds_max or new_max < world_bounds_min:
                # frustum went out of the world entirely
                # no skybox :)
                break
            
            if column_bot > new_max or column_top < new_min:
                # this column doesn't overlap writable world bounds
                ray, x_steps, y_steps = step_ray(ray, x_steps, y_steps)
                ((intersection_distances_x,intersection_distances_y),
                _, _, 
                (position_x, position_y), 
                (step_x, step_y)) = ray
                if x_steps < 0 or y_steps < 0:
                    break
            
            world_bounds_min = new_min
            world_bounds_max = new_max

        # plane_start_bot, plane_start_top are the camera space positions of the start of the
        # ray plane for this raycast

        # calculate the position along the ray via adding the ray_direction * dist to these values
        plane_ray_dir_times_dist_x = scale3(plane_ray_dir, intersection_distances_x)
        cam_space_min_last = add3(plane_start_bot, plane_ray_dir_times_dist_x)
        cam_space_max_last = add3(plane_start_top, plane_ray_dir_times_dist_x)
        plane_ray_dir_times_dist_y = scale3(plane_ray_dir, intersection_distances_y)
        cam_space_min_next = add3(plane_start_bot, plane_ray_dir_times_dist_y)
        cam_space_max_next = add3(plane_start_top, plane_ray_dir_times_dist_y)

        """
        if intersection_distances_x > 2 and frustum_dir_max_world == FLOAT_EPS:
            # determine world/clip space min/max of the writable frustum
            clipped_last, clip_last_min_lerp, clip_last_max_lerp = get_world_bounds_clipping_cam_space(
                cam_space_min_last, cam_space_max_last,
                frustum_bounds_min, frustum_bounds_max
            )
            clipped_next, clip_next_min_lerp, clip_next_max_lerp = get_world_bounds_clipping_cam_space(
                cam_space_min_next, cam_space_max_next,
                frustum_bounds_min, frustum_bounds_max
            )

            if clipped_last:
                if clipped_next:
                    # end of ray
                    break
                else:
                    world_bounds_min = lerp(0, world_max_y, clip_next_min_lerp)
                    world_bounds_max = lerp(0, world_max_y, clip_next_max_lerp)

                    frustum_dir_max_world = (world_bounds_max - position_y) / intersection_distances_y
                    frustum_dir_min_world = (world_bounds_min - position_y) / intersection_distances_y

                    min_clip = lerp3(cam_space_min_next, cam_space_max_next, clip_next_min_lerp)
                    max_clip = lerp3(cam_space_min_next, cam_space_max_next, clip_next_max_lerp)
                    cam_space_clipped_min = min_clip[0] / min_clip[2]
                    cam_space_clipped_max = max_clip[0] / max_clip[2]
                    if cam_space_clipped_max < cam_space_clipped_min:
                        cam_space_clipped_min, cam_space_clipped_max = cam_space_clipped_max, cam_space_clipped_min
            else:
                if clipped_next:
                    world_bounds_min = lerp(0, world_max_y, clip_last_min_lerp)
                    world_bounds_max = lerp(0, world_max_y, clip_last_max_lerp)

                    frustum_dir_max_world = (world_bounds_max - position_y) / intersection_distances_x
                    frustum_dir_min_world = (world_bounds_min - position_y) / intersection_distances_x

                    min_clip = lerp3(cam_space_min_last, cam_space_max_last, clip_last_min_lerp)
                    max_clip = lerp3(cam_space_min_last, cam_space_max_last, clip_last_max_lerp)

                    cam_space_clipped_min = min_clip[0] / min_clip[2]
                    cam_space_clipped_max = max_clip[0] / max_clip[2]

                    if cam_space_clipped_max < cam_space_clipped_min:
                        cam_space_clipped_min, cam_space_clipped_max = cam_space_clipped_max, cam_space_clipped_min
                else:
                    if clip_last_min_lerp < clip_next_min_lerp:
                        world_bounds_min = lerp(0, world_max_y, clip_last_min_lerp)
                        frustum_dir_min_world = (world_bounds_min - position_y) / intersection_distances_x
                    else:
                        world_bounds_min = lerp(0, world_max_y, clip_next_min_lerp)
                        frustum_dir_min_world = (world_bounds_min - position_y) / intersection_distances_y
                    
                    if clip_last_max_lerp > clip_next_max_lerp:
                        world_bounds_max = lerp(0, world_max_y, clip_last_max_lerp)
                        frustum_dir_max_world = (world_bounds_max - position_y) / intersection_distances_x
                    else:
                        world_bounds_max = lerp(0, world_max_y, clip_last_min_lerp)
                        frustum_dir_max_world = (world_bounds_max - position_y) / intersection_distances_y

                    min_clip_a = lerp(cam_space_min_last, cam_space_max_last, clip_last_min_lerp)
                    max_clip_a = lerp(cam_space_min_last, cam_space_max_last, clip_last_max_lerp)

                    min_clip_b = lerp(cam_space_min_next, cam_space_max_next, clip_next_min_lerp)
                    max_clip_b = lerp(cam_space_min_next, cam_space_max_next, clip_next_max_lerp)

                    min_next = min_clip_b[0] / min_clip_b[2]
                    min_last = min_clip_a[0] / min_clip_a[2]
                    max_next = max_clip_b[0] / max_clip_b[2]
                    max_last = max_clip_a[0] / max_clip_a[2]

                    if max_next < min_next:
                        min_next,max_next = max_next,min_next
                    if max_last < min_last:
                        min_last,max_last = max_last,min_last

                    cam_space_clipped_min = min(min_last, min_next)
                    cam_space_clipped_max = max(max_last, max_next)

            world_bounds_min = math.floor(world_bounds_min)
            world_bounds_max = math.ceil(world_bounds_max)

            writable_min_pixel = int(math.floor(cam_space_clipped_min))
            writable_max_pixel = int(math.ceil(cam_space_clipped_max))

            if writable_max_pixel < cur_next_free_pix_min or writable_min_pixel > cur_next_free_pix_max:
                # break out :)
                return
            if writable_min_pixel > cur_next_free_pix_min:
                cur_next_free_pix_min = writable_min_pixel

        """


        #for chunk_idx in range(1):
            #start = chunks[chunk_idx][0]

        #element_bounds_min = chunks[chunk_idx][0]
        #element_bounds_max = chunks[chunk_idx][1] 

        # calculate the position, between 0 and 1, in world space
        # of the top and bottom of the solid chunk of voxels for this column
        portion_top = element_bounds_max * one_over_world_max_y
        portion_bottom = element_bounds_min * one_over_world_max_y #element_bounds_min * one_over_world_max_y

        # now lerp the camera space top and bottom ray positions with the portions that
        # correspond to the bottom and top of the solid voxel chunk

        # this gives us a camera space position for the voxel chunk
        cam_space_front_bottom = lerp3(cam_space_min_last, cam_space_max_last, portion_bottom)
        cam_space_front_top = lerp3(cam_space_min_last, cam_space_max_last, portion_top)


        (onscreen, cam_space_clipped_front_top, cam_space_clipped_front_bot) = clip_homogeneous_camera_space_line(
            cam_space_front_top, cam_space_front_bottom,
        )

        if onscreen:
            (cur_next_free_pix_min, cur_next_free_pix_max) = fill_raybuffer_col(
                cam_space_clipped_front_top, cam_space_clipped_front_bot,
                cur_next_free_pix_min, cur_next_free_pix_max,
                original_next_free_pix_min, original_next_free_pix_max,
                seen_pixel_cache, pix_arr, col, color)


            # if the frustum doesn't cover a pixel, break out of the loop
            if cur_next_free_pix_min > cur_next_free_pix_max:
                break

                
        # figure out if we drap the top or bottom
        if (portion_top < camera_pos_y_normalized):
            # maybe we can ignored this check
            #if element_bounds_max > world_bounds_max:
            #    continue 
            cam_space_secondary_a = lerp3(cam_space_min_next, cam_space_max_next, portion_top)
            cam_space_secondary_b = cam_space_front_top
        else:
            cam_space_secondary_a = lerp3(cam_space_min_next, cam_space_max_next, portion_bottom)
            cam_space_secondary_b = cam_space_front_bottom
        
        (onscreen, cam_space_clipped_secondary_a, cam_space_clipped_secondary_b) = clip_homogeneous_camera_space_line(
            cam_space_secondary_a,
            cam_space_secondary_b,
        )

        if onscreen:
            (cur_next_free_pix_min, cur_next_free_pix_max) = fill_raybuffer_col(
                cam_space_clipped_secondary_a, cam_space_clipped_secondary_b,
                cur_next_free_pix_min, cur_next_free_pix_max,
                original_next_free_pix_min, original_next_free_pix_max,
                seen_pixel_cache, pix_arr, col, color)

        # step the ray to the next grid intersection
        ray, x_steps, y_steps = step_ray(ray, x_steps, y_steps)
        ((intersection_distances_x, intersection_distances_y), 
         _, _, 
         (position_x, position_y), _) = ray
        if x_steps < 0 or y_steps < 0:
            break


# calculate DDA information for a given start position and direction
@numba.njit(cache=True)
def make_ray(start, dir) -> tuple:
    position = (math.floor(start[0]), math.floor(start[1]))
    eps = .0000001
    absDir = (abs(dir[0]), abs(dir[1]))
    t_delta = (1 / max(eps, absDir[0]), 1 / max(eps, absDir[1]))

    sign_dir = (sign(dir[0]), sign(dir[1]))
    step = (int(sign_dir[0]), int(sign_dir[1]))

    t_max = mul2(t_delta, 
                   offset2(
                        add2(mul2(sign_dir, neg2(frac2(start))),
                                scale2(sign_dir,  0.5)),
                        0.5)) # * t_delta

    intersection_distances = (cmax2(sub2(t_max, t_delta)), cmin2(t_max))
    return (
        position, 
        step, start, 
        dir, 
        t_delta, 
        t_max, 
        intersection_distances
    )



#def draw_skybox(next_free_pix_min, next_free_pix_max, seen_pixel_cache, pix_arr, ray_column_off):
#    #pass
#    for y in range(next_free_pix_min, next_free_pix_max):
#        if is_pixel_set(seen_pixel_cache, y) == 0: #seen_pixel_cache[y] == 0:
#            pix_arr[ray_column_off, y] = skybox_col_int


# sets up the remaining information for each ray
@numba.njit(parallel=True,cache=True)
def execute_rays_in_segment(
    rays_in_segment: int,
    ray_buffer_base_offset: int,
    overall_ray_offset: int,
    cam_local_plane_ray_min: float2,
    cam_local_plane_ray_max: float2,
    axis_mapped_to_y: int,
    original_next_free_pix_min: int,
    original_next_free_pix_max: int,
    world_to_screen_mat: float44,
    camera_position: float3,
    far_clip: float,
    height_map: np.ndarray[typing.Any, np.dtype[np.uint16]], 
    color_map: np.ndarray[typing.Any, np.dtype[np.uint32]],
    full_seen_pixel_cache: np.ndarray[(typing.Any, typing.Any), np.dtype[np.uint8]],
    ray_buffer: np.ndarray[(typing.Any, typing.Any), np.dtype[np.uint32]],
    world_max_y: int,
):
    
    voxel_scale = 1

    one_over_world_max_y = 1/world_max_y 
    camera_pos_y_normalized = camera_position.y / world_max_y

    for ray_in_segment_idx in numba.prange(rays_in_segment):

        end_ray_lerp = ray_in_segment_idx / rays_in_segment
        
        cam_local_plane_ray_direction = lerp2(cam_local_plane_ray_min, cam_local_plane_ray_max, end_ray_lerp)
        #cam_local_plane_ray_direction = pyglet.math.Vec2.lerp(
        #    cam_local_plane_ray_min, cam_local_plane_ray_max, end_ray_lerp
        #)
        norm_ray_dir = np.zeros(2)
        normalize2(cam_local_plane_ray_direction, norm_ray_dir) #cam_local_plane_ray_direction.normalize()
        (ray_position, ray_step, ray_start,
         ray_dir, ray_t_delta, ray_t_max, ray_intersection_distances
        ) = make_ray(
            np.array([camera_position.x, camera_position.z]),
            norm_ray_dir)

        ray_column = ray_in_segment_idx + ray_buffer_base_offset

        seen_pixel_cache = full_seen_pixel_cache[overall_ray_offset + ray_in_segment_idx]


        # small offset to the frustums to prevent a division by zero in the clipping algorithm
        #frustum_bounds_min = cur_next_free_pix_min - .501
        #frustum_bounds_max = cur_next_free_pix_max + .501
        (plane_start_bottom_projected,
        plane_start_top_projected,
        plane_ray_direction_projected) = setup_projected_plane_params(
            world_to_screen_mat,
            ray_start, ray_dir,
            world_max_y, 
            axis_mapped_to_y
        )

        #intersection_distances_x, intersection_distances_y = ray.intersection_distances

        #step_x, step_y = ray.step
        #t_delta_x, t_delta_y = ray.t_delta
        #t_max_x, t_max_y = ray.t_max
        #position_x,position_y = ray.position
        
        #original_next_free_pix_min = original_next_free_pixel_min
        #original_next_free_pix_max = .original_next_free_pixel_max
        ray = (ray_intersection_distances,
               ray_t_max,
               ray_t_delta,
               ray_position,
               ray_step)
        # (intersection_distances: float2, t_max: float2, t_delta: float2, position: int2, step: int2)

        ray_loop(far_clip, world_max_y, height_map, color_map,
                 plane_start_bottom_projected, plane_start_top_projected, plane_ray_direction_projected,
                 one_over_world_max_y, seen_pixel_cache, 
                 ray_buffer, ray_column,
                 original_next_free_pix_min, original_next_free_pix_max,
                 ray, camera_pos_y_normalized)
    #return res



@numba.njit()
def raycast_segments(
    segment_ray_counts: int4,
    segment_next_free_pixel_mins: int4,
    segment_next_free_pixel_maxs: int4,
    segment_cam_local_plane_ray_mins: float4,
    segment_cam_local_plane_ray_maxs: float4,
    camera_pos: float3, 
    camera_far_clip: float,
    world_to_screen_mat: float44,
    heights, colors, 
    top_down_pix_arr: np.ndarray, 
    left_right_pix_arr: np.ndarray, 
    world_max_y: int,
    aligned_bytes_per_seen_pixel_cache_column: int,
    skybox_col_int: int):

    top_down_pix_arr.fill(skybox_col_int)
    left_right_pix_arr.fill(skybox_col_int)
    
    #
    total_rays = 0 #sum([s.ray_count for s in segments])
    for s in segment_ray_counts:
        total_rays += s

    full_seen_pixel_cache = np.zeros((total_rays, aligned_bytes_per_seen_pixel_cache_column), dtype=np.uint8) 



    overall_ray_offset = 0
    for segment_index in range(4):
        #segment = segments[segment_index]
        segment_ray_count = segment_ray_counts[segment_index]
        if segment_ray_count == 0:
            continue

        segment_ray_index_offset = 0
        if segment_index == 1:
            segment_ray_index_offset = segment_ray_counts[0]
        if segment_index == 3:
            segment_ray_index_offset = segment_ray_counts[2]\
            
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

        execute_rays_in_segment(
            segment_ray_count, segment_ray_index_offset, overall_ray_offset, 
            cam_local_plane_ray_min, cam_local_plane_ray_max,
            axis_mapped_to_y, next_free_pixel_min, next_free_pixel_max,
            world_to_screen_mat, camera_pos, camera_far_clip,
            heights, colors, full_seen_pixel_cache, pix_arr, world_max_y)
        overall_ray_offset += segment_ray_count


@numba.njit(parallel=True, cache=True)
def transpose_and_create_bytes(np_arr: np.ndarray, dims):
    
    x1,y1,w,h = dims
    output_arr = np.empty((w*h*3), dtype=np.uint8)
    #w = x2-x1
    #h = y2-y1
    for y in numba.prange(h):
        for x in range(w):
            rgba = np_arr[x+x1][y+y1]
            #output_arr[y][x][0] = rgba&0xFF
            #output_arr[y][x][1] = (rgba>>8)&0xFF
            #output_arr[y][x][2] = (rgba>>16)&0xFF
            output_arr[(y*w+x)*3+0] = rgba&0xFF
            output_arr[(y*w+x)*3+1] = (rgba>>8)&0xFF
            output_arr[(y*w+x)*3+2] = (rgba>>16)&0xFF
    return output_arr
