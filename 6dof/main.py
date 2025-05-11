import gc
import typing

import moderngl
import pygame
import pygame.image


import math
from dataclasses import dataclass
import pyglet
import numpy as np
import numba

from utils import clampf, clampi, color_to_int, color_tuple_to_int, skybox_col_int, skybox_col
from vectypes import float2, float3
from camera import Camera 
from numba_funcs import raycast_segments, transpose_and_create_bytes
from vxl import load_voxlap_map
# CONSTANTS


# non-numba versions of these functions for AOT compilation
def lerp3(a:float3,b:float3,f:float):
    #a + (b-a) * f
    (ax,ay,az) = a
    (bx,by,bz) = b
    return (ax + (bx-ax)*f,
     ay + (by-ay)*f,
     az + (bz-az)*f)


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
RENDER_WIDTH = OUTPUT_WIDTH
RENDER_HEIGHT = OUTPUT_HEIGHT

MAX_RAYS_LEFT_RIGHT = 2*RENDER_WIDTH + RENDER_HEIGHT
MAX_RAYS_UP_DOWN = RENDER_WIDTH + 2*RENDER_HEIGHT
LEFT_RIGHT_RAY_BUFFER_DIMS = (MAX_RAYS_LEFT_RIGHT, RENDER_WIDTH)
TOP_DOWN_RAY_BUFFER_DIMS = (MAX_RAYS_UP_DOWN, RENDER_HEIGHT)

FULL_CIRCLE = 2.0 * math.pi        # 360 degrees
HALF_CIRCLE = FULL_CIRCLE/2.0      # 180 degrees
QUARTER_CIRCLE = HALF_CIRCLE/2.0   # 90 degrees
EIGHTH_CIRCLE = QUARTER_CIRCLE/2.0 # 45 degrees
THREE_QUARTERS_CIRCLE = 3.0 * QUARTER_CIRCLE
MOVE_SPEED = 30/1000
ANG_SPEED = .6/1000


NEAR_CLIP_PLANE = .05
FAR_CLIP_PLANE = 1024

# FILE IO, loading maps images, etc

def change_resolution(new_target_render_width, new_target_render_height):
    # recreate textures
    # reset pygame output window size?
    # recreate surfaces
    pass

T = typing.TypeVar('T')
def __load_transformed_img(file: str) -> typing.Generator[pygame.Color, typing.Any, typing.Any]:
    img = pygame.image.load(f"{file}")

    print(img.get_at((0,0)))
    w = img.get_width()
    h = img.get_height()
    if w != 512:
        img = pygame.transform.scale(img, (512, 512))
        w,h = 512,512

    for y in range(h):
        for x in range(w):
            yield img.get_at((y, x))



load_color_img = __load_transformed_img
load_grayscale_img = lambda f: (x[0] for x in __load_transformed_img(f))


# data types 




@dataclass
class Segment:
    index: int
    min_screen: float2
    max_screen: float2
    cam_local_plane_ray_min: float2
    cam_local_plane_ray_max: float2
    next_free_pixel_min: int
    next_free_pixel_max: int
    ray_count: int




# unused!
# we just fill the entire buffer with blue every frame instead of calling this
                        # B G R A



def normalize_radians(f):
    while f < 0:
        f += math.radians(360)
    while f > math.radians(360):
        f -= math.radians(360)
    return f

RENDER_MODE = 0
last_keys_down = set()
import copy 

def handle_input(camera: Camera, keys_down: set, dt: float):
    global RENDER_MODE, last_keys_down
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

    
    if pygame.K_n in keys_down and pygame.K_n not in last_keys_down:
        load_map()
    last_keys_down = copy.copy(keys_down)

    dpitch = 0
    if pygame.K_z in keys_down:
        dpitch = -dt * ANG_SPEED
    elif pygame.K_x in keys_down:
        dpitch = dt * ANG_SPEED
    


    # NOTE:
    # pitches around the camera's local x-axis
    # if we instead rotated around the world's global x-axis
    # our pitch would be very odd :)

    if dpitch != 0:
        s,c = math.sin(dpitch),math.cos(dpitch)
        t = 1-c
        x,y,z = camera.right

        rot_mat = pyglet.math.Mat3(
            t*x*x+c, t*x*y + z*s, t*x*z - y*s,
            t*x*y - z*s, t*y*y + c, t*y*z + x*s,
            t*x*z + y*s, t*y*z - x*s, t*z*z + c
        )

        camera.forward = (rot_mat @ camera.forward).normalize()
        camera.right = (rot_mat @ camera.right).normalize()
        camera.up = (rot_mat @ camera.up).normalize()
        #camera.set_dirty()

    dyaw = 0
    if pygame.K_LEFT in keys_down:
        dyaw = dt * ANG_SPEED
    elif pygame.K_RIGHT in keys_down:
        dyaw = -dt * ANG_SPEED

    # NOTE:
    # this rotates around the world y-axis
    # if instead, we'd like to rotate around the camera's local y axis, we must do something like the above transformation

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
        #camera.set_dirty()

    # NOTE: disable ROLL for now, it's tricky and our rendering probably won't work with it yet
    #droll = 0
    #if pygame.K_q in keys_down:
    #    droll = -0.3
    #elif pygame.K_f in keys_down:
    #    droll = .3

    #if droll != 0:
    #    sr,cr = math.sin(droll),math.cos(droll)
    #    rot_mat = pyglet.math.Mat3(
    #        cr, -sr, 0,
    #        -sr, cr, 0,
    #        0, 0, 1
    #    )
    #    #camera.forward = (rot_mat @ camera.forward).normalize()
    #    camera.right = (rot_mat @ camera.right).normalize()
    #    camera.up = (rot_mat @ camera.up).normalize()
    #    camera.set_dirty()

    
    if abs(camera.forward.y) < 0.001:
        newForward = pyglet.math.Vec3(camera.forward.x, sign(camera.forward.y)*.001, camera.forward.z).normalize()
        camera.set_forward(newForward)

    # NOTE: adjusts position by GLOBAL axes, not camera local
    new_pos = camera.pos
    if pygame.K_c in keys_down:
        new_pos = new_pos + pyglet.math.Vec3(0,1,0) * MOVE_SPEED * dt
    elif pygame.K_v in keys_down:
        new_pos = new_pos + pyglet.math.Vec3(0,1,0) * -MOVE_SPEED * dt

    # NOTE: adjusts position by camera local forward axis, not global
    if pygame.K_UP in keys_down:
        new_pos = new_pos + camera.forward * MOVE_SPEED * dt
    elif pygame.K_DOWN in keys_down:
        new_pos = new_pos + camera.forward * -MOVE_SPEED * dt
    
    camera.pos = new_pos

    if pygame.K_KP_6 in keys_down:
        RENDER_MODE = 0
    if pygame.K_KP_7 in keys_down:
        RENDER_MODE = 1
    elif pygame.K_KP_8 in keys_down:
        RENDER_MODE = 2
    elif pygame.K_KP_9 in keys_down:
        RENDER_MODE = 3
    

def calc_vanishing_point_world(camera: Camera):
    rot_y = 1 * (-camera.near_clip / -camera.forward.y)
    return camera.pos + pyglet.math.Vec3(0, rot_y, 0)


def project_vanishing_point_world_to_screen(camera: Camera, vp_world: pyglet.math.Vec3):
    forward = camera.forward
    right = camera.right
    up = camera.up

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
    dot = clampf(dot, -1, 1)
    return math.acos(dot) * 57.29
    

def vec2_signed_angle(frm: pyglet.math.Vec2, to: pyglet.math.Vec2) -> float:
    unsigned_angle = vec2_angle(frm, to)
    sgn = sign(frm.x * to.y - frm.y * to.x)
    return unsigned_angle * sgn


def transform_pixel(camera: Camera, pix: pyglet.math.Vec2):
    scaled_pix = pix / camera.dims
    off_pix = scaled_pix[0]-0.5, scaled_pix[1]-0.5
    homogenoeous_pix = pyglet.math.Vec4(off_pix[0]*2, off_pix[1]*2, 1, 1)

    forward = camera.forward
    right = camera.right
    up = camera.forward.cross(right)

    lookat_matrix = pyglet.math.Mat4(
        right.x, right.y, right.z, 0,
        up.x, up.y, up.z, 0,
        forward.x, forward.y, forward.z, 0,
        0, 0, 0, 1
    )

    matrix = camera.get_projection_matrix().__invert__()
    matrix = pyglet.math.Mat4.from_scale(pyglet.math.Vec3(1,1,-1)) @ matrix
    matrix = lookat_matrix @ matrix

    world_pix = matrix @ homogenoeous_pix
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
                           max_ray_count: int):
    secondary_axis = 1 - primary_axis

    segment = Segment(segment_index, (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0), 0, 0, 0)

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




def adjust_screen_pixel_for_mesh(screen_pixel: float2, screen_size: float2) -> float3:
        return (2* screen_pixel[0] / screen_size[0] - 1, 2 * screen_pixel[1]/screen_size[1] - 1, 0.5)

def create_vertfrag_shader(ctx: moderngl.Context, vertex_filepath: str, fragment_filepath: str) -> moderngl.Program:
        """
        Create a moderngl shader program containing the shaders at the given filepaths.
        """
        with open(vertex_filepath,'r') as f:
            vertex_src = f.read()
       
        with open(fragment_filepath,'r') as f:
            fragment_src = f.read()
       

        shader = ctx.program(vertex_shader=vertex_src, fragment_shader=fragment_src)
        return shader


class Texture:
    """
    Responsible for handling an OpenGL texture object.
    """
    def __init__(self, image: pygame.Surface, ctx: moderngl.Context) -> None:
        image = pygame.transform.flip(image, False, True)
        self.image_width, self.image_height = image.get_rect().size
        img_data = pygame.image.tobytes(image, "RGB")
        self.texture = ctx.texture(size=image.get_size(), components=3, data=img_data)
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)


    def update_from_bytes(self, img_data: bytes, viewport: tuple[int, int, int, int]) -> None:
        """
        Writes the contents of the pygame Surface to OpenGL texture. 
        """
        #pygame.surface
        if viewport is not None:
            self.texture.write(img_data, viewport=viewport)
        else:
            self.texture.write(img_data)


    def update(self, image: pygame.Surface, vflip: bool=False) -> None:
        """
        Writes the contents of the pygame Surface to OpenGL texture. 
        """
        if vflip:
            image = pygame.transform.flip(image, False, True)
        image_width, image_height = image.get_rect().size
        #pygame.surface
        img_data = pygame.image.tobytes(image, "RGB")
        self.texture.write(img_data)

    def use(self, _id: typing.Union[None, int] = None) -> None:
        """
        Use the texture object for rendering
        """
        if not _id:
            self.texture.use()  
        else:
            self.texture.use(_id)

maps = [#('skyline_test.png', 'skyline_test.png', True), 

        ('./maps/de_dust2x2.vxl', None, False),
        ('./maps/Alpine.vxl', None, False),
        ('./maps/Arab.vxl', None, False),
        ('./maps/AtlasNovus.vxl', None, False),
        ('./maps/LostValley.vxl', None, False),
        ('./maps/DragonsReach.vxl', None, False),
        ('./maps/cloudfall.vxl', None, False),
        ('./comanche_maps/C1W.png', './comanche_maps/D1.png', False),
        ('./comanche_maps/C2W.png', './comanche_maps/D2.png', False),
        ('./comanche_maps/C22W.png', './comanche_maps/D22.png', False),]
next_map = 0

def gen_height_color(height):
    if height > 200:
        return pygame.Color(255,255,255,255)
    elif height > 40:
        i = (height-40)/160
        (r,g,b) = lerp3((40,80, 5), (255,255,255), i)
        return pygame.Color(int(r),int(g),int(b),255)
    else:
        return pygame.Color(0x2B, 0x57, 0x70,0xFF)
        #color = color_tuple_to_int((0x1E, 0x4F, 0x20, 0xFF))

spans = None
colors = None
columns = None

def load_map():
    #global colors, heights, next_map
    global spans, colors, columns, next_map
    nmc, nmh, dyn_color = maps[next_map]
    
    next_map += 1
    


    if next_map >= len(maps):
        next_map = 0
    
    if colors is not None:
        colors.fill(0)
        spans.fill(0)
        columns.fill(0)
    else:
        colors = np.zeros((512,512,256), dtype=np.uint32)
        spans = np.zeros((512,512,32,4), dtype=np.uint32)
        columns = np.zeros((512,512,2), dtype=np.uint8)
    if ".vxl" in nmc:
        spans, colors, columns = load_voxlap_map(nmc, colors, spans, columns)
    else:
        #assert False
        height_pix = load_grayscale_img(nmh) #"D1.png") #"test_comanche_D.png")
        heights = np.array([x for x in height_pix], dtype=np.uint16)

        if dyn_color:
            raw_colors = np.array([color_to_int(gen_height_color(x)) for x in heights], dtype=np.uint32)
        else:
            color_pix = load_color_img(nmc) #"C1W.png") #"test_comanche_C.png")
            raw_colors = np.array([color_to_int(x) for x in color_pix], dtype=np.uint32)

        for idx in range(512*512):
            y = idx >> 9
            x = idx & 0b111111111
            columns[y][x][0] = heights[idx]
            columns[y][x][1] = 1 # 1 span
            spans[y][x][0][0] = heights[idx]
            spans[y][x][0][1] = 0 # ( or -1 ?? )
            spans[y][x][0][2] = 0
            spans[y][x][0][3] = 0
            rgba = raw_colors[idx]
            for yy in range(heights[idx]+1):
                #tmp_rgba = rgba
                #r = tmp_rgba&0xFF
                #g = (tmp_rgba>>8)&0xFF
                #b = (tmp_rgba>>16)&0xFF
                #tmp_rgba = color_tuple_to_int( tweak_color_rand_table[r], tweak_color_rand_table[g], tweak_color_rand_table[b], 255)



                colors[y][x][yy] = rgba
        
        pass




def main():
    global colors, heights
    load_map()

    pygame.init()
    pygame.font.init()

    pygame.display.set_mode((OUTPUT_WIDTH, OUTPUT_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    display_surf = pygame.display.get_surface()
    clock = pygame.time.Clock()

    running = True

    camera = Camera(RENDER_WIDTH, RENDER_HEIGHT, NEAR_CLIP_PLANE, FAR_CLIP_PLANE)

    left_right_draw_surf = pygame.Surface(LEFT_RIGHT_RAY_BUFFER_DIMS, depth=32)
    top_down_draw_surf = pygame.Surface(TOP_DOWN_RAY_BUFFER_DIMS, depth=32)

    left_right_pix_arr = np.full([MAX_RAYS_LEFT_RIGHT, RENDER_WIDTH], 
                                 skybox_col_int, dtype=np.uint32)
    top_down_pix_arr = np.full([MAX_RAYS_UP_DOWN, RENDER_HEIGHT], 
                               skybox_col_int, dtype=np.uint32)
    
    last_fps = 0
    keys_down = set()
    #my_font = pygame.font.SysFont('Comic Sans MS', 20)
    
    time = 0
    empty_surface = pygame.Surface((OUTPUT_WIDTH, OUTPUT_HEIGHT))
    empty_surface.fill(skybox_col)
    
    gl_ctx = moderngl.create_context()
    gl_ctx.enable(moderngl.BLEND)
    gl_ctx.enable(moderngl.NEAREST)
    gl_ctx.enable(moderngl.LINEAR)
    gl_ctx.blend_func = gl_ctx.SRC_ALPHA, gl_ctx.ONE_MINUS_SRC_ALPHA
    gl_ctx.disable(moderngl.CULL_FACE | moderngl.DEPTH_TEST)

        
        #self.shader_data = {}
    seg01_vf_shader = create_vertfrag_shader(gl_ctx, "./vert.glsl", "./seg01_frag.glsl")
    seg23_vf_shader = create_vertfrag_shader(gl_ctx, "./vert.glsl", "./seg23_frag.glsl")
    full_screen_shader = create_vertfrag_shader(gl_ctx, "./full_screen_vert.glsl", "./full_screen_frag.glsl")
        
    seg01_target_texture = Texture(pygame.Surface(TOP_DOWN_RAY_BUFFER_DIMS), gl_ctx)
    seg23_target_texture = Texture(pygame.Surface(LEFT_RIGHT_RAY_BUFFER_DIMS), gl_ctx)

    seg01_vf_shader = create_vertfrag_shader(gl_ctx, "./vert.glsl", "./seg01_frag.glsl")
    seg23_vf_shader = create_vertfrag_shader(gl_ctx, "./vert.glsl", "./seg23_frag.glsl")
    ray_buffer_shader = create_vertfrag_shader(gl_ctx, "./default_vert.glsl", "./default_frag.glsl")

    max_seg_width = max(MAX_RAYS_LEFT_RIGHT, MAX_RAYS_UP_DOWN)
    max_seg_height = max(RENDER_WIDTH, RENDER_HEIGHT)
    upload_arr = np.empty((max_seg_height*max_seg_width*3), dtype=np.uint8)

    max_screen_dim = max(camera.dims)
    total_bytes_per_column = math.ceil(max_screen_dim / 8)
    aligned_bytes_per_column = 1 << (total_bytes_per_column.bit_length())
    
    full_seen_pixel_cache = np.zeros((max(MAX_RAYS_LEFT_RIGHT, MAX_RAYS_UP_DOWN), aligned_bytes_per_column), dtype=np.uint8) 

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

        fps = clock.get_fps()
        last_ticks = clock.get_time()
        handle_input(camera, keys_down, last_ticks)

        camera.set_dirty()

        world_vp = calc_vanishing_point_world(camera)
        screen_vp = project_vanishing_point_world_to_screen(camera, world_vp)

        segments = [
            Segment(0, (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0), 0, 0, 0), Segment(1, (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0), 0, 0, 0), 
            Segment(2, (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0), 0, 0, 0), Segment(3, (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0), 0, 0, 0)
        ]
        

        max_td_ray_count = top_down_draw_surf.get_width()
        max_lr_ray_count = left_right_draw_surf.get_width()
        if screen_vp.y < camera.dims.y:
        #    # top segment
            segments[0] = get_segment_parameters(
                0,
                camera, screen_vp, camera.dims.y - screen_vp.y, 
                pyglet.math.Vec2(0, 1), 1, WORLD_MAX_Y, max_td_ray_count)

        if screen_vp.y > 0:
            # down segment
            segments[1] = get_segment_parameters(
                1,
                camera, screen_vp, screen_vp.y, 
                pyglet.math.Vec2(0, -1), 1, WORLD_MAX_Y, max_td_ray_count)

        if screen_vp.x < camera.dims.x:
        #    # right segment
            segments[2] = get_segment_parameters(
                2,
                camera, screen_vp,  camera.dims.x - screen_vp.x, 
                pyglet.math.Vec2(1, 0), 0, WORLD_MAX_Y, max_lr_ray_count)

        if screen_vp.x > 0:
        #    # left segment
            segments[3] = get_segment_parameters(
                3,
                camera, screen_vp, screen_vp.x, 
                pyglet.math.Vec2(-1, 0), 0, WORLD_MAX_Y, max_lr_ray_count)
        
        vp_x,vp_y = screen_vp
        screen_width, screen_height = camera.dims
        for segment_index in range(4):
            segment = segments[segment_index]
            if segment_index < 2:
                if segment_index == 0:
                    next_free_pixel_min = clampi(round(vp_y), 0, screen_height-1)
                    next_free_pixel_max = screen_height-1
                else:
                    next_free_pixel_min = 0
                    next_free_pixel_max = clampi(round(vp_y), 0, screen_height-1)
            else:
                if segment_index == 3:
                    next_free_pixel_min = 0
                    next_free_pixel_max = clampi(round(vp_x), 0, screen_width-1)
                else:
                    next_free_pixel_min = clampi(round(vp_x), 0, screen_width-1)
                    next_free_pixel_max = screen_width-1
            
            segment.next_free_pixel_min = next_free_pixel_min
            segment.next_free_pixel_max = next_free_pixel_max


        mat = camera.get_world_to_screen_matrix()
        world_to_screen_mat = (
            mat.a,mat.b,mat.c,mat.d,
            mat.e,mat.f,mat.g,mat.h,
            mat.i,mat.j,mat.k,mat.l,
            mat.m,mat.n,mat.o,mat.p
        )
        segment_ray_counts = tuple([s.ray_count for s in segments])
        segment_next_free_pixel_mins = tuple([s.next_free_pixel_min for s in segments])
        segment_next_free_pixel_maxs = tuple([s.next_free_pixel_max for s in segments])
        segment_cam_local_plane_ray_mins = tuple([(s.cam_local_plane_ray_min[0], s.cam_local_plane_ray_min[1]) for s in segments])
        segment_cam_local_plane_ray_maxs = tuple([(s.cam_local_plane_ray_max[0], s.cam_local_plane_ray_max[1]) for s in segments])

        
        raycast_segments(
            segment_ray_counts, segment_next_free_pixel_mins, segment_next_free_pixel_maxs, 
            segment_cam_local_plane_ray_mins, segment_cam_local_plane_ray_maxs,
            camera.pos, camera.far_clip, world_to_screen_mat,
            spans, colors, columns, top_down_pix_arr, left_right_pix_arr,
            WORLD_MAX_Y, full_seen_pixel_cache, aligned_bytes_per_column, skybox_col_int
        )

        

        vertices = [None for _ in range(12)]
        uvs = [None for _ in range(12)]
        for tri in range(0,12,3):
            # for some reason this is just reversed
            vertices[tri+0] = adjust_screen_pixel_for_mesh(screen_vp, camera.dims)
            vertices[tri+1] = adjust_screen_pixel_for_mesh(segments[tri // 3].max_screen, camera.dims)
            vertices[tri+2] = adjust_screen_pixel_for_mesh(segments[tri // 3].min_screen, camera.dims)
            uvs[tri + 0] = (0, 0, 1, tri // 3)
            uvs[tri + 1] = (1, 0, 0, tri // 3)
            uvs[tri + 2] = (0, 1, 0, tri // 3)
                

        
        scales = [
            segments[0].ray_count / top_down_draw_surf.get_width(),
            segments[1].ray_count / top_down_draw_surf.get_width(),
            segments[2].ray_count / left_right_draw_surf.get_width(),
            segments[3].ray_count / left_right_draw_surf.get_width(),
        ]
        
        column = segments[0].ray_count+segments[1].ray_count #+top_down_draw_surf.get_height()
        
        # copy second column of the down segment (backwards quadrant)
        # to it's first column
        # this fixes a bug in the shader, where it seems to wrap around :/
        #if segments[1].ray_count > 0 and segments[0].ray_count > 0:
        #    for y in range(top_down_draw_surf.get_height()):
        #        #left_col_idx = max(0, segments[0].ray_count-1)
        #        prev_pix = top_down_draw_surf.get_at(
        #            (segments[0].ray_count-1, y), 
        #        )
        #        if prev_pix == skybox_col:
        #            top_down_draw_surf.set_at(
        #                (segments[0].ray_count-1, y), 
        #                top_down_draw_surf.get_at((segments[0].ray_count, y))
        #                #pygame.Color(255,0,0,255)
        #            )

        offsets = [0.0, scales[0], 0.0, scales[2]]
        
        #text_surf = my_font.render(
        #    f'fwd y: {round(camera.forward.y,3)} rays: {[seg.ray_count for seg in segments]}',
        #    False, (255,255,255)
        #)
        if 'rayScales' in seg01_vf_shader:
            seg01_vf_shader['rayScales'] = scales
            seg23_vf_shader['rayScales'] = scales
        if 'rayOffsets' in seg01_vf_shader:
            seg01_vf_shader['rayOffsets'] = offsets
            seg23_vf_shader['rayOffsets'] = offsets
        try:
            full_screen_shader['lookdown'] = camera.forward.y > 0
        except:
            pass
        try:
            full_screen_shader['rayScales'] = scales
        except:
            pass
        try:
            full_screen_shader['rayOffsets'] = offsets
        except:
            pass
        try:
            full_screen_shader['rayBuffer1'].value = 0
        except:
            pass
        try:
            full_screen_shader['rayBuffer2'].value = 1
        except:
            pass
        full_screen_shader['res'] = (OUTPUT_WIDTH, OUTPUT_HEIGHT)


        screen_vp2 = adjust_screen_pixel_for_mesh(screen_vp, camera.dims)
        full_screen_shader['vp'] = (.5,.5)

        if RENDER_MODE == 0:



            optimized_bytes = 0
            mapping_table = [
                (top_down_pix_arr, seg01_target_texture, 0),
                (top_down_pix_arr, seg01_target_texture, segments[0].ray_count),
                (left_right_pix_arr, seg23_target_texture,0),
                (left_right_pix_arr, seg23_target_texture,segments[2].ray_count)
            ]

            for i in range(4):
                (src_pix_arr, dst_tex, x_offset) = mapping_table[i]
                segment = segments[i]
                src_height = len(src_pix_arr[0])
                seg_height = ((segment.next_free_pixel_max+1) - segment.next_free_pixel_min)
                seg_width = segments[i].ray_count
                y_offset = src_height-1 - segment.next_free_pixel_max
                

                tmp_upload_arr = upload_arr[0:seg_height*seg_width*3] #np.empty((seg_height*seg_width*3), dtype=np.uint8)
                transpose_and_create_bytes(src_pix_arr, tmp_upload_arr, [x_offset, y_offset, seg_width, seg_height])

                dst_tex.update_from_bytes(tmp_upload_arr, [x_offset, y_offset, seg_width, seg_height])

                seg_bytes = seg_height*seg_width*3
                optimized_bytes += seg_bytes



            #seg01_target_texture.update(top_down_draw_surf)

            #total_raybuffer_bytes = (top_down_pix_arr.size * 3 + left_right_pix_arr.size * 3)
            #print(f"Optimized bytes {optimized_bytes}, {optimized_bytes / total_raybuffer_bytes}%")
            
            #seg23_target_texture.update(left_right_draw_surf)
            
            for shdr, base_vert_idx, texture in [
                (seg01_vf_shader, 0, seg01_target_texture),
                (seg01_vf_shader, 3, seg01_target_texture),
                (seg23_vf_shader, 6, seg23_target_texture),
                (seg23_vf_shader, 9, seg23_target_texture)
                ]:
                seg_verts = [
                    vertices[0+base_vert_idx], 
                    vertices[1+base_vert_idx], 
                    vertices[2+base_vert_idx]
                ]
                seg_uvs = [
                    uvs[0+base_vert_idx],
                    uvs[1+base_vert_idx],
                    uvs[2+base_vert_idx]
                ]

                data = np.hstack([np.array(seg_verts, dtype=np.float32), np.array(seg_uvs, dtype=np.float32)])

                #self.vertex_count = 3 if tex_coords else 6

                vbo = gl_ctx.buffer(data)
                vao = gl_ctx.vertex_array(shdr, [
                    (vbo, '3f 4f', 'vertexPos', 'vertexTexCoord'),
                ])


                texture.use()
                vao.render()

        elif RENDER_MODE == 1:
            verts = [
                (-1.0,1.0,.5),
                (1.0,1.0,.5),
                (-1.0,-1.0,.5),

                (-1.0,-1.0,.5),
                (1.0,1.0,.5),
                (1.0,-1.0,.5)
            ]
            data = np.array(verts, dtype=np.float32)
            vbo = gl_ctx.buffer(data)
            vao = gl_ctx.vertex_array(full_screen_shader, [
                (vbo, '3f', 'vertexPos')
            ])

            seg01_target_texture.use(0)
            seg23_target_texture.use(1)
            vao.render()
        else:
            pygame.surfarray.blit_array(top_down_draw_surf, top_down_pix_arr)
            pygame.surfarray.blit_array(left_right_draw_surf, left_right_pix_arr)
            vbo = gl_ctx.buffer(np.array([[-1,-1, 0, 0], # bottom left
                          [-1,1,0,1], # top left
                          [1,1,1,1], # top right
                          [-1,-1, 0, 0], # bottom left
                          [1,1,1,1], # top right
                          [1,-1,1,0], # bottom right
                          ], dtype=np.float32)
            )
            vao = gl_ctx.vertex_array(ray_buffer_shader, [
                (vbo, '2f 2f', 'vertexPos', 'vertexTexCoord'),
            ])
            if RENDER_MODE == 2:
                #pygame.surfarray.blit_array(top_down_draw_surf, top_down_pix_arr)
                seg01_target_texture.update(top_down_draw_surf, vflip=True)
                seg01_target_texture.use()
            else:
                #pygame.surfarray.blit_array(left_right_draw_surf, left_right_pix_arr)
                seg23_target_texture.update(left_right_draw_surf, vflip=True)
                seg23_target_texture.use()
            vao.render()
        
        #display_surf.blit(text_surf, (0,0))
        pygame.display.flip()
        print(f"fps: {fps}")
        clock.tick(400)  # limits FPS to 60

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