import math 
import pyglet

class Camera:
    def __init__(self, render_width, render_height, near_clip_plane, far_clip_plane):
        self.pos = pyglet.math.Vec3(256.0, 128.0, 256.0)
        self.forward = pyglet.math.Vec3(0.0, -1.0, 0.0) #pyglet.math.Vec3(0, 0, 1.0)
        self.right = pyglet.math.Vec3(1.0, 0.0, 0.0)
        #self.up = pyglet.math.Vec3(0, 0, 1) 
        self.up = self.forward.cross(self.right)
        self.dims = pyglet.math.Vec2(render_width, render_height)
        self.near_clip = near_clip_plane
        self.far_clip = far_clip_plane
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
    
