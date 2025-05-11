
import pygame
import numba
from vectypes import int4

def color_to_int(color: pygame.Color) -> int:
    return (color.r | (color.g<<8) | (color.b<<16)) # | (color.a<<24))

@numba.njit
def color_tuple_to_int(color: int4) -> int:
    return (color[0] | (color[1]<<8) | (color[2]<<16)) # | (color.a<<24))


@numba.njit
def clampf(a:float,mi:float,ma:float):
    return min(max(a,mi), ma)

@numba.njit
def clampi(a:int,mi:int,ma:int):
    return min(max(a,mi), ma)

skybox_col = pygame.Color(135,206,235,0xFF)
skybox_col_int = color_to_int(skybox_col)
grey = pygame.Color(128,128,128,0xFF)
grey_int = color_to_int(grey)