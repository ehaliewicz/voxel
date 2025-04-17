
import pygame
import numba

def color_to_int(color: pygame.Color) -> int:
    return (color.r | (color.g<<8) | (color.b<<16)) # | (color.a<<24))


@numba.njit
def clampf(a:float,mi:float,ma:float):
    return min(max(a,mi), ma)

@numba.njit
def clampi(a:int,mi:int,ma:int):
    return min(max(a,mi), ma)

skybox_col = pygame.Color(135,206,235,0xFF)
skybox_col_int = color_to_int(skybox_col)