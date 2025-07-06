--- Building instructions ---

Needs SDL2.dll and SDL2 dev tools installed at ./SDL2

Compile with gcc.exe -LSDL2_LIB_DIRECTORY -ISDL_INCLUDE_DIRECTORY raycast.c -Wpsabi -mavx2 -O3 -lmingw32 -lSDL2main -lSDL2 -o raycast.exe


--- Running instructions --- 
Put ace of spades .vxl maps in a /maps directory and launch without any arguments.


- Controls

F - toggles fog
L - toggles lighting
O - toggles ambient otcclusion
Arrow keys - move forward and back, turn left and right
Z/X - go down / go up
A/S - look up / look down
Q/E - roll left / roll right

Middle mouse - deletes random blocks in a sphere around where the cursor or center of the screen is pointing
Right mouse - turns blocks bright bink

T - toggle transparency
R - switch render resolution (1x/ 1/2x / 1/4x)
N - next map


- Screenshots

![Foggy morning on Rocket Island](rocket_island.png?raw=true "Foggy morning on Rocket Island")
![Voxel City](vox_city.png?raw=true "Voxel City")


- Video

- https://www.youtube.com/watch?v=xjlQ-fhrC9A
- https://www.youtube.com/watch?v=V8qHlTwUiV0
- https://www.youtube.com/watch?v=YC6HvZVuShw 
