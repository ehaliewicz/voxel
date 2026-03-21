
type float2 = tuple[float, float]
type float3 = tuple[float, float, float]
type float4 = tuple[float, float, float, float]
type float5 = tuple[float, float, float, float]
type int2 = tuple[int, int]
type int4 = tuple[int, int, int, int]


type float44 = tuple[float, float, float, float,
                float, float, float, float,
                float, float, float, float,
                float, float, float, float]
type RayTuple = tuple[float2, int2, float2, float2, float2, float2, float]
#type RayTuple = tuple[float2, float2, float2, int2]
#(position, step, start, dir, delta_dist, side_dist)
#position, 
#step, 
#start, 
#dir, 
#t_delta, 
#t_max, 
#0.0,
