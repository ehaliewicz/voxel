#version 330 core

in vec3 vertex;

out vec4 color;

uniform sampler2D rayBuffer1, rayBuffer2;
uniform vec4 rayScales, rayOffsets;
uniform vec4 rayBufferWidths, rayBufferHeights;
uniform bool bilinear;
uniform vec2 res;
uniform vec2 vp;
uniform float lookdown;

void main() {
    float ratio = res.y / res.x; // 12
    float border = (1. - ratio) / 2.;
    // (width - height) / width

    vec2 vp = 1. - vp;
    vp.y = (vp.y - border) / ratio;
    
    float xs = gl_FragCoord.x / res.x;
    float ys = gl_FragCoord.y / res.y;
    float xn = xs - vp.x;
    float yn = ys - vp.y;

    float high = float(yn < 0.);
    float left = float(xn < 0.);

    float wide = float(abs(xn) * res.x > abs(yn) * res.y);

    
	float seg_up = (1. - high) * (1. - wide);
	float seg_dn =       high  * (1. - wide);
	float seg_rt = (1. - left) *       wide ;
	float seg_lt =       left  *       wide ;

    float nseg_up = lookdown * seg_up + (1. - lookdown) * seg_dn;
	float nseg_dn = lookdown * seg_dn + (1. - lookdown) * seg_up;
	float nseg_rt = lookdown * seg_rt + (1. - lookdown) * seg_lt;
	float nseg_lt = lookdown * seg_lt + (1. - lookdown) * seg_rt;

	float ang_vert =
		xn          * abs(1. - high - vp.y) / yn +
		high        * (1. - vp.x) +
		(1. - high) * vp.x;

	float ang_horz =
		yn          * abs(1. - left - vp.x) / xn +
		left        * (1. - vp.y) +
		(1. - left) * vp.y;

	ang_horz = ang_horz * ratio + border;

	vec2 texpos;
    texpos.y = seg_up * (ang_vert) +
       seg_dn * (1 -   ang_vert) +
       seg_rt * (1. - ang_horz) +
       seg_lt * (     ang_horz);

    //texpos.y *= 0.25;

    ys = ys * ratio + border;

    texpos.x = nseg_up * (     ys) +
               nseg_dn * (1. - ys) +
               nseg_rt * (     xs) +
               nseg_lt * (1. - xs);
    vec4 sample;
    int tri_idx;
    if (seg_up > 0.) {
        tri_idx = 0;
        float ray_buffer_height = rayBufferHeights[0];
        float ray_buffer_width = rayBufferWidths[0];
        float ray_buffer_scale = rayScales[0];
        float ray_buffer_offset = rayOffsets[0];
        float x =  ys;
        float y = texpos.y; //ys * 0.5;
        color = vec4(0.0, x, 0.0, 1.0); 

    } else if (seg_dn > 0.) {
        tri_idx = 1;
        float ray_buffer_height = rayBufferHeights[1];
        float ray_buffer_width = rayBufferWidths[1];
        float ray_buffer_scale = rayScales[1];
        float ray_buffer_offset = rayOffsets[1];
        float x =  ys;
        float y =  texpos.y; //(1.-ys) * 0.5;
        color = vec4(0.0, x, 0.0, 1.0); 

    } else if (seg_lt > 0.) {
        tri_idx = 2;
        float ray_buffer_height = rayBufferHeights[tri_idx];
        float ray_buffer_width = rayBufferWidths[tri_idx];
        float ray_buffer_scale = rayScales[tri_idx];
        float ray_buffer_offset = rayOffsets[tri_idx];
        color = vec4(0.0, 0.0, 0.0, 1.0); 
    } else if (seg_rt > 0.) {
        tri_idx = 3;
        float ray_buffer_height = rayBufferHeights[tri_idx];
        float ray_buffer_width = rayBufferWidths[tri_idx];
        float ray_buffer_scale = rayScales[tri_idx];
        float ray_buffer_offset = rayOffsets[tri_idx];
        color = vec4(0.0, 0.0, 0.0, 1.0); 
    }
    //color = sample;
}