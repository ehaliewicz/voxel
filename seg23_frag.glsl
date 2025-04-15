#version 330 core

in vec3 vertex;
in vec4 uv;

out vec4 color;

uniform sampler2D rayBuffer;
uniform vec4 rayScales, rayOffsets;
uniform vec4 rayBufferWidths, rayBufferHeights;
uniform bool bilinear;

void main() {
    float y = 1-(vertex.x + 1.0) * 0.5; // Convert from [-1,1] to [0,1]
    float x = clamp(uv.x, 0.0, 1.0) / (uv.x + uv.y);

    int tri_idx = int(uv.w);
    float ray_buffer_height = rayBufferHeights[tri_idx];
    float ray_buffer_width = rayBufferWidths[tri_idx];
    float ray_buffer_scale = rayScales[tri_idx];
    float ray_buffer_offset = rayOffsets[tri_idx];

    x = ray_buffer_offset + x * ray_buffer_scale;
    vec4 sample = texture2D(rayBuffer, vec2(x,y));
    color = sample;

}