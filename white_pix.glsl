#version 330 core

in vec3 vertex;
in vec2 uv;

out vec4 color;

uniform sampler2D tex;
uniform int useTex;
void main() {
    vec4 col = texture(tex, uv);
    color = vec4(col.y, col.z, col.w, col.x);
}