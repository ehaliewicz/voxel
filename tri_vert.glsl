#version 330 core

layout (location = 0) in vec3 vertexPos;
layout (location = 1) in vec2 vertexTexCoord;

uniform mat4 viewProj;
out vec2 uv;
void main() {
    uv = vertexTexCoord;
    gl_Position = viewProj * vec4(vertexPos, 1.0);
}