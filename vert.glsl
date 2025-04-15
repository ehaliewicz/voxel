#version 330 core

layout (location = 0) in vec3 vertexPos;
layout (location = 1) in vec4 vertexTexCoord;

out vec3 vertex;
out vec4 uv;

void main()
{
    vertex = vertexPos;
    uv = vertexTexCoord;
    gl_Position = vec4(vertexPos, 1.0);
}