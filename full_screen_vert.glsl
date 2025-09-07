#version 330 core

layout (location = 0) in vec3 vertexPos;

out vec3 vertex;
out vec4 uv;

void main()
{
    vertex = vertexPos;
    gl_Position = vec4(vertexPos, 1.0);
}