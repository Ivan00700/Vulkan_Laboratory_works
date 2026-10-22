#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 outColor;

layout(set = 0, binding = 0, std140) uniform Scene {
    mat4 viewProjection;
    vec4 cameraPositionFov;
    vec4 lightDirectionIntensity;
    vec4 lightColor;
    vec4 ambient;
    uvec4 lightCounts;
} scene;

void main() {
    // вершины маркеров уже заданы в мировых координатах
    gl_Position = scene.viewProjection * vec4(position, 1.0);
    outColor = color;
}
