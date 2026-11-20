#version 450

layout(location = 0) in vec3 position;

layout(set = 0, binding = 0, std140) uniform Scene {
    mat4 viewProjection;
    vec4 cameraPositionFov;
    vec4 lightDirectionIntensity;
    vec4 lightColor;
    vec4 ambient;
    uvec4 lightCounts;
    mat4 lightViewProjection;
} scene;

layout(set = 1, binding = 0, std140) uniform Model {
    mat4 model;
    vec4 specularShininess;
} transform;

void main() {
    gl_Position = scene.lightViewProjection * transform.model * vec4(position, 1.0);
}
