#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outTexCoord;
layout(location = 2) out vec3 outWorldPosition;

layout(set = 0, binding = 0, std140) uniform Scene {
    mat4 viewProjection;
    vec4 cameraPositionFov;
    vec4 lightDirectionIntensity;
    vec4 lightColor;
    vec4 ambient;
    uvec4 lightCounts;
} scene;

layout(set = 1, binding = 0, std140) uniform Model {
    mat4 model;
    vec4 specularShininess;
} transform;

void main() {
    vec4 worldPosition = transform.model * vec4(position, 1.0);

    outWorldPosition = worldPosition.xyz;
    // обратная транспонированная матрица учитывает масштаб по разным осям
    outNormal = normalize(transpose(inverse(mat3(transform.model))) * normal);
    outTexCoord = texCoord;

    gl_Position = scene.viewProjection * worldPosition;
}
