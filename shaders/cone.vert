#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 0) out vec3 outNormal;

layout(push_constant) uniform Transform {
    float yaw;
    float pitch;
    float aspect;
    float distance;
} transform;

void main() {
    float cy = cos(transform.yaw), sy = sin(transform.yaw);
    float cx = cos(transform.pitch), sx = sin(transform.pitch);
    mat3 aroundY = mat3(cy, 0, -sy,  0, 1, 0,  sy, 0, cy);
    mat3 aroundX = mat3(1, 0, 0,  0, cx, sx,  0, -sx, cx);
    mat3 rotation = aroundX * aroundY;

    vec3 point = rotation * position;
    point.z -= transform.distance;
    outNormal = normalize(rotation * normal);

    // Perspective with 50-degree vertical field of view and Vulkan's
    // depth range [0, 1]. Flip Y for Vulkan's screen coordinates.
    const float focalLength = 2.1445069;
    const float nearPlane = 0.1, farPlane = 100.0;
    gl_Position = vec4(
        point.x * focalLength / transform.aspect,
        -point.y * focalLength,
        farPlane / (nearPlane - farPlane) * point.z +
            farPlane * nearPlane / (nearPlane - farPlane),
        -point.z
    );
}
