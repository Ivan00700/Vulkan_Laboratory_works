#version 450

layout(location = 0) in vec3 normal;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 light = normalize(vec3(0.45, 0.85, 0.7));
    float brightness = 0.28 + 0.72 * max(dot(normalize(normal), light), 0.0);
    outColor = vec4(vec3(0.97, 0.48, 0.18) * brightness, 1.0);
}
