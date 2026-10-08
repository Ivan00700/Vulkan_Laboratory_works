#version 450

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Transform {
    vec4 positionProjection;
    vec4 rotation;
    vec4 scale;
    vec4 color;
    vec4 view;
} transform;

void main() {
    vec3 light = normalize(vec3(0.45, 0.85, 0.7));
    // яркость зависит от того, насколько поверхность обращена к свету
    float brightness = 0.28 + 0.72 * max(dot(normalize(normal), light), 0.0);
    // итоговый цвет сочетает выбор пользователя, цвет вершины и освещение
    outColor = vec4(transform.color.rgb * vertexColor * brightness, 1.0);
}
