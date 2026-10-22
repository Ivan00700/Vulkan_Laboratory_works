#version 450

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in vec3 worldPosition;

layout(location = 0) out vec4 outColor;

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
    vec4 albedo;
    vec4 specularShininess;
} transform;

struct PointLight {
    vec4 positionIntensity;
    vec4 color;
};

struct SpotLight {
    vec4 positionIntensity;
    vec4 color;
    vec4 directionInner;
    vec4 outerAngle;
};

layout(set = 0, binding = 1, std430) readonly buffer PointLights {
    PointLight pointLights[];
};

layout(set = 0, binding = 2, std430) readonly buffer SpotLights {
    SpotLight spotLights[];
};

vec3 safeNormalize(vec3 value) {
    return value / max(length(value), 0.00001);
}

vec3 blinnPhong(vec3 N, vec3 L, vec3 V, vec3 albedo) {
    vec3 H = safeNormalize(L + V);
    float diffuse = max(dot(N, L), 0.0);
    float specular = 0.0;
    if (diffuse > 0.0)
        specular = pow(max(dot(N, H), 0.0), transform.specularShininess.w);

    return albedo * diffuse + transform.specularShininess.rgb * specular;
}

void main() {
    vec3 N = safeNormalize(normal);
    vec3 V = safeNormalize(scene.cameraPositionFov.xyz - worldPosition);
    vec3 albedo = transform.albedo.rgb * vertexColor;
    vec3 result = scene.ambient.rgb * scene.ambient.w * albedo;

    // направление к источнику противоположно направлению его лучей
    vec3 L = safeNormalize(-scene.lightDirectionIntensity.xyz);
    result += scene.lightColor.rgb * scene.lightDirectionIntensity.w * blinnPhong(N, L, V, albedo);

    for (uint i = 0; i < scene.lightCounts.x; ++i) {
        PointLight light = pointLights[i];
        vec3 offset = light.positionIntensity.xyz - worldPosition;
        float distanceSquared = max(dot(offset, offset), 0.0001);
        L = safeNormalize(offset);

        // яркость точечного источника убывает как 1 / r^2
        result += light.color.rgb * light.positionIntensity.w / distanceSquared *
                  blinnPhong(N, L, V, albedo);
    }

    for (uint i = 0; i < scene.lightCounts.y; ++i) {
        SpotLight light = spotLights[i];
        vec3 offset = light.positionIntensity.xyz - worldPosition;
        float distanceSquared = max(dot(offset, offset), 0.0001);
        L = safeNormalize(offset);

        float theta = dot(-L, safeNormalize(light.directionInner.xyz));
        float inner = cos(radians(light.directionInner.w));
        float outer = cos(radians(light.outerAngle.x));

        // между двумя углами свет плавно затухает; равные углы дают резкий край
        float cone = inner > outer ? smoothstep(outer, inner, theta) : step(inner, theta);
        result += light.color.rgb * light.positionIntensity.w * cone / distanceSquared *
                  blinnPhong(N, L, V, albedo);
    }

    outColor = vec4(result, 1.0);
}
