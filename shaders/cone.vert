#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 vertexColor;
layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outVertexColor;

layout(push_constant) uniform Transform {
    vec4 positionProjection;
    vec4 rotation;
    vec4 scale;
    vec4 color;
    vec4 view;
} transform;

void main() {
    float cx = cos(transform.rotation.x), sx = sin(transform.rotation.x);
    float cy = cos(transform.rotation.y), sy = sin(transform.rotation.y);
    float cz = cos(transform.rotation.z), sz = sin(transform.rotation.z);
    mat3 aroundY = mat3(cy, 0, -sy,  0, 1, 0,  sy, 0, cy);
    mat3 aroundX = mat3(1, 0, 0,  0, cx, sx,  0, -sx, cx);
    mat3 aroundZ = mat3(cz, sz, 0,  -sz, cz, 0,  0, 0, 1);
    mat3 objectRotation = aroundZ * aroundY * aroundX;

    // матрица модели объединяет масштаб, поворот и перенос в этом порядке
    mat4 scaleMatrix = mat4(1.0);
    scaleMatrix[0][0] = transform.scale.x;
    scaleMatrix[1][1] = transform.scale.y;
    scaleMatrix[2][2] = transform.scale.z;
    mat4 rotationMatrix = mat4(objectRotation);
    mat4 translationMatrix = mat4(1.0);
    translationMatrix[3].xyz = transform.positionProjection.xyz;
    mat4 model = translationMatrix * rotationMatrix * scaleMatrix;

    vec3 point = (model * vec4(position, 1.0)).xyz;
    point.z -= transform.view.x;
    // нормаль преобразуется обратной транспонированной матрицей модели
    mat3 normalMatrix = transpose(inverse(mat3(model)));
    outNormal = normalize(normalMatrix * normal);
    outVertexColor = vertexColor;

    // обе проекции используют одно направление взгляда и пропорции окна
    const float focalLength = 2.1445069;
    const float nearPlane = 0.1, farPlane = 100.0;
    if (transform.positionProjection.w < 0.5) {
        // перспектива уменьшает видимый размер удалённых точек
        gl_Position = vec4(
            point.x * focalLength / transform.view.y,
            -point.y * focalLength,
            farPlane / (nearPlane - farPlane) * point.z +
                farPlane * nearPlane / (nearPlane - farPlane),
            -point.z
        );
    } else {
        // ортографическая проекция не меняет размер из-за глубины точки
        float zoom = focalLength / transform.view.x;
        gl_Position = vec4(
            point.x * zoom / transform.view.y,
            -point.y * zoom,
            (-point.z - nearPlane) / (farPlane - nearPlane),
            1.0
        );
    }
}
