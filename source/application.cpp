#include "application.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace application {

namespace {

// вершина хранит координаты, нормаль для освещения и процедурный цвет
struct Vertex {
    float x, y, z, nx, ny, nz, r, g, b;
};

struct Vec4 {
    float x, y, z, w;
};

struct Matrix4 {
    float values[16];
};

// порядок полей совпадает с uniform-блоком в шейдерах
struct Transform {
    Matrix4 model;
    Vec4 albedo;
    Vec4 specular_shininess;
};

struct SceneData {
    Matrix4 view_projection;
    Vec4 camera_position_fov;

    Vec4 light_direction_intensity;
    Vec4 light_color;
    Vec4 ambient;
    uint32_t light_counts[4];
};

struct PointLight {
    Vec4 position_intensity;
    Vec4 color;
};

struct SpotLight {
    Vec4 position_intensity;
    Vec4 color;
    Vec4 direction_inner;
    Vec4 outer_angle;
};

struct MappedBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* data = nullptr;
    VkDeviceSize size = 0;
};

struct MarkerVertex {
    float x, y, z;
    float r, g, b;
};

static_assert(sizeof(Transform) == 96);
static_assert(sizeof(SceneData) == 144);
static_assert(sizeof(PointLight) == 32);
static_assert(sizeof(SpotLight) == 64);

std::vector<Vertex> vertices;
VkBuffer buffer = VK_NULL_HANDLE;
VmaAllocation allocation = VK_NULL_HANDLE;

VkPipelineLayout layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

VkBuffer uniform_buffer = VK_NULL_HANDLE;
VmaAllocation uniform_allocation = VK_NULL_HANDLE;
void* uniform_data = nullptr;

VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
VkDescriptorSetLayout model_descriptor_layout = VK_NULL_HANDLE;
VkDescriptorSet model_descriptor_set = VK_NULL_HANDLE;

MappedBuffer model_buffer;
MappedBuffer point_buffer;
MappedBuffer spot_buffer;
std::vector<PointLight> point_lights;
std::vector<SpotLight> spot_lights;

VkPipelineLayout marker_layout = VK_NULL_HANDLE;
VkPipeline marker_pipeline = VK_NULL_HANDLE;
VkPipeline line_pipeline = VK_NULL_HANDLE;
MappedBuffer marker_buffer;
MappedBuffer line_buffer;
std::vector<MarkerVertex> marker_vertices;
std::vector<MarkerVertex> line_vertices;
bool show_light_markers = true;
bool show_spot_cones = true;

float object_position[3] = {0.0f, 0.0f, 0.0f};
float object_rotation_deg[3] = {-17.0f, 29.0f, 0.0f};
float object_scale[3] = {1.0f, 1.0f, 1.0f};
float selected_color[3] = {1.0f, 0.65f, 0.35f};

float camera_position[3] = {0.0f, 0.0f, 4.5f};
float camera_rotation_deg[3] = {0.0f, 0.0f, 0.0f};
float camera_fov = 50.0f;
float near_plane = 0.1f, far_plane = 100.0f;

float light_direction[3] = {-0.45f, -0.85f, -0.7f};
float light_color[3] = {1.0f, 1.0f, 1.0f};
float light_intensity = 1.0f;

float specular_color[3] = {1.0f, 1.0f, 1.0f};
float shininess = 32.0f;

float ambient_color[3] = {1.0f, 1.0f, 1.0f};
float ambient_intensity = 0.18f;

void check(VkResult result, const char* message) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(message);
}

Matrix4 rotationMatrix(const float angles[3]) {
    const float radians = std::numbers::pi_v<float> / 180.0f;
    const float cx = std::cos(angles[0] * radians), sx = std::sin(angles[0] * radians);
    const float cy = std::cos(angles[1] * radians), sy = std::sin(angles[1] * radians);
    const float cz = std::cos(angles[2] * radians), sz = std::sin(angles[2] * radians);

    // столбцы матрицы поворота Rz * Ry * Rx
    return {{
        cz * cy, sz * cy, -sy, 0,
        cz * sy * sx - sz * cx, sz * sy * sx + cz * cx, cy * sx, 0,
        cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx, 0,
        0, 0, 0, 1,
    }};
}

Matrix4 multiplyMatrices(const Matrix4& a, const Matrix4& b) {
    Matrix4 result = {};

    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            for (int k = 0; k < 4; ++k)
                result.values[column * 4 + row] +=
                    a.values[k * 4 + row] * b.values[column * 4 + k];

    return result;
}

Matrix4 makeViewProjection(float aspect) {
    const Matrix4 rotation = rotationMatrix(camera_rotation_deg);
    Matrix4 view = {};
    view.values[15] = 1.0f;

    // обратный поворот камеры и перенос на минус её позицию
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            view.values[column * 4 + row] = rotation.values[row * 4 + column];
            view.values[12 + row] -=
                view.values[column * 4 + row] * camera_position[column];
        }
    }

    const float radians = std::numbers::pi_v<float> / 180.0f;
    const float focal_length = 1.0f / std::tan(camera_fov * radians * 0.5f);
    Matrix4 projection = {};
    projection.values[0] = focal_length / aspect;
    projection.values[5] = -focal_length;
    projection.values[10] = far_plane / (near_plane - far_plane);
    projection.values[11] = -1.0f;
    projection.values[14] = far_plane * near_plane / (near_plane - far_plane);

    return multiplyMatrices(projection, view);
}

void createMappedBuffer(MappedBuffer& target, VkDeviceSize size, VkBufferUsageFlags usage) {
    const auto& c = graphics::internal::context;

    const VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    };
    check(vmaCreateBuffer(c.allocator, &info, &alloc_info,
                         &target.buffer, &target.allocation, nullptr),
          "Cannot allocate mapped buffer");
    check(vmaMapMemory(c.allocator, target.allocation, &target.data),
          "Cannot map buffer");
    target.size = size;
}

void destroyMappedBuffer(MappedBuffer& target) {
    const auto& c = graphics::internal::context;

    if (target.buffer != VK_NULL_HANDLE) {
        vmaUnmapMemory(c.allocator, target.allocation);
        vmaDestroyBuffer(c.allocator, target.buffer, target.allocation);
        target = {};
    }
}

void writeBufferDescriptor(VkDescriptorSet set, uint32_t binding,
                           VkDescriptorType type, VkBuffer target_buffer, VkDeviceSize size) {
    const VkDescriptorBufferInfo buffer_info = {target_buffer, 0, size};
    const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = type,
        .pBufferInfo = &buffer_info,
    };
    vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

template<typename Light>
void uploadLights(MappedBuffer& target, const std::vector<Light>& lights, uint32_t binding) {
    const auto& c = graphics::internal::context;
    const VkDeviceSize bytes = lights.size() * sizeof(Light);

    // пустому массиву тоже нужен действительный дескриптор буфера
    const VkDeviceSize required = std::max(VkDeviceSize(sizeof(Light)), bytes);
    if (required > target.size) {
        destroyMappedBuffer(target);
        createMappedBuffer(target, required, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        writeBufferDescriptor(descriptor_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                              target.buffer, target.size);
    }

    if (!lights.empty()) {
        std::memcpy(target.data, lights.data(), size_t(bytes));
        check(vmaFlushAllocation(c.allocator, target.allocation, 0, bytes),
              "Cannot flush light buffer");
    }
}

void addMarkerCube(const Vec4& position, const Vec4& color) {
    constexpr float half_size = 0.08f;
    constexpr float corners[8][3] = {
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
        {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1},
    };
    constexpr int indices[] = {
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7,
        0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2,
    };

    for (int index : indices) {
        const auto& corner = corners[index];
        marker_vertices.push_back({
            position.x + corner[0] * half_size,
            position.y + corner[1] * half_size,
            position.z + corner[2] * half_size,
            color.x, color.y, color.z,
        });
    }
}

void addSpotCone(const SpotLight& light) {
    Vec4 direction = light.direction_inner;
    const float length = std::sqrt(direction.x * direction.x + direction.y * direction.y +
                                   direction.z * direction.z);
    if (length < 0.00001f)
        return;
    direction.x /= length;
    direction.y /= length;
    direction.z /= length;

    // выбираем поперечную ось, которая не параллельна направлению прожектора
    Vec4 right = std::abs(direction.y) < 0.99f ?
        Vec4{-direction.z, 0, direction.x, 0} :
        Vec4{0, direction.z, -direction.y, 0};
    const float right_length = std::sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
    right.x /= right_length;
    right.y /= right_length;
    right.z /= right_length;
    const Vec4 up = {
        direction.y * right.z - direction.z * right.y,
        direction.z * right.x - direction.x * right.z,
        direction.x * right.y - direction.y * right.x, 0,
    };

    constexpr float cone_length = 1.5f;
    constexpr int segments = 32;
    const float radius = cone_length * std::tan(light.outer_angle.x * std::numbers::pi_v<float> / 180.0f);
    const Vec4& position = light.position_intensity;
    const Vec4 center = {
        position.x + direction.x * cone_length,
        position.y + direction.y * cone_length,
        position.z + direction.z * cone_length, 0,
    };
    const auto circlePoint = [&](int i) {
        const float angle = 2.0f * std::numbers::pi_v<float> * float(i) / segments;
        const float x = radius * std::cos(angle), y = radius * std::sin(angle);
        return Vec4{center.x + right.x * x + up.x * y,
                    center.y + right.y * x + up.y * y,
                    center.z + right.z * x + up.z * y, 0};
    };
    const auto addLine = [&](const Vec4& a, const Vec4& b) {
        line_vertices.push_back({a.x, a.y, a.z, 1, 1, 1});
        line_vertices.push_back({b.x, b.y, b.z, 1, 1, 1});
    };

    // ось показывает направление, окружность и четыре ребра — внешний угол
    addLine(position, center);
    for (int i = 0; i < segments; ++i) {
        addLine(circlePoint(i), circlePoint(i + 1));
        if (i % (segments / 4) == 0)
            addLine(position, circlePoint(i));
    }
}

void uploadMarkerVertices(MappedBuffer& target, const std::vector<MarkerVertex>& data) {
    if (data.empty())
        return;

    const VkDeviceSize bytes = data.size() * sizeof(MarkerVertex);
    if (bytes > target.size) {
        destroyMappedBuffer(target);
        createMappedBuffer(target, bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    }

    std::memcpy(target.data, data.data(), size_t(bytes));
    check(vmaFlushAllocation(graphics::internal::context.allocator, target.allocation, 0, bytes),
          "Cannot flush marker vertices");
}

void updateLightMarkers() {
    marker_vertices.clear();
    line_vertices.clear();
    if (!show_light_markers)
        return;

    for (const PointLight& light : point_lights)
        addMarkerCube(light.position_intensity, light.color);
    for (const SpotLight& light : spot_lights) {
        addMarkerCube(light.position_intensity, light.color);
        if (show_spot_cones)
            addSpotCone(light);
    }

    uploadMarkerVertices(marker_buffer, marker_vertices);
    uploadMarkerVertices(line_buffer, line_vertices);
}

// каждый из 48 секторов даёт треугольник боковой поверхности и основания
void buildCone() {
    constexpr uint32_t segments = 48;
    constexpr float radius = 1.0f, height = 2.0f;

    vertices.reserve(segments * 6);

    // цвет вершины вычисляется из её локальных координат до преобразований
    const auto addVertex = [&](float x, float y, float z, float nx, float ny, float nz) {
        const float r = 0.4f + 0.6f * (x / radius + 1.0f) * 0.5f;
        const float g = 0.4f + 0.6f * (y / height + 0.5f);
        const float b = 0.4f + 0.6f * (z / radius + 1.0f) * 0.5f;

        vertices.push_back({x, y, z, nx, ny, nz, r, g, b});
    };

    for (uint32_t i = 0; i < segments; ++i) {
        const float a = 2.0f * std::numbers::pi_v<float> * float(i) / segments;
        const float b = 2.0f * std::numbers::pi_v<float> * float(i + 1) / segments;

        const float m = (a + b) * 0.5f;

        const float x0 = radius * std::cos(a), z0 = radius * std::sin(a);
        const float x1 = radius * std::cos(b), z1 = radius * std::sin(b);

        // нормаль боковой грани вычисляется по середине сектора
        const float normal_scale = 1.0f / std::sqrt(1.0f + radius * radius / (height * height));
        const float nx = std::cos(m) * normal_scale;
        const float ny = radius / height * normal_scale;
        const float nz = std::sin(m) * normal_scale;

        addVertex(0, height / 2, 0, nx, ny, nz);
        addVertex(x1, -height / 2, z1, nx, ny, nz);
        addVertex(x0, -height / 2, z0, nx, ny, nz);

        addVertex(0, -height / 2, 0, 0, -1, 0);
        addVertex(x0, -height / 2, z0, 0, -1, 0);
        addVertex(x1, -height / 2, z1, 0, -1, 0);
    }
}

VkShaderModule makeShader(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open compiled shader");

    const std::streamsize bytes = file.tellg();
    if (bytes <= 0 || bytes % sizeof(uint32_t) != 0)
        throw std::runtime_error("Invalid compiled shader size");

    std::vector<uint32_t> code(size_t(bytes) / sizeof(uint32_t));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(code.data()), bytes))
        throw std::runtime_error("Cannot read compiled shader");

    const VkShaderModuleCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size_t(bytes),
        .pCode = code.data(),
    };

    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module),
          "Cannot create shader module");

    return module;
}

void makePipeline() {
    const auto& c = graphics::internal::context;

    VkShaderModule vs = makeShader("shaders/cone.vert.spv");
    VkShaderModule fs = makeShader("shaders/cone.frag.spv");
    const VkPipelineShaderStageCreateInfo stages[] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vs,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fs,
         .pName = "main"},
    };

    const VkVertexInputBindingDescription binding = {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription attributes[] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, x)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, nx)},
        {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, r)},
    };

    const VkPipelineVertexInputStateCreateInfo input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 3,
        .pVertexAttributeDescriptions = attributes,
    };

    const VkPipelineInputAssemblyStateCreateInfo assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    const VkPipelineViewportStateCreateInfo viewport = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    const VkPipelineRasterizationStateCreateInfo raster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1,
    };

    const VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo depth = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    const VkPipelineColorBlendAttachmentState color = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color,
    };

    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamic_states,
    };

    const VkDescriptorSetLayout set_layouts[] = {descriptor_layout, model_descriptor_layout};
    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = set_layouts,
    };
    check(vkCreatePipelineLayout(c.device, &layout_info, nullptr, &layout),
          "Cannot create pipeline layout");

    const VkGraphicsPipelineCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &input,
        .pInputAssemblyState = &assembly,
        .pViewportState = &viewport,
        .pRasterizationState = &raster,
        .pMultisampleState = &ms,
        .pDepthStencilState = &depth,
        .pColorBlendState = &blend,
        .pDynamicState = &dynamic,
        .layout = layout,
        .renderPass = c.render_pass,
        .subpass = 0,
    };

    const VkResult result = vkCreateGraphicsPipelines(c.device, VK_NULL_HANDLE, 1,
                                                    &info, nullptr, &pipeline);

    vkDestroyShaderModule(c.device, fs, nullptr);
    vkDestroyShaderModule(c.device, vs, nullptr);
    check(result, "Cannot create graphics pipeline");

    // маркеры используют отдельные шейдеры без расчёта освещения
    const VkPipelineLayoutCreateInfo marker_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_layout,
    };
    check(vkCreatePipelineLayout(c.device, &marker_layout_info, nullptr, &marker_layout),
          "Cannot create marker pipeline layout");

    VkShaderModule marker_vs = makeShader("shaders/marker.vert.spv");
    VkShaderModule marker_fs = makeShader("shaders/marker.frag.spv");
    const VkPipelineShaderStageCreateInfo marker_stages[] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = marker_vs, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = marker_fs, .pName = "main"},
    };
    const VkVertexInputBindingDescription marker_binding = {
        0, sizeof(MarkerVertex), VK_VERTEX_INPUT_RATE_VERTEX
    };
    const VkVertexInputAttributeDescription marker_attributes[] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MarkerVertex, x)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MarkerVertex, r)},
    };
    const VkPipelineVertexInputStateCreateInfo marker_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &marker_binding,
        .vertexAttributeDescriptionCount = 2,
        .pVertexAttributeDescriptions = marker_attributes,
    };

    VkGraphicsPipelineCreateInfo marker_info = info;
    marker_info.pStages = marker_stages;
    marker_info.pVertexInputState = &marker_input;
    marker_info.layout = marker_layout;
    const VkResult marker_result = vkCreateGraphicsPipelines(c.device, VK_NULL_HANDLE, 1,
                                                            &marker_info, nullptr, &marker_pipeline);

    const VkPipelineInputAssemblyStateCreateInfo line_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    };
    VkPipelineDepthStencilStateCreateInfo line_depth = depth;
    line_depth.depthWriteEnable = VK_FALSE;
    marker_info.pInputAssemblyState = &line_assembly;
    marker_info.pDepthStencilState = &line_depth;
    const VkResult line_result = vkCreateGraphicsPipelines(c.device, VK_NULL_HANDLE, 1,
                                                          &marker_info, nullptr, &line_pipeline);

    vkDestroyShaderModule(c.device, marker_fs, nullptr);
    vkDestroyShaderModule(c.device, marker_vs, nullptr);
    check(marker_result, "Cannot create marker graphics pipeline");
    check(line_result, "Cannot create line graphics pipeline");
}

} // namespace

bool initialize() {
    const auto& c = graphics::internal::context;

    buildCone();

    const VkDeviceSize bytes = vertices.size() * sizeof(Vertex);

    // буфер вершин создаётся один раз, так как геометрия конуса не меняется
    const VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes,
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    };
    check(vmaCreateBuffer(c.allocator, &info, &alloc_info, &buffer, &allocation, nullptr),
          "Cannot allocate cone vertices");

    void* data = nullptr;
    check(vmaMapMemory(c.allocator, allocation, &data), "Cannot map vertex buffer");
    std::memcpy(data, vertices.data(), size_t(bytes));
    vmaFlushAllocation(c.allocator, allocation, 0, bytes);
    vmaUnmapMemory(c.allocator, allocation);

    // параметры сцены обновляются в этом буфере перед отрисовкой
    const VkBufferCreateInfo uniform_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = sizeof(SceneData),
        .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    check(vmaCreateBuffer(c.allocator, &uniform_info, &alloc_info,
                         &uniform_buffer, &uniform_allocation, nullptr),
          "Cannot allocate scene uniform buffer");
    check(vmaMapMemory(c.allocator, uniform_allocation, &uniform_data),
          "Cannot map scene uniform buffer");

    createMappedBuffer(model_buffer, sizeof(Transform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // set 0 содержит общие данные сцены и два массива источников света
    const VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo descriptor_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    check(vkCreateDescriptorSetLayout(c.device, &descriptor_info, nullptr, &descriptor_layout),
          "Cannot create scene descriptor layout");

    // set 1 содержит только матрицу модели и её материал
    const VkDescriptorSetLayoutBinding model_binding = {
        0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr
    };
    const VkDescriptorSetLayoutCreateInfo model_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &model_binding,
    };
    check(vkCreateDescriptorSetLayout(c.device, &model_info, nullptr, &model_descriptor_layout),
          "Cannot create model descriptor layout");

    const VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
    };
    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes,
    };
    check(vkCreateDescriptorPool(c.device, &pool_info, nullptr, &descriptor_pool),
          "Cannot create descriptor pool");

    const VkDescriptorSetLayout set_layouts[] = {descriptor_layout, model_descriptor_layout};
    const VkDescriptorSetAllocateInfo set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 2,
        .pSetLayouts = set_layouts,
    };
    VkDescriptorSet sets[2];
    check(vkAllocateDescriptorSets(c.device, &set_info, sets),
          "Cannot allocate descriptor sets");
    descriptor_set = sets[0];
    model_descriptor_set = sets[1];

    writeBufferDescriptor(descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                          uniform_buffer, sizeof(SceneData));
    writeBufferDescriptor(model_descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                          model_buffer.buffer, sizeof(Transform));
    uploadLights(point_buffer, point_lights, 1);
    uploadLights(spot_buffer, spot_lights, 2);

    makePipeline();

    return true;
}

void shutdown() {
    const auto& c = graphics::internal::context;

    // сначала дожидаемся завершения работы с ресурсами на GPU
    vkQueueWaitIdle(c.graphics_queue);

    vkDestroyPipeline(c.device, pipeline, nullptr);
    vkDestroyPipeline(c.device, marker_pipeline, nullptr);
    vkDestroyPipeline(c.device, line_pipeline, nullptr);
    vkDestroyPipelineLayout(c.device, layout, nullptr);
    vkDestroyPipelineLayout(c.device, marker_layout, nullptr);
    vkDestroyDescriptorPool(c.device, descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(c.device, descriptor_layout, nullptr);
    vkDestroyDescriptorSetLayout(c.device, model_descriptor_layout, nullptr);

    destroyMappedBuffer(point_buffer);
    destroyMappedBuffer(spot_buffer);
    destroyMappedBuffer(model_buffer);
    destroyMappedBuffer(marker_buffer);
    destroyMappedBuffer(line_buffer);

    vmaUnmapMemory(c.allocator, uniform_allocation);
    vmaDestroyBuffer(c.allocator, uniform_buffer, uniform_allocation);
    vmaDestroyBuffer(c.allocator, buffer, allocation);
}

void update(double) {
    // вращаем объект только при перетаскивании вне интерфейса
    const ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        object_rotation_deg[1] += io.MouseDelta.x * 0.4f;
        object_rotation_deg[0] += io.MouseDelta.y * 0.4f;
    }

    ImGui::SetNextWindowPos(ImVec2(15, 15), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(440, 650), ImGuiCond_FirstUseEver);
    ImGui::Begin("Cone - lab 2");

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Camera position", camera_position, 0.02f);
        ImGui::DragFloat3("Camera rotation", camera_rotation_deg, 0.5f);
        ImGui::SliderFloat("Field of view", &camera_fov, 10.0f, 120.0f);
        ImGui::DragFloat("Near plane", &near_plane, 0.01f, 0.01f, 10.0f);
        ImGui::DragFloat("Far plane", &far_plane, 1.0f, 1.0f, 500.0f);

        // проверяем границы и при ручном вводе с клавиатуры
        if (camera_fov < 10.0f) camera_fov = 10.0f;
        if (camera_fov > 120.0f) camera_fov = 120.0f;
        if (near_plane < 0.01f) near_plane = 0.01f;
        if (far_plane < near_plane + 0.1f) far_plane = near_plane + 0.1f;

        if (ImGui::Button("Reset camera")) {
            camera_position[0] = camera_position[1] = 0.0f;
            camera_position[2] = 4.5f;
            camera_rotation_deg[0] = camera_rotation_deg[1] = camera_rotation_deg[2] = 0.0f;
            camera_fov = 50.0f;
            near_plane = 0.1f;
            far_plane = 100.0f;
        }
    }

    if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Ray direction", light_direction, 0.01f, -1.0f, 1.0f);
        ImGui::ColorEdit3("Light color", light_color);
        ImGui::SliderFloat("Light intensity", &light_intensity, 0.0f, 3.0f);

        if (light_intensity < 0.0f) light_intensity = 0.0f;
    }

    if (ImGui::CollapsingHeader("Light visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show light markers", &show_light_markers);
        ImGui::Checkbox("Show spot cones", &show_spot_cones);
    }

    if (ImGui::CollapsingHeader("Point lights")) {
        ImGui::PushID("point_lights");
        if (ImGui::Button("Add point light"))
            point_lights.push_back({{2.0f, 1.0f, 2.0f, 5.0f}, {1.0f, 1.0f, 1.0f, 0.0f}});

        ImGui::Text("Count: %d", int(point_lights.size()));
        for (size_t i = 0; i < point_lights.size();) {
            ImGui::PushID(int(i));
            ImGui::Separator();
            ImGui::Text("Point light %d", int(i + 1));
            PointLight& light = point_lights[i];
            ImGui::DragFloat3("Position", &light.position_intensity.x, 0.02f);
            ImGui::ColorEdit3("Color", &light.color.x);
            ImGui::DragFloat("Intensity", &light.position_intensity.w, 0.1f, 0.0f, 100.0f);
            light.position_intensity.w = std::max(0.0f, light.position_intensity.w);

            const bool remove = ImGui::Button("Remove point light");
            ImGui::PopID();
            if (remove)
                point_lights.erase(point_lights.begin() + i);
            else
                ++i;
        }
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Spot lights")) {
        ImGui::PushID("spot_lights");
        if (ImGui::Button("Add spot light")) {
            spot_lights.push_back({
                {0.0f, 1.5f, 3.0f, 8.0f}, {1.0f, 1.0f, 1.0f, 0.0f},
                {0.0f, -0.45f, -1.0f, 15.0f}, {25.0f, 0.0f, 0.0f, 0.0f},
            });
        }

        ImGui::Text("Count: %d", int(spot_lights.size()));
        for (size_t i = 0; i < spot_lights.size();) {
            ImGui::PushID(int(i));
            ImGui::Separator();
            ImGui::Text("Spot light %d", int(i + 1));
            SpotLight& light = spot_lights[i];
            ImGui::DragFloat3("Position", &light.position_intensity.x, 0.02f);
            ImGui::DragFloat3("Direction", &light.direction_inner.x, 0.01f, -1.0f, 1.0f);
            ImGui::ColorEdit3("Color", &light.color.x);
            ImGui::DragFloat("Intensity", &light.position_intensity.w, 0.1f, 0.0f, 100.0f);
            ImGui::SliderFloat("Inner angle", &light.direction_inner.w, 0.0f, 89.0f);
            ImGui::SliderFloat("Outer angle", &light.outer_angle.x, 0.0f, 89.0f);

            // углы отсчитываются от оси прожектора, внешний не меньше внутреннего
            light.position_intensity.w = std::max(0.0f, light.position_intensity.w);
            light.direction_inner.w = std::clamp(light.direction_inner.w, 0.0f, 89.0f);
            light.outer_angle.x = std::clamp(light.outer_angle.x, light.direction_inner.w, 89.0f);

            const bool remove = ImGui::Button("Remove spot light");
            ImGui::PopID();
            if (remove)
                spot_lights.erase(spot_lights.begin() + i);
            else
                ++i;
        }
        ImGui::PopID();
    }

    if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Albedo", selected_color);
        ImGui::ColorEdit3("Specular color", specular_color);
        ImGui::SliderFloat("Shininess", &shininess, 1.0f, 256.0f);

        if (shininess < 1.0f) shininess = 1.0f;
    }

    if (ImGui::CollapsingHeader("Ambient light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient color", ambient_color);
        ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0.0f, 1.0f);

        if (ambient_intensity < 0.0f) ambient_intensity = 0.0f;
    }

    if (ImGui::CollapsingHeader("Object")) {
        ImGui::DragFloat3("Position", object_position, 0.02f);
        ImGui::DragFloat3("Rotation (degrees)", object_rotation_deg, 0.5f);
        ImGui::DragFloat3("Scale", object_scale, 0.01f, 0.1f, 3.0f);

        for (float& scale : object_scale)
            if (scale < 0.1f) scale = 0.1f;
    }

    ImGui::TextUnformatted("Camera rotation: X / Y / Z, degrees");
    ImGui::TextUnformatted("Left mouse drag outside UI: rotate object");
    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    const auto& c = graphics::internal::context;

    check(vkResetCommandBuffer(fd.command_buffer, 0), "Cannot reset command buffer");
    updateLightMarkers();

    const VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    check(vkBeginCommandBuffer(fd.command_buffer, &begin), "Cannot begin command buffer");

    VkClearValue clear[2] = {};
    clear[0].color = {{0.08f, 0.10f, 0.14f, 1}};
    clear[1].depthStencil = {1, 0};
    const VkRenderPassBeginInfo pass = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = c.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = {{0, 0}, c.swapchain_extent},
        .clearValueCount = 2,
        .pClearValues = clear,
    };
    vkCmdBeginRenderPass(fd.command_buffer, &pass, VK_SUBPASS_CONTENTS_INLINE);

    // подключаем конвейер и геометрию конуса
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &buffer, &offset);

    const VkViewport viewport = {0, 0, float(c.swapchain_extent.width),
                                 float(c.swapchain_extent.height), 0, 1};
    const VkRect2D scissor = {{0, 0}, c.swapchain_extent};
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    Matrix4 model = rotationMatrix(object_rotation_deg);
    for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
            model.values[column * 4 + row] *= object_scale[column];
    model.values[12] = object_position[0];
    model.values[13] = object_position[1];
    model.values[14] = object_position[2];

    const Transform transform = {
        model,
        {selected_color[0], selected_color[1], selected_color[2], 0.0f},
        {specular_color[0], specular_color[1], specular_color[2], shininess},
    };
    const SceneData scene = {
        makeViewProjection(float(c.swapchain_extent.width) / float(c.swapchain_extent.height)),
        {camera_position[0], camera_position[1], camera_position[2], 0.0f},
        {light_direction[0], light_direction[1], light_direction[2], light_intensity},
        {light_color[0], light_color[1], light_color[2], 0.0f},
        {ambient_color[0], ambient_color[1], ambient_color[2], ambient_intensity},
        {uint32_t(point_lights.size()), uint32_t(spot_lights.size()), 0, 0},
    };

    // prepare() дождался предыдущего кадра: можно обновлять и заменять буферы
    uploadLights(point_buffer, point_lights, 1);
    uploadLights(spot_buffer, spot_lights, 2);
    std::memcpy(uniform_data, &scene, sizeof(scene));
    std::memcpy(model_buffer.data, &transform, sizeof(transform));
    check(vmaFlushAllocation(c.allocator, uniform_allocation, 0, sizeof(scene)),
          "Cannot flush scene uniform buffer");
    check(vmaFlushAllocation(c.allocator, model_buffer.allocation, 0, sizeof(transform)),
          "Cannot flush model uniform buffer");

    const VkDescriptorSet sets[] = {descriptor_set, model_descriptor_set};
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            layout, 0, 2, sets, 0, nullptr);

    vkCmdDraw(fd.command_buffer, uint32_t(vertices.size()), 1, 0, 0);

    if (!marker_vertices.empty()) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, marker_pipeline);
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               marker_layout, 0, 1, &descriptor_set, 0, nullptr);
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &marker_buffer.buffer, &offset);
        vkCmdDraw(fd.command_buffer, uint32_t(marker_vertices.size()), 1, 0, 0);
    }

    if (!line_vertices.empty()) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &line_buffer.buffer, &offset);
        vkCmdDraw(fd.command_buffer, uint32_t(line_vertices.size()), 1, 0, 0);
    }

    vkCmdEndRenderPass(fd.command_buffer);
    check(vkEndCommandBuffer(fd.command_buffer), "Cannot end command buffer");
}

} // namespace application
