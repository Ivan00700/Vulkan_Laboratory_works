#include "application.hpp"
#include <imgui.h>
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
struct Vertex { float x, y, z, nx, ny, nz, r, g, b; };
struct Vec4 { float x, y, z, w; };
// параметры одного кадра передаются шейдерам через push-константы
struct Transform {
    Vec4 position_projection; // xyz — положение, w — режим проекции
    Vec4 rotation;            // xyz — углы поворота в радианах
    Vec4 scale;               // xyz — масштаб по осям
    Vec4 color;               // rgb — цвет, выбранный в интерфейсе
    Vec4 view;                // x — расстояние, y — отношение ширины к высоте
};
static_assert(sizeof(Transform) == 80);
std::vector<Vertex> vertices;
VkBuffer buffer = VK_NULL_HANDLE;
VmaAllocation allocation = VK_NULL_HANDLE;
VkPipelineLayout layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
float object_position[3] = {0.0f, 0.0f, 0.0f};
float object_rotation_deg[3] = {-17.0f, 29.0f, 0.0f};
float object_scale[3] = {1.0f, 1.0f, 1.0f};
float selected_color[3] = {1.0f, 0.65f, 0.35f};
float distance_to_cone = 4.5f;
int projection_mode = 0; // перспектива — 0, ортографическая проекция — 1
bool animation_playing = true;
float animation_speed = 1.0f;
float trajectory_radius = 0.65f;
float animation_phase = 0.0f;
double previous_time = -1.0;

void check(VkResult result, const char* message) {
    if (result != VK_SUCCESS) throw std::runtime_error(message);
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
        // углы начала и конца сектора окружности
        const float a = 2.0f * std::numbers::pi_v<float> * float(i) / segments;
        const float b = 2.0f * std::numbers::pi_v<float> * float(i + 1) / segments;

        // угол посередине сектора для нормали
        const float m = (a + b) * 0.5f;

        // две соседние точки на окружности основания
        const float x0 = radius * std::cos(a), z0 = radius * std::sin(a);
        const float x1 = radius * std::cos(b), z1 = radius * std::sin(b);

        // нормаль боковой грани нужна для расчёта освещения
        const float normal_scale = 1.0f / std::sqrt(1.0f + radius * radius / (height * height));
        const float nx = std::cos(m) * normal_scale;
        const float ny = radius / height * normal_scale;
        const float nz = std::sin(m) * normal_scale;

        // каждые три последовательные вершины образуют один треугольник
        addVertex(0, height / 2, 0, nx, ny, nz);
        addVertex(x1, -height / 2, z1, nx, ny, nz);
        addVertex(x0, -height / 2, z0, nx, ny, nz);
        addVertex(0, -height / 2, 0, 0, -1, 0);
        addVertex(x0, -height / 2, z0, 0, -1, 0);
        addVertex(x1, -height / 2, z1, 0, -1, 0);
    }
}

VkShaderModule makeShader(const char* path) {
    // при сборке шейдеры компилируются в двоичный формат SPIR-V
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open compiled shader");
    const std::streamsize bytes = file.tellg();
    if (bytes <= 0 || bytes % sizeof(uint32_t) != 0)
        throw std::runtime_error("Invalid compiled shader size");
    std::vector<uint32_t> code(size_t(bytes) / sizeof(uint32_t));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(code.data()), bytes))
        throw std::runtime_error("Cannot read compiled shader");
    const VkShaderModuleCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size_t(bytes), .pCode = code.data(),
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
         .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"},
    };
    const VkVertexInputBindingDescription binding = {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    // три атрибута вершины: координаты, нормаль и процедурный цвет
    const VkVertexInputAttributeDescription attributes[] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, x)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, nx)},
        {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, r)},
    };
    const VkPipelineVertexInputStateCreateInfo input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = attributes,
    };
    const VkPipelineInputAssemblyStateCreateInfo assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    const VkPipelineViewportStateCreateInfo viewport = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1,
    };
    const VkPipelineRasterizationStateCreateInfo raster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1,
    };
    const VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const VkPipelineDepthStencilStateCreateInfo depth = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };
    const VkPipelineColorBlendAttachmentState color = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &color,
    };
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dynamic_states,
    };
    // одни и те же параметры кадра доступны вершинному и фрагментному шейдерам
    const VkPushConstantRange range = {
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Transform)
    };
    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &range,
    };
    check(vkCreatePipelineLayout(c.device, &layout_info, nullptr, &layout),
          "Cannot create pipeline layout");
    const VkGraphicsPipelineCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = stages, .pVertexInputState = &input,
        .pInputAssemblyState = &assembly, .pViewportState = &viewport,
        .pRasterizationState = &raster, .pMultisampleState = &ms,
        .pDepthStencilState = &depth, .pColorBlendState = &blend,
        .pDynamicState = &dynamic, .layout = layout,
        .renderPass = c.render_pass, .subpass = 0,
    };
    const VkResult result = vkCreateGraphicsPipelines(c.device, VK_NULL_HANDLE, 1,
                                                       &info, nullptr, &pipeline);
    vkDestroyShaderModule(c.device, fs, nullptr);
    vkDestroyShaderModule(c.device, vs, nullptr);
    check(result, "Cannot create graphics pipeline");
}
}

bool initialize() {
    const auto& c = graphics::internal::context;
    buildCone();
    const VkDeviceSize bytes = vertices.size() * sizeof(Vertex); // размер данных в байтах
    // буфер вершин создаётся один раз, так как геометрия конуса не меняется
    const VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    };
    check(vmaCreateBuffer(c.allocator, &info, &alloc_info, &buffer, &allocation, nullptr),
          "Cannot allocate cone vertices");
    // копируем вершины из памяти программы в буфер, который читает Vulkan
    void* data = nullptr;
    check(vmaMapMemory(c.allocator, allocation, &data), "Cannot map vertex buffer");
    std::memcpy(data, vertices.data(), size_t(bytes));
    vmaFlushAllocation(c.allocator, allocation, 0, bytes);
    vmaUnmapMemory(c.allocator, allocation);
    makePipeline();
    return true;
}

void shutdown() {
    const auto& c = graphics::internal::context;
    vkQueueWaitIdle(c.graphics_queue);
    vkDestroyPipeline(c.device, pipeline, nullptr);
    vkDestroyPipelineLayout(c.device, layout, nullptr);
    vmaDestroyBuffer(c.allocator, buffer, allocation);
}

void update(double time) {
    // фаза меняется только при воспроизведении, с учётом времени между кадрами
    if (previous_time >= 0.0 && animation_playing) {
        const double elapsed = time - previous_time;
        if (elapsed > 0.0 && elapsed < 1.0) {
            animation_phase = std::fmod(
                animation_phase + float(elapsed) * animation_speed,
                2.0f * std::numbers::pi_v<float>);
        }
    }
    previous_time = time;

    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            constexpr float degrees_per_pixel = 0.008f * 180.0f / std::numbers::pi_v<float>;
            object_rotation_deg[1] += io.MouseDelta.x * degrees_per_pixel;
            object_rotation_deg[0] += io.MouseDelta.y * degrees_per_pixel;
        }
        distance_to_cone -= io.MouseWheel * 0.3f;
        if (distance_to_cone < 2.5f) distance_to_cone = 2.5f;
        if (distance_to_cone > 10.0f) distance_to_cone = 10.0f;
    }
    ImGui::Begin("Cone - lab 1");
    // элементы интерфейса изменяют базовые параметры объекта
    const char* projections[] = {"Perspective", "Orthographic"};
    ImGui::Combo("Projection", &projection_mode, projections, 2);
    ImGui::DragFloat3("Position", object_position, 0.02f);
    ImGui::DragFloat3("Rotation (degrees)", object_rotation_deg, 0.5f);
    ImGui::DragFloat3("Scale", object_scale, 0.01f, 0.1f, 3.0f);
    ImGui::ColorEdit3("Color", selected_color);
    if (ImGui::Button(animation_playing ? "Pause animation" : "Play animation"))
        animation_playing = !animation_playing;
    ImGui::SliderFloat("Animation speed", &animation_speed, 0.1f, 3.0f);
    ImGui::SliderFloat("Trajectory radius", &trajectory_radius, 0.0f, 1.2f);
    ImGui::TextUnformatted("Mouse drag: rotate; wheel: zoom");
    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    const auto& c = graphics::internal::context;
    check(vkResetCommandBuffer(fd.command_buffer, 0), "Cannot reset command buffer");
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
        .renderPass = c.render_pass, .framebuffer = fd.framebuffer,
        .renderArea = {{0, 0}, c.swapchain_extent},
        .clearValueCount = 2, .pClearValues = clear,
    };
    vkCmdBeginRenderPass(fd.command_buffer, &pass, VK_SUBPASS_CONTENTS_INLINE);
    // конвейер задаёт способ рисования, а буфер содержит вершины конуса
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &buffer, &offset);
    const VkViewport viewport = {0, 0, float(c.swapchain_extent.width),
                                 float(c.swapchain_extent.height), 0, 1};
    const VkRect2D scissor = {{0, 0}, c.swapchain_extent};
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
    const float phase = animation_phase;
    const float radius = trajectory_radius;
    const float degree_to_radian = std::numbers::pi_v<float> / 180.0f;
    // смещение и поворот по траектории добавляются к значениям из интерфейса
    const Transform transform = {
        {object_position[0] + radius * std::cos(phase),
         object_position[1] + 0.35f * radius * std::sin(3.0f * phase),
         object_position[2] + 0.6f * radius * std::sin(2.0f * phase),
         float(projection_mode)},
        {object_rotation_deg[0] * degree_to_radian + 0.25f * std::sin(2.0f * phase),
         object_rotation_deg[1] * degree_to_radian + phase,
         object_rotation_deg[2] * degree_to_radian, 0.0f},
        {object_scale[0], object_scale[1], object_scale[2], 0.0f},
        {selected_color[0], selected_color[1], selected_color[2], 1.0f},
        {distance_to_cone,
         float(c.swapchain_extent.width) / float(c.swapchain_extent.height), 0.0f, 0.0f},
    };
    // параметры текущего кадра передаются обоим шейдерам перед рисованием
    vkCmdPushConstants(fd.command_buffer, layout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(transform), &transform);
    vkCmdDraw(fd.command_buffer, uint32_t(vertices.size()), 1, 0, 0);
    vkCmdEndRenderPass(fd.command_buffer);
    check(vkEndCommandBuffer(fd.command_buffer), "Cannot end command buffer");
}
}
