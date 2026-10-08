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

// x y z задают позицию в 3D
// nx ny nz задают нормаль к поверхности в этой точке для корректного освещения
struct Vertex { float x, y, z, nx, ny, nz; };


struct Transform { float yaw, pitch, aspect, distance; };
std::vector<Vertex> vertices;
VkBuffer buffer = VK_NULL_HANDLE;
VmaAllocation allocation = VK_NULL_HANDLE;
VkPipelineLayout layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
float yaw = 0.5f, pitch = -0.3f, distance_to_cone = 4.5f;

void check(VkResult result, const char* message) {
    if (result != VK_SUCCESS) throw std::runtime_error(message);
}

// 48 треугольников в боковине и 48 треугольников в основании конуса всего 96
void buildCone() {
    constexpr uint32_t segments = 48;
    constexpr float radius = 1.0f, height = 2.0f;
    vertices.reserve(segments * 6);
    for (uint32_t i = 0; i < segments; ++i) {
        // углы начала и конца сектора окружности
        const float a = 2.0f * std::numbers::pi_v<float> * float(i) / segments;
        const float b = 2.0f * std::numbers::pi_v<float> * float(i + 1) / segments;

        // угол посередине сектора для нормали
        const float m = (a + b) * 0.5f;

        // две соседние точки на окружности основания
        const float x0 = radius * std::cos(a), z0 = radius * std::sin(a);
        const float x1 = radius * std::cos(b), z1 = radius * std::sin(b);

        // для освещения
        const float normal_scale = 1.0f / std::sqrt(1.0f + radius * radius / (height * height));
        const float nx = std::cos(m) * normal_scale;
        const float ny = radius / height * normal_scale;
        const float nz = std::sin(m) * normal_scale;

        // добавление точек треугольника боковины и основания
        vertices.push_back({0, height / 2, 0, nx, ny, nz});
        vertices.push_back({x1, -height / 2, z1, nx, ny, nz});
        vertices.push_back({x0, -height / 2, z0, nx, ny, nz});
        vertices.push_back({0, -height / 2, 0, 0, -1, 0});
        vertices.push_back({x0, -height / 2, z0, 0, -1, 0});
        vertices.push_back({x1, -height / 2, z1, 0, -1, 0});
    }
}

VkShaderModule makeShader(const char* path) {
    // CMake компилирует шейдеры в бинарный формат SPIR-V
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
    const VkPipelineShaderStageCreateInfo   [] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"},
    };
    const VkVertexInputBindingDescription binding = {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription attributes[] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, x)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, nx)},
    };
    const VkPipelineVertexInputStateCreateInfo input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = attributes,
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
    const VkPushConstantRange range = {VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Transform)};
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
    const VkDeviceSize bytes = vertices.size() * sizeof(Vertex); // размер буфера в байтах
    // размер и назначение буфера
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

void update([[maybe_unused]] double time) {
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            yaw += io.MouseDelta.x * 0.008f; // скорость вращения камеры вокруг конуса
            pitch += io.MouseDelta.y * 0.008f; // скорость вращения камеры вверх-вниз
            if (pitch > 1.5f) pitch = 1.5f;
            if (pitch < -1.5f) pitch = -1.5f;
        }
        distance_to_cone -= io.MouseWheel * 0.3f;
        if (distance_to_cone < 2.5f) distance_to_cone = 2.5f;
        if (distance_to_cone > 10.0f) distance_to_cone = 10.0f;
    }
    ImGui::Begin("Cone - basic task");
    ImGui::Text("48 segments, %zu triangles", vertices.size() / 3);
    ImGui::TextUnformatted("Left mouse drag: rotate");
    ImGui::TextUnformatted("Mouse wheel: zoom");
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
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &buffer, &offset);
    const VkViewport viewport = {0, 0, float(c.swapchain_extent.width),
                                 float(c.swapchain_extent.height), 0, 1};
    const VkRect2D scissor = {{0, 0}, c.swapchain_extent};
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
    const Transform transform = {yaw, pitch,
        float(c.swapchain_extent.width) / float(c.swapchain_extent.height), distance_to_cone};
    vkCmdPushConstants(fd.command_buffer, layout, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(transform), &transform);
    vkCmdDraw(fd.command_buffer, uint32_t(vertices.size()), 1, 0, 0);
    vkCmdEndRenderPass(fd.command_buffer);
    check(vkEndCommandBuffer(fd.command_buffer), "Cannot end command buffer");
}
}
