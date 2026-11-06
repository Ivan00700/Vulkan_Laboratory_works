#include "gpu_resources.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>

namespace gpu {
namespace {

VkCommandPool transfer_pool = VK_NULL_HANDLE;

void check(VkResult result, const char* message) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(message);
}

VkCommandBuffer beginTransfer() {
    const auto& c = graphics::internal::context;
    const VkCommandBufferAllocateInfo info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = transfer_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer command;
    check(vkAllocateCommandBuffers(c.device, &info, &command), "Cannot allocate transfer command");
    const VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    check(vkBeginCommandBuffer(command, &begin), "Cannot begin transfer");
    return command;
}

void endTransfer(VkCommandBuffer command) {
    const auto& c = graphics::internal::context;
    check(vkEndCommandBuffer(command), "Cannot end transfer");
    const VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command,
    };
    check(vkQueueSubmit(c.graphics_queue, 1, &submit, VK_NULL_HANDLE), "Cannot submit transfer");
    check(vkQueueWaitIdle(c.graphics_queue), "Cannot wait for transfer");
    vkFreeCommandBuffers(c.device, transfer_pool, 1, &command);
}

Buffer stagingBuffer(const void* data, VkDeviceSize size) {
    const auto& c = graphics::internal::context;
    const VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocation = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    };
    Buffer result;
    check(vmaCreateBuffer(c.allocator, &info, &allocation, &result.buffer,
                          &result.allocation, nullptr), "Cannot allocate staging buffer");
    void* mapped = nullptr;
    check(vmaMapMemory(c.allocator, result.allocation, &mapped), "Cannot map staging buffer");
    std::memcpy(mapped, data, size_t(size));
    check(vmaFlushAllocation(c.allocator, result.allocation, 0, size), "Cannot flush staging buffer");
    vmaUnmapMemory(c.allocator, result.allocation);
    return result;
}

} // namespace

void initialize() {
    const auto& c = graphics::internal::context;
    const VkCommandPoolCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = c.graphics_queue_index,
    };
    check(vkCreateCommandPool(c.device, &info, nullptr, &transfer_pool), "Cannot create transfer pool");
}

void shutdown() {
    vkDestroyCommandPool(graphics::internal::context.device, transfer_pool, nullptr);
    transfer_pool = VK_NULL_HANDLE;
}

Buffer uploadBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage) {
    const auto& c = graphics::internal::context;
    Buffer staging = stagingBuffer(data, size);
    const VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocation = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };
    Buffer result;
    check(vmaCreateBuffer(c.allocator, &info, &allocation, &result.buffer,
                          &result.allocation, nullptr), "Cannot allocate GPU buffer");
    const VkCommandBuffer command = beginTransfer();
    const VkBufferCopy copy = {0, 0, size};
    vkCmdCopyBuffer(command, staging.buffer, result.buffer, 1, &copy);
    const VkBufferMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = result.buffer,
        .size = size,
    };
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
    endTransfer(command);
    destroy(staging);
    return result;
}

Texture loadTexture(const char* path, VkFormat format) {
    int width, height, channels;
    stbi_uc* pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels)
        throw std::runtime_error(std::string("Cannot load texture: ") + path);
    Buffer staging = stagingBuffer(pixels, VkDeviceSize(width) * height * 4);
    stbi_image_free(pixels);

    const auto& c = graphics::internal::context;
    const VkImageCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {uint32_t(width), uint32_t(height), 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const VmaAllocationCreateInfo allocation = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };
    Texture texture;
    check(vmaCreateImage(c.allocator, &info, &allocation, &texture.image,
                         &texture.allocation, nullptr), "Cannot allocate texture");

    const VkCommandBuffer command = beginTransfer();
    VkImageMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = texture.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    const VkBufferImageCopy copy = {
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = info.extent,
    };
    vkCmdCopyBufferToImage(command, staging.buffer, texture.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    endTransfer(command);
    destroy(staging);

    const VkImageViewCreateInfo view = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = texture.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = info.format,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    check(vkCreateImageView(c.device, &view, nullptr, &texture.view), "Cannot create texture view");
    const VkSamplerCreateInfo sampler = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .maxLod = 0.0f,
    };
    check(vkCreateSampler(c.device, &sampler, nullptr, &texture.sampler), "Cannot create sampler");
    return texture;
}

void destroy(Buffer& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
    buffer = {};
}

void destroy(Texture& texture) {
    const auto& c = graphics::internal::context;
    vkDestroySampler(c.device, texture.sampler, nullptr);
    vkDestroyImageView(c.device, texture.view, nullptr);
    if (texture.image != VK_NULL_HANDLE)
        vmaDestroyImage(c.allocator, texture.image, texture.allocation);
    texture = {};
}

} // namespace gpu
