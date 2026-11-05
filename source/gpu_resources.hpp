#pragma once

#include "graphics_internal.hpp"

namespace gpu {

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
};

void initialize();
void shutdown();
Buffer uploadBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage);
Texture loadTexture(const char* path);
void destroy(Buffer& buffer);
void destroy(Texture& texture);

} // namespace gpu
