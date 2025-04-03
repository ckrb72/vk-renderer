#include <iostream>
#include <fstream>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>
#include <vector>
#include <span>
#include "vma.h"

const int WIN_WIDTH = 1920;
const int WIN_HEIGHT = 1080;

constexpr uint32_t FRAME_OVERLAP = 2;

struct AllocatedImage
{
    VkImage image;
    VkImageView image_view;
    VmaAllocation allocation;
    VkExtent3D extent;
    VkFormat format;
};

struct DescriptorLayoutBuilder
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    void add_binding(uint32_t binding, VkDescriptorType type);
    void clear();
    VkDescriptorSetLayout build(VkDevice device, VkShaderStageFlags shader_stages, void* pNext = nullptr, VkDescriptorSetLayoutCreateFlags = 0);
};

struct DescriptorAllocator
{
    struct PoolSizeRatio
    {
        VkDescriptorType type;
        float ratio;
    };

    VkDescriptorPool pool;

    void init_pool(VkDevice device, uint32_t max_sets, std::span<PoolSizeRatio> pool_ratios);
    void clear_descriptors(VkDevice device);
    void destroy_pool(VkDevice device);

    VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout);
};

struct VulkanContext
{
    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkPhysicalDevice gpu;
    VkDevice device;
    VkSurfaceKHR surface;
    VkQueue graphics_queue;
    uint32_t graphics_queue_family;

    struct VulkanSwapchain
    {
        VkSwapchainKHR swapchain;
        VkFormat format;
        std::vector<VkImage> images;
        std::vector<VkImageView> image_views;
        VkExtent2D extent;
    };

    VulkanSwapchain swapchain;


    struct FrameData
    {
        VkCommandPool command_pool;
        VkCommandBuffer command_buffer;
        VkSemaphore swapchain_semaphore, render_semaphore;
        VkFence render_fence;
    };

    FrameData frames[FRAME_OVERLAP];

    VmaAllocator allocator;

    AllocatedImage draw_image;
    VkExtent2D draw_extent;

    DescriptorAllocator descriptor_allocator;
    VkDescriptorSet draw_image_descriptors;
    VkDescriptorSetLayout draw_image_descriptor_layout;

    VkPipeline gradient_pipeline;
    VkPipelineLayout gradient_pipeline_layout;

};


void init_vulkan(VulkanContext& context, GLFWwindow* window);
void create_swapchain(uint32_t width, uint32_t height, VulkanContext& context);
void destroy_swapchain(VulkanContext& context);
void init_commands(VulkanContext& context);
void init_sync_structures(VulkanContext& context);
void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout current_layout, VkImageLayout new_layout);
VkSemaphoreSubmitInfo semaphore_submit_info(VkPipelineStageFlags2 stage_mask, VkSemaphore semaphore);
VkCommandBufferSubmitInfo command_buffer_submit_info(VkCommandBuffer command_buffer);
VkSubmitInfo2 submit_info(VkCommandBufferSubmitInfo* cmd, VkSemaphoreSubmitInfo* signal_semaphore_info, VkSemaphoreSubmitInfo* wait_semaphore_info);
VkImageCreateInfo image_create_info(VkFormat format, VkImageUsageFlags usage_flags, VkExtent3D extent);
VkImageViewCreateInfo imageview_create_info(VkFormat format, VkImage image, VkImageAspectFlags aspect_flags);
void copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D src_size, VkExtent2D dst_size);
void init_descriptors(VulkanContext& context);
void init_pipelines(VulkanContext& context);
void init_background_pipelines(VulkanContext& context);

#define VK_CHECK(func) func

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow* window = glfwCreateWindow(WIN_WIDTH, WIN_HEIGHT, "vk-renderer", NULL, NULL);
    if(!window)
    {
        std::cerr << "Failed to create glfw context" << std::endl;
        return -1;
    }

    VulkanContext context;
    init_vulkan(context, window);
    create_swapchain(WIN_WIDTH, WIN_HEIGHT, context);
    init_commands(context);
    init_sync_structures(context);
    init_descriptors(context);
    init_pipelines(context);

    uint32_t current_frame = 0;
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        vkWaitForFences(context.device, 1, &context.frames[current_frame % FRAME_OVERLAP].render_fence, true, UINT64_MAX);
        vkResetFences(context.device, 1, &context.frames[current_frame % FRAME_OVERLAP].render_fence);

        uint32_t swapchain_index;
        vkAcquireNextImageKHR(context.device, context.swapchain.swapchain, UINT64_MAX, context.frames[current_frame % FRAME_OVERLAP].swapchain_semaphore, nullptr, &swapchain_index);
        vkResetCommandBuffer(context.frames[current_frame % FRAME_OVERLAP].command_buffer, 0);

        context.draw_extent.width = context.draw_image.extent.width;
        context.draw_extent.height = context.draw_image.extent.height;

        // Record Command Buffer
        VkCommandBufferBeginInfo cmd_buffer_info = {};
        cmd_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cmd_buffer_info.pNext = nullptr;
        cmd_buffer_info.pInheritanceInfo = nullptr;
        cmd_buffer_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(context.frames[current_frame % FRAME_OVERLAP].command_buffer, &cmd_buffer_info);

        transition_image(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.draw_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

        /*VkClearColorValue clear_value;
        float flash = std::abs(std::sin(current_frame / 120.0f));
        clear_value = { {0.0f, 0.0f, flash, 1.0f } };

        VkImageSubresourceRange clear_range = {};
        clear_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        clear_range.baseMipLevel = 0;
        clear_range.levelCount = VK_REMAINING_MIP_LEVELS;
        clear_range.baseArrayLayer = 0;
        clear_range.layerCount = VK_REMAINING_ARRAY_LAYERS;

        vkCmdClearColorImage(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.draw_image.image, VK_IMAGE_LAYOUT_GENERAL, &clear_value, 1, &clear_range);*/

        vkCmdBindPipeline(context.frames[current_frame % FRAME_OVERLAP].command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, context.gradient_pipeline);
        vkCmdBindDescriptorSets(context.frames[current_frame % FRAME_OVERLAP].command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, context.gradient_pipeline_layout, 0, 1, &context.draw_image_descriptors, 0, nullptr);
        vkCmdDispatch(context.frames[current_frame % FRAME_OVERLAP].command_buffer, std::ceil(context.draw_extent.width / 16.0), std::ceil(context.draw_extent.height / 16.0), 1);

        transition_image(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.draw_image.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        transition_image(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.swapchain.images[swapchain_index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copy_image_to_image(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.draw_image.image, context.swapchain.images[swapchain_index], context.draw_extent, context.swapchain.extent);
        transition_image(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.swapchain.images[swapchain_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

        vkEndCommandBuffer(context.frames[current_frame % FRAME_OVERLAP].command_buffer);

        VkCommandBufferSubmitInfo cmd_info = command_buffer_submit_info(context.frames[current_frame % FRAME_OVERLAP].command_buffer);
        VkSemaphoreSubmitInfo wait_info = semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, context.frames[current_frame % FRAME_OVERLAP].swapchain_semaphore);
        VkSemaphoreSubmitInfo signal_info = semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, context.frames[current_frame % FRAME_OVERLAP].render_semaphore);

        VkSubmitInfo2 submit = submit_info(&cmd_info, &signal_info, &wait_info);

        // Note: Settings render_fence to be signaled when these commands are done so when we get back to the top we will wait until these commands are done (depending on how far ahead the cpu is)
        vkQueueSubmit2(context.graphics_queue, 1, &submit, context.frames[current_frame % FRAME_OVERLAP].render_fence);

        VkPresentInfoKHR present_info = {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.pSwapchains = &context.swapchain.swapchain;
        present_info.swapchainCount = 1;

        present_info.pWaitSemaphores = &context.frames[current_frame % FRAME_OVERLAP].render_semaphore;
        present_info.waitSemaphoreCount = 1;

        present_info.pImageIndices = &swapchain_index;

        vkQueuePresentKHR(context.graphics_queue, &present_info);

        current_frame++;

    }

    vkDeviceWaitIdle(context.device);

    vkDestroyPipelineLayout(context.device, context.gradient_pipeline_layout, nullptr);
    vkDestroyPipeline(context.device, context.gradient_pipeline, nullptr);

    context.descriptor_allocator.destroy_pool(context.device);
    vkDestroyDescriptorSetLayout(context.device, context.draw_image_descriptor_layout, nullptr);

    for(int i = 0; i < FRAME_OVERLAP; i++)
    {
        vkDestroyCommandPool(context.device, context.frames[i].command_pool, NULL);

        vkDestroyFence(context.device, context.frames[i].render_fence, NULL);
        vkDestroySemaphore(context.device, context.frames[i].render_semaphore, NULL);
        vkDestroySemaphore(context.device, context.frames[i].swapchain_semaphore, NULL);
    }

    vkDestroyImageView(context.device, context.draw_image.image_view, NULL);
    vmaDestroyImage(context.allocator, context.draw_image.image, context.draw_image.allocation);

    vmaDestroyAllocator(context.allocator);

    destroy_swapchain(context);
    vkDestroySurfaceKHR(context.instance, context.surface, NULL);
    vkDestroyDevice(context.device, NULL);
    vkb::destroy_debug_utils_messenger(context.instance, context.debug_messenger);
    vkDestroyInstance(context.instance, NULL);

    glfwDestroyWindow(window);
    glfwTerminate();
}

void init_vulkan(VulkanContext& context, GLFWwindow* window)
{
    vkb::InstanceBuilder builder;
    vkb::Result<vkb::Instance> inst_ret = builder.set_app_name("vk-renderer")
                           .request_validation_layers(true)
                           .use_default_debug_messenger()
                           .require_api_version(1, 3, 0)
                           .build();

    vkb::Instance instance = inst_ret.value();

    context.instance = instance.instance;
    context.debug_messenger = instance.debug_messenger;

    glfwCreateWindowSurface(context.instance, window, NULL, &context.surface);

    VkPhysicalDeviceVulkan13Features features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features.dynamicRendering = true;
    features.synchronization2 = true;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing = true;
    
    vkb::PhysicalDeviceSelector selector(instance);
    auto physical_device = selector.set_minimum_version(1, 3)
                                   .set_required_features_13(features)
                                   .set_required_features_12(features12)
                                   .set_surface(context.surface)
                                   .select()
                                   .value();


    vkb::DeviceBuilder device_builder {physical_device};
    vkb::Device device = device_builder.build().value();

    context.device = device;
    context.gpu = physical_device;

    context.graphics_queue = device.get_queue(vkb::QueueType::graphics).value();
    context.graphics_queue_family = device.get_queue_index(vkb::QueueType::graphics).value();

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice = context.gpu;
    allocator_info.device = context.device;
    allocator_info.instance = context.instance;
    allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocator_info, &context.allocator);
}

void create_swapchain(uint32_t width, uint32_t height, VulkanContext& context)
{
    vkb::SwapchainBuilder builder(context.gpu, context.device, context.surface);
    context.swapchain.format = VK_FORMAT_B8G8R8A8_UNORM;

    VkSurfaceFormatKHR format;
    format.format = context.swapchain.format;
    format.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

    vkb::Swapchain vkb_swapchain = builder.set_desired_format(format)
                                   .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                                   .set_desired_extent(width, height)
                                   .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                                   .build()
                                   .value();
    context.swapchain.extent = vkb_swapchain.extent;
    context.swapchain.swapchain = vkb_swapchain.swapchain;
    context.swapchain.images = vkb_swapchain.get_images().value();
    context.swapchain.image_views = vkb_swapchain.get_image_views().value();

    VkExtent3D draw_image_extent = {};
    draw_image_extent.width = context.swapchain.extent.width;
    draw_image_extent.height = context.swapchain.extent.height;
    draw_image_extent.depth = 1;

    context.draw_image.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    context.draw_image.extent = draw_image_extent;

    VkImageUsageFlags draw_image_usages = {};
    draw_image_usages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    draw_image_usages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    draw_image_usages |= VK_IMAGE_USAGE_STORAGE_BIT;
    draw_image_usages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo rimg_info = image_create_info(context.draw_image.format, draw_image_usages, draw_image_extent);
    VmaAllocationCreateInfo rimg_allocinfo = {};
    rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vmaCreateImage(context.allocator, &rimg_info, &rimg_allocinfo, &context.draw_image.image, &context.draw_image.allocation, nullptr);

    VkImageViewCreateInfo rview_info = imageview_create_info(context.draw_image.format, context.draw_image.image, VK_IMAGE_ASPECT_COLOR_BIT);

    vkCreateImageView(context.device, &rview_info, nullptr, &context.draw_image.image_view);

}

void copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D src_size, VkExtent2D dst_size)
{
    VkImageBlit2 blit_region = {};
    blit_region.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;

    blit_region.srcOffsets[1].x = src_size.width;
    blit_region.srcOffsets[1].y = src_size.height;
    blit_region.srcOffsets[1].z = 1;

    blit_region.dstOffsets[1].x = dst_size.width;
    blit_region.dstOffsets[1].y = dst_size.height;
    blit_region.dstOffsets[1].z = 1;

    blit_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit_region.srcSubresource.baseArrayLayer = 0;
    blit_region.srcSubresource.layerCount = 1;
    blit_region.srcSubresource.mipLevel = 0;

    blit_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit_region.dstSubresource.baseArrayLayer = 0;
    blit_region.dstSubresource.layerCount = 1;
    blit_region.dstSubresource.mipLevel = 0;

    VkBlitImageInfo2 blit_info = {};
    blit_info.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
    blit_info.dstImage = destination;
    blit_info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    blit_info.srcImage = source;
    blit_info.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    blit_info.filter = VK_FILTER_LINEAR;
    blit_info.regionCount = 1;
    blit_info.pRegions = &blit_region;

    vkCmdBlitImage2(cmd, &blit_info);
}

void init_commands(VulkanContext& context)
{
    VkCommandPoolCreateInfo command_pool_info = {};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.pNext = nullptr;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = context.graphics_queue_family;

    for(int i = 0; i < FRAME_OVERLAP; i++)
    {
        vkCreateCommandPool(context.device, &command_pool_info, NULL, &context.frames[i].command_pool);

        VkCommandBufferAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.pNext = nullptr;
        alloc_info.commandPool = context.frames[i].command_pool;
        alloc_info.commandBufferCount = 1;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        vkAllocateCommandBuffers(context.device, &alloc_info, &context.frames[i].command_buffer);
    }
}

void init_sync_structures(VulkanContext& context)
{
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.pNext = nullptr;
    fence_info.flags =  VK_FENCE_CREATE_SIGNALED_BIT;

    VkSemaphoreCreateInfo semaphore_info = {};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphore_info.pNext = nullptr;

    for(int i = 0; i < FRAME_OVERLAP; i++)
    {
        vkCreateFence(context.device, &fence_info, NULL, &context.frames[i].render_fence);
        vkCreateSemaphore(context.device, &semaphore_info, NULL, &context.frames[i].swapchain_semaphore);
        vkCreateSemaphore(context.device, &semaphore_info, NULL, &context.frames[i].render_semaphore);
    }
}

void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout current_layout, VkImageLayout new_layout)
{
    VkImageMemoryBarrier2 image_barrier = {};
    image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    image_barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    image_barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    image_barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    image_barrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;

    image_barrier.oldLayout = current_layout;
    image_barrier.newLayout = new_layout;

    VkImageAspectFlags aspect_mask = (new_layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    
    VkImageSubresourceRange sub_image = {};
    sub_image.aspectMask = aspect_mask;
    sub_image.baseMipLevel = 0;
    sub_image.levelCount = VK_REMAINING_MIP_LEVELS;
    sub_image.baseArrayLayer = 0;
    sub_image.layerCount = VK_REMAINING_ARRAY_LAYERS;

    image_barrier.subresourceRange = sub_image;
    image_barrier.image = image;

    VkDependencyInfo dep_info = {};
    dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep_info.imageMemoryBarrierCount = 1;
    dep_info.pImageMemoryBarriers = &image_barrier;

    vkCmdPipelineBarrier2(cmd, &dep_info);
}

VkSemaphoreSubmitInfo semaphore_submit_info(VkPipelineStageFlags2 stage_mask, VkSemaphore semaphore)
{
    VkSemaphoreSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    submit_info.semaphore = semaphore;
    submit_info.stageMask = stage_mask;
    submit_info.deviceIndex = 0;
    submit_info.value = 1;

    return submit_info;
}

VkCommandBufferSubmitInfo command_buffer_submit_info(VkCommandBuffer command_buffer)
{
    VkCommandBufferSubmitInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    info.commandBuffer = command_buffer;
    info.deviceMask = 0;

    return info;
}

VkSubmitInfo2 submit_info(VkCommandBufferSubmitInfo* cmd, VkSemaphoreSubmitInfo* signal_semaphore_info, VkSemaphoreSubmitInfo* wait_semaphore_info)
{
    VkSubmitInfo2 info = {};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    info.waitSemaphoreInfoCount = wait_semaphore_info == nullptr ? 0 : 1;
    info.pWaitSemaphoreInfos = wait_semaphore_info;

    info.signalSemaphoreInfoCount = signal_semaphore_info  == nullptr ? 0 : 1;
    info.pSignalSemaphoreInfos = signal_semaphore_info;

    info.commandBufferInfoCount = 1;
    info.pCommandBufferInfos = cmd;

    return info;
}

VkImageCreateInfo image_create_info(VkFormat format, VkImageUsageFlags usage_flags, VkExtent3D extent)
{
    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = extent;

    info.mipLevels = 1;
    info.arrayLayers = 1;

    info.samples = VK_SAMPLE_COUNT_1_BIT;

    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage_flags;

    return info;
}

VkImageViewCreateInfo imageview_create_info(VkFormat format, VkImage image, VkImageAspectFlags aspect_flags)
{
    VkImageViewCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;

    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.image = image;
    info.format = format;
    info.subresourceRange.baseMipLevel = 0;
    info.subresourceRange.levelCount = 1;
    info.subresourceRange.baseArrayLayer = 0;
    info.subresourceRange.layerCount = 1;
    info.subresourceRange.aspectMask = aspect_flags;

    return info;
}

void destroy_swapchain(VulkanContext& context)
{
    vkDestroySwapchainKHR(context.device, context.swapchain.swapchain, NULL);

    for(const VkImageView& view : context.swapchain.image_views)
    {
        vkDestroyImageView(context.device, view, NULL);
    }
}

void DescriptorLayoutBuilder::add_binding(uint32_t binding, VkDescriptorType type)
{
    VkDescriptorSetLayoutBinding new_bind = {};
    new_bind.binding = binding;
    new_bind.descriptorCount = 1;
    new_bind.descriptorType = type;

    bindings.push_back(new_bind);
}

void DescriptorLayoutBuilder::clear()
{
    bindings.clear();
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device, VkShaderStageFlags shader_stages, void* pNext, VkDescriptorSetLayoutCreateFlags flags)
{
    for(VkDescriptorSetLayoutBinding& b : bindings)
    {
        b.stageFlags |= shader_stages;
    }

    VkDescriptorSetLayoutCreateInfo info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.pNext = pNext;
    info.pBindings = bindings.data();
    info.bindingCount = (uint32_t)bindings.size();
    info.flags = flags;

    VkDescriptorSetLayout set;
    if(vkCreateDescriptorSetLayout(device, &info, nullptr, &set) != VK_SUCCESS)
    {
        std::cerr << "Failed to create descriptor layout" << std::endl;
        exit(-1);
    }

    return set;
}

void DescriptorAllocator::init_pool(VkDevice device, uint32_t max_sets, std::span<PoolSizeRatio> pool_ratios)
{
    std::vector<VkDescriptorPoolSize> pool_sizes;
    for(PoolSizeRatio ratio : pool_ratios)
    {
        pool_sizes.push_back(VkDescriptorPoolSize{
            .type = ratio.type,
            .descriptorCount = (uint32_t)(ratio.ratio * max_sets)
        });
    }

    VkDescriptorPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.flags = 0;
    pool_info.maxSets = max_sets;
    pool_info.poolSizeCount = (uint32_t)pool_sizes.size();
    pool_info.pPoolSizes = pool_sizes.data();

    vkCreateDescriptorPool(device, &pool_info, nullptr, &pool);
}

void DescriptorAllocator::clear_descriptors(VkDevice device)
{
    vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroy_pool(VkDevice device)
{
    vkDestroyDescriptorPool(device, pool, nullptr);
}

VkDescriptorSet DescriptorAllocator::allocate(VkDevice device, VkDescriptorSetLayout layout)
{
    VkDescriptorSetAllocateInfo alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc_info.pNext = nullptr;
    alloc_info.descriptorPool = pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout;

    VkDescriptorSet set;
    if(vkAllocateDescriptorSets(device, &alloc_info, &set) != VK_SUCCESS)
    {
        std::cerr << "Failed to create descriptor set" << std::endl;
        exit(-1);
    }

    return set;
}

void init_descriptors(VulkanContext& context)
{
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = 
    {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}
    };

    context.descriptor_allocator.init_pool(context.device, 10, sizes);

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        context.draw_image_descriptor_layout = builder.build(context.device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    context.draw_image_descriptors = context.descriptor_allocator.allocate(context.device, context.draw_image_descriptor_layout);

    VkDescriptorImageInfo img_info = {};
    img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    img_info.imageView = context.draw_image.image_view;
    
    VkWriteDescriptorSet draw_image_write = {};
    draw_image_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    draw_image_write.pNext = nullptr;
    draw_image_write.dstBinding = 0;
    draw_image_write.dstSet = context.draw_image_descriptors;
    draw_image_write.descriptorCount = 1;
    draw_image_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    draw_image_write.pImageInfo = &img_info;

    vkUpdateDescriptorSets(context.device, 1, &draw_image_write, 0, nullptr);
}

bool load_shader_module(const char* path, VkDevice device, VkShaderModule* out_module)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if(!file.is_open())
    {
        std::cerr << "Failed to open file" << std::endl;
        exit(-1);
    }

    size_t file_size = (size_t)file.tellg();

    std::vector<uint32_t> buffer(file_size / sizeof(uint32_t));

    file.seekg(0);

    file.read((char*)buffer.data(), file_size);

    file.close();

    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.pNext = nullptr;

    create_info.codeSize = buffer.size() * sizeof(uint32_t);
    create_info.pCode = buffer.data();

    VkShaderModule shader_module;
    if(vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS)
    {
        std::cerr << "Failed to create shader module" << std::endl;
        exit(-1);
    }

    *out_module = shader_module;

    return true;
}

void init_pipelines(VulkanContext& context)
{
    init_background_pipelines(context);
    std::cout << "Compute pipeline created" << std::endl;
}


void init_background_pipelines(VulkanContext& context)
{
    VkPipelineLayoutCreateInfo compute_layout = {};
    compute_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    compute_layout.pNext= nullptr;
    compute_layout.pSetLayouts = &context.draw_image_descriptor_layout;
    compute_layout.setLayoutCount = 1;

    if(vkCreatePipelineLayout(context.device, &compute_layout, nullptr, &context.gradient_pipeline_layout) != VK_SUCCESS)
    {
        std::cerr << "Failed to create pipeline layout" << std::endl;
        exit(-1);
    }

    VkShaderModule compute_draw_shader;
    if(!load_shader_module("../gradient.spv", context.device, &compute_draw_shader))
    {
        std::cerr << "Failed to create shader module" << std::endl;
        exit(-1);
    }

    VkPipelineShaderStageCreateInfo stage_info = {};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.pNext = nullptr;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = compute_draw_shader;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo compute_pipeline_create_info = {};
    compute_pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_pipeline_create_info.pNext = nullptr;
    compute_pipeline_create_info.layout = context.gradient_pipeline_layout;
    compute_pipeline_create_info.stage = stage_info;
    
    if(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &compute_pipeline_create_info, nullptr, &context.gradient_pipeline) != VK_SUCCESS)
    {
        std::cerr << "Failed to create pipeline" << std::endl;
        exit(-1);
    }

    vkDestroyShaderModule(context.device, compute_draw_shader, nullptr);
}