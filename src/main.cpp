#include <iostream>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>
#include <vector>

const int WIN_WIDTH = 1920;
const int WIN_HEIGHT = 1080;

constexpr uint32_t FRAME_OVERLAP = 2;

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

    uint32_t current_frame = 0;
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        vkWaitForFences(context.device, 1, &context.frames[current_frame % FRAME_OVERLAP].render_fence, true, UINT64_MAX);
        vkResetFences(context.device, 1, &context.frames[current_frame % FRAME_OVERLAP].render_fence);

        uint32_t swapchain_index;
        vkAcquireNextImageKHR(context.device, context.swapchain.swapchain, UINT64_MAX, context.frames[current_frame % FRAME_OVERLAP].swapchain_semaphore, nullptr, &swapchain_index);
        vkResetCommandBuffer(context.frames[current_frame % FRAME_OVERLAP].command_buffer, 0);

        // Record Command Buffer
        VkCommandBufferBeginInfo cmd_buffer_info = {};
        cmd_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cmd_buffer_info.pNext = nullptr;
        cmd_buffer_info.pInheritanceInfo = nullptr;
        cmd_buffer_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(context.frames[current_frame % FRAME_OVERLAP].command_buffer, &cmd_buffer_info);

        transition_image(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.swapchain.images[swapchain_index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

        VkClearColorValue clear_value;
        float flash = std::abs(std::sin(current_frame / 120.0f));
        clear_value = { {0.0f, 0.0f, flash, 1.0f } };

        VkImageSubresourceRange clear_range = {};
        clear_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        clear_range.baseMipLevel = 0;
        clear_range.levelCount = VK_REMAINING_MIP_LEVELS;
        clear_range.baseArrayLayer = 0;
        clear_range.layerCount = VK_REMAINING_ARRAY_LAYERS;

        vkCmdClearColorImage(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.swapchain.images[swapchain_index], VK_IMAGE_LAYOUT_GENERAL, &clear_value, 1, &clear_range);

        transition_image(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.swapchain.images[swapchain_index], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
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


    for(int i = 0; i < FRAME_OVERLAP; i++)
    {
        vkDestroyCommandPool(context.device, context.frames[i].command_pool, NULL);

        vkDestroyFence(context.device, context.frames[i].render_fence, NULL);
        vkDestroySemaphore(context.device, context.frames[i].render_semaphore, NULL);
        vkDestroySemaphore(context.device, context.frames[i].swapchain_semaphore, NULL);
    }

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

void destroy_swapchain(VulkanContext& context)
{
    vkDestroySwapchainKHR(context.device, context.swapchain.swapchain, NULL);

    for(const VkImageView& view : context.swapchain.image_views)
    {
        vkDestroyImageView(context.device, view, NULL);
    }
}