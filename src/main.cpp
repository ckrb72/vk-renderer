#include <iostream>
#include <fstream>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>
#include <vector>
#include <span>
#include <array>

#define VMA_IMPLEMENTATION
#include "vma.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"

#include <glm/glm.hpp>

const int WIN_WIDTH = 1920;
const int WIN_HEIGHT = 1080;

constexpr uint32_t FRAME_OVERLAP = 2;

struct AllocatedBuffer
{
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
};

struct Vertex
{
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
};

struct GPUMeshBuffers
{
    AllocatedBuffer index_buffer;
    AllocatedBuffer vertex_buffer;
    VkDeviceAddress vertex_buffer_address;
};

struct GPUDrawPushConstants
{
    glm::mat4 world_matrix;
    VkDeviceAddress vertex_buffer;
};

struct AllocatedImage
{
    VkImage image;
    VkImageView image_view;
    VmaAllocation allocation;
    VkExtent3D extent;
    VkFormat format;
};

struct ComputePushConstants
{
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
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

class PipelineBuilder
{
public:
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;

    VkPipelineInputAssemblyStateCreateInfo input_assembly;
    VkPipelineRasterizationStateCreateInfo rasterizer;
    VkPipelineColorBlendAttachmentState color_blend_attachment;
    VkPipelineMultisampleStateCreateInfo multisampling;
    VkPipelineLayout pipeline_layout;
    VkPipelineDepthStencilStateCreateInfo depth_stencil;
    VkPipelineRenderingCreateInfo render_info;
    VkFormat color_attachment_format;

    PipelineBuilder(){ clear(); }

    void clear();
    VkPipeline build_pipeline(VkDevice device);
    void set_shaders(VkShaderModule vertex_shader, VkShaderModule fragment_shader);
    void set_input_topology(VkPrimitiveTopology topology);
    void set_polygon_mode(VkPolygonMode mode);
    void set_cull_mode(VkCullModeFlags cull_mode, VkFrontFace front_face);
    void set_multisampling_none();
    void disable_blending();
    void set_color_attachment_format(VkFormat format);
    void set_depth_format(VkFormat format);
    void disable_depthtest();
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

    VkPipelineLayout triangle_pipeline_layout;
    VkPipeline triangle_pipeline;

    struct ImGuiHandle
    {
        VkDescriptorPool descriptor_pool;
    };

    ImGuiHandle imgui;

    struct ImmediateSubmit
    {
        VkFence fence;
        VkCommandBuffer command_buffer;
        VkCommandPool command_pool;
    };

    ImmediateSubmit immediate_submit;

    VkPipelineLayout mesh_pipeline_layout;
    VkPipeline mesh_pipeline;

    GPUMeshBuffers rectangle;
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
void init_imgui(VulkanContext& context, GLFWwindow* window);
void destroy_imgui(VulkanContext& context);
void draw_imgui(VulkanContext& context, VkCommandBuffer cmd, VkImageView target_image_view);
void init_triangle_pipeline(VulkanContext& context);
VkRenderingAttachmentInfo attachment_info(VkImageView view, VkClearValue* clear, VkImageLayout layout);
AllocatedBuffer create_buffer(VulkanContext& context, size_t alloc_size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage);
void destroy_buffer(VulkanContext& context, const AllocatedBuffer& buffer);
void immediate_submit(VulkanContext& context, std::function<void(VkCommandBuffer cmd)>&& function);
void init_mesh_pipeline(VulkanContext& context);
GPUMeshBuffers upload_mesh(VulkanContext& context, std::span<uint32_t> indices, std::span<Vertex> vertices);

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
    init_imgui(context, window);

    std::array<Vertex, 4> rect_vertices;
	rect_vertices[0].position = {0.5,-0.5, 0};
	rect_vertices[1].position = {0.5,0.5, 0};
	rect_vertices[2].position = {-0.5,-0.5, 0};
	rect_vertices[3].position = {-0.5,0.5, 0};

	rect_vertices[0].color = {0,0, 0,1};
	rect_vertices[1].color = { 0.5,0.5,0.5 ,1};
	rect_vertices[2].color = { 1,0, 0,1 };
	rect_vertices[3].color = { 0,1, 0,1 };

    std::array<uint32_t, 6> rect_indices;
    rect_indices[0] = 0;
	rect_indices[1] = 1;
	rect_indices[2] = 2;

	rect_indices[3] = 2;
	rect_indices[4] = 1;
	rect_indices[5] = 3;

    context.rectangle = upload_mesh(context, rect_indices, rect_vertices);

    uint32_t current_frame = 0;
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::ShowDemoWindow();
        ImGui::Render();    // Doesn't actually draw imgui, just calculates vertices

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
        
        ComputePushConstants pc;
        pc.data1 = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        pc.data2 = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);

        vkCmdPushConstants(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.gradient_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &pc);

        vkCmdDispatch(context.frames[current_frame % FRAME_OVERLAP].command_buffer, std::ceil(context.draw_extent.width / 16.0), std::ceil(context.draw_extent.height / 16.0), 1);

        transition_image(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.draw_image.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        
       VkRenderingAttachmentInfo color_attachment = attachment_info(context.draw_image.image_view, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

       VkRenderingInfo render_info = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr,
            .renderArea = VkRect2D{{0, 0}, {context.draw_extent.width, context.draw_extent.height}},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_attachment
        };
        vkCmdBeginRendering(context.frames[current_frame % FRAME_OVERLAP].command_buffer, &render_info);
        vkCmdBindPipeline(context.frames[current_frame % FRAME_OVERLAP].command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, context.triangle_pipeline);

        VkViewport viewport = {};
        viewport.x = 0;
        viewport.y = 0;
        viewport.width = context.draw_extent.width;
        viewport.height = context.draw_extent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        vkCmdSetViewport(context.frames[current_frame % FRAME_OVERLAP].command_buffer, 0, 1, &viewport);
        VkRect2D scissor = {};
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent.width = context.draw_extent.width;
        scissor.extent.height = context.draw_extent.height;

        vkCmdSetScissor(context.frames[current_frame % FRAME_OVERLAP].command_buffer, 0, 1, &scissor);
        vkCmdDraw(context.frames[current_frame % FRAME_OVERLAP].command_buffer, 3, 1, 0, 0);

        vkCmdBindPipeline(context.frames[current_frame % FRAME_OVERLAP].command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, context.mesh_pipeline);

        GPUDrawPushConstants push_constants;
        push_constants.world_matrix = glm::mat4(1.0f);
        push_constants.vertex_buffer = context.rectangle.vertex_buffer_address;

        vkCmdPushConstants(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.mesh_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &push_constants);
        vkCmdBindIndexBuffer(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.rectangle.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(context.frames[current_frame % FRAME_OVERLAP].command_buffer, 6, 1, 0, 0, 0);

        vkCmdEndRendering(context.frames[current_frame % FRAME_OVERLAP].command_buffer);

        transition_image(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.draw_image.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        
        transition_image(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.swapchain.images[swapchain_index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copy_image_to_image(context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.draw_image.image, context.swapchain.images[swapchain_index], context.draw_extent, context.swapchain.extent);
        
        draw_imgui(context, context.frames[current_frame % FRAME_OVERLAP].command_buffer, context.swapchain.image_views[swapchain_index]);

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

    destroy_imgui(context);

    destroy_buffer(context, context.rectangle.vertex_buffer);
    destroy_buffer(context, context.rectangle.index_buffer);

    vkDestroyPipelineLayout(context.device, context.mesh_pipeline_layout, nullptr);
    vkDestroyPipeline(context.device, context.mesh_pipeline, nullptr);

    vkDestroyPipelineLayout(context.device, context.gradient_pipeline_layout, nullptr);
    vkDestroyPipeline(context.device, context.gradient_pipeline, nullptr);

    vkDestroyPipelineLayout(context.device, context.triangle_pipeline_layout, nullptr);
    vkDestroyPipeline(context.device, context.triangle_pipeline, nullptr);

    context.descriptor_allocator.destroy_pool(context.device);
    vkDestroyDescriptorSetLayout(context.device, context.draw_image_descriptor_layout, nullptr);

    for(int i = 0; i < FRAME_OVERLAP; i++)
    {
        vkDestroyCommandPool(context.device, context.frames[i].command_pool, NULL);

        vkDestroyFence(context.device, context.frames[i].render_fence, NULL);
        vkDestroySemaphore(context.device, context.frames[i].render_semaphore, NULL);
        vkDestroySemaphore(context.device, context.frames[i].swapchain_semaphore, NULL);
    }

    vkDestroyCommandPool(context.device, context.immediate_submit.command_pool, nullptr);
    vkDestroyFence(context.device, context.immediate_submit.fence, nullptr);

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

    if(vkCreateCommandPool(context.device, &command_pool_info, nullptr, &context.immediate_submit.command_pool) != VK_SUCCESS)
    {
        std::cerr << "Failed to create immediate submit command pool" << std::endl;
        exit(-1);
    }

    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = context.immediate_submit.command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    if(vkAllocateCommandBuffers(context.device, &alloc_info, &context.immediate_submit.command_buffer) != VK_SUCCESS)
    {
        std::cerr << "Failed to create immediate command buffer" << std::endl;
        exit(-1);
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

    vkCreateFence(context.device, &fence_info, nullptr, &context.immediate_submit.fence);
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
        std::cerr << "Failed to open file: " << path << std::endl;
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
    init_triangle_pipeline(context);
    init_mesh_pipeline(context);
    std::cout << "All pipelines created" << std::endl;
}


void init_background_pipelines(VulkanContext& context)
{
    VkPipelineLayoutCreateInfo compute_layout = {};
    compute_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    compute_layout.pNext= nullptr;
    compute_layout.pSetLayouts = &context.draw_image_descriptor_layout;
    compute_layout.setLayoutCount = 1;

    VkPushConstantRange push_constant = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(ComputePushConstants)
    };
    
    compute_layout.pPushConstantRanges = &push_constant;
    compute_layout.pushConstantRangeCount = 1;

    if(vkCreatePipelineLayout(context.device, &compute_layout, nullptr, &context.gradient_pipeline_layout) != VK_SUCCESS)
    {
        std::cerr << "Failed to create pipeline layout" << std::endl;
        exit(-1);
    }

    VkShaderModule compute_draw_shader;
    if(!load_shader_module("../gradient_color.spv", context.device, &compute_draw_shader))
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

void init_imgui(VulkanContext& context, GLFWwindow* window)
{
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1000,
        .poolSizeCount = (uint32_t)std::size(pool_sizes),
        .pPoolSizes = pool_sizes
    };

    if(vkCreateDescriptorPool(context.device, &pool_info, nullptr, &context.imgui.descriptor_pool) != VK_SUCCESS)
    {
        std::cerr << "Failed to create imgui descriptor pool" << std::endl;
        exit(-1);
    }


    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = context.instance;
    init_info.PhysicalDevice = context.gpu;
    init_info.Device = context.device;
    init_info.QueueFamily = context.graphics_queue_family;
    init_info.Queue = context.graphics_queue;
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.Allocator = nullptr;
    init_info.DescriptorPool = context.imgui.descriptor_pool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;

    init_info.PipelineRenderingCreateInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &context.swapchain.format;

    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info);
    ImGui_ImplVulkan_CreateFontsTexture();

}

void destroy_imgui(VulkanContext& context)
{
    ImGui_ImplVulkan_Shutdown();
    vkDestroyDescriptorPool(context.device, context.imgui.descriptor_pool, nullptr);
}

VkRenderingAttachmentInfo attachment_info(VkImageView view, VkClearValue* clear, VkImageLayout layout)
{
    VkRenderingAttachmentInfo color_attachment = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = view,
        .imageLayout = layout,
        .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE
    };

    if(clear)
        color_attachment.clearValue = *clear;

    
    return color_attachment;
}

void draw_imgui(VulkanContext& context, VkCommandBuffer cmd, VkImageView target_image_view)
{
    VkRenderingAttachmentInfo color_attachment = attachment_info(target_image_view, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo render_info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr,
        .renderArea = VkRect2D{{0, 0}, {context.draw_extent.width, context.draw_extent.height}},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment
    };

    vkCmdBeginRendering(cmd, &render_info);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
}

void PipelineBuilder::clear()
{
    input_assembly = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    rasterizer = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    color_blend_attachment = {};
    multisampling = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    pipeline_layout = {};
    depth_stencil = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    render_info = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    shader_stages.clear();
}

VkPipeline PipelineBuilder::build_pipeline(VkDevice device)
{
    VkPipelineViewportStateCreateInfo viewport_state = {};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.pNext = nullptr;

    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineColorBlendStateCreateInfo color_blending = {};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.pNext = nullptr;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &color_blend_attachment;

    VkPipelineVertexInputStateCreateInfo vertex_input_info = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};


    VkGraphicsPipelineCreateInfo pipeline_info = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline_info.pNext = &render_info;

    pipeline_info.stageCount = (uint32_t)shader_stages.size();
    pipeline_info.pStages = shader_stages.data();
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.layout = pipeline_layout;

    VkDynamicState state[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_info = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic_info.pDynamicStates = &state[0];
    dynamic_info.dynamicStateCount = 2;

    pipeline_info.pDynamicState = &dynamic_info;

    VkPipeline pipeline;
    if(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS)
    {
        std::cout << "Failed to create graphics pipeline" << std::endl;
        exit(-1);
    }

    return pipeline;
}

void PipelineBuilder::set_shaders(VkShaderModule vertex_shader, VkShaderModule fragment_shader)
{
    shader_stages.clear();
    VkPipelineShaderStageCreateInfo vertex_shader_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertex_shader,
        .pName = "main"
    };
    shader_stages.push_back(vertex_shader_info);

    VkPipelineShaderStageCreateInfo fragment_shader_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fragment_shader,
        .pName = "main"
    };

    shader_stages.push_back(fragment_shader_info);
}

void PipelineBuilder::set_input_topology(VkPrimitiveTopology topology)
{
    input_assembly.topology = topology;
    input_assembly.primitiveRestartEnable = VK_FALSE;
}

void PipelineBuilder::set_polygon_mode(VkPolygonMode mode)
{
    rasterizer.polygonMode = mode;
    rasterizer.lineWidth = 1.0f;
}

void PipelineBuilder::set_cull_mode(VkCullModeFlags cull_mode, VkFrontFace front_face)
{
    rasterizer.cullMode = cull_mode;
    rasterizer.frontFace = front_face;
}

void PipelineBuilder::set_multisampling_none()
{
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;
}

void PipelineBuilder::disable_blending()
{
    color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.blendEnable = VK_FALSE;
}

void PipelineBuilder::set_color_attachment_format(VkFormat format)
{
    color_attachment_format = format;
    render_info.colorAttachmentCount = 1;
    render_info.pColorAttachmentFormats = &color_attachment_format;
}

void PipelineBuilder::set_depth_format(VkFormat format)
{
    render_info.depthAttachmentFormat = format;
}

void PipelineBuilder::disable_depthtest()
{
    depth_stencil.depthTestEnable = VK_FALSE;
    depth_stencil.depthWriteEnable = VK_FALSE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_NEVER;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;
    depth_stencil.front = {};
    depth_stencil.back = {};
    depth_stencil.minDepthBounds = 0.0f;
    depth_stencil.maxDepthBounds = 1.0f;
}

void init_triangle_pipeline(VulkanContext& context)
{
    VkShaderModule triangle_frag_shader;
    if(!load_shader_module("../colored_triangle.frag.spv", context.device, &triangle_frag_shader))
    {
        std::cerr << "Failed to create triangle fragment shader module" << std::endl;
        exit(-1);
    }

    VkShaderModule triangle_vertex_shader;
    if(!load_shader_module("../colored_triangle.vert.spv", context.device, &triangle_vertex_shader))
    {
        std::cerr << "Failed to create tirnagle vertex shader module" << std::endl;
        exit(-1);
    }

    VkPipelineLayoutCreateInfo graphics_layout = {};
    graphics_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    if(vkCreatePipelineLayout(context.device, &graphics_layout, nullptr, &context.triangle_pipeline_layout) != VK_SUCCESS)
    {
        std::cerr << "Failed to create triangle pipeline layout" << std::endl;
        exit(-1);
    }

    PipelineBuilder builder;
    builder.pipeline_layout = context.triangle_pipeline_layout;
    builder.set_shaders(triangle_vertex_shader, triangle_frag_shader);
    builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    builder.set_multisampling_none();
    builder.disable_blending();
    builder.disable_depthtest();

    builder.set_color_attachment_format(context.draw_image.format);
    builder.set_depth_format(VK_FORMAT_UNDEFINED);

    context.triangle_pipeline = builder.build_pipeline(context.device);

    vkDestroyShaderModule(context.device, triangle_vertex_shader, nullptr);
    vkDestroyShaderModule(context.device, triangle_frag_shader, nullptr);
}

void init_mesh_pipeline(VulkanContext& context)
{
    VkShaderModule mesh_vert_shader;
    if(!load_shader_module("../colored_triangle_mesh.vert.spv", context.device, &mesh_vert_shader))
    {
        std::cerr << "Failed to load mesh vertex shader" << std::endl;
        exit(-1);
    }

    VkShaderModule mesh_frag_shader;
    if(!load_shader_module("../colored_triangle.frag.spv", context.device, &mesh_frag_shader))
    {
        std::cerr << "Failed to load mesh fragment shader" << std::endl;
        exit(-1);
    }

    VkPushConstantRange buffer_range = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(GPUDrawPushConstants)
    };

    VkPipelineLayoutCreateInfo graphics_layout = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &buffer_range
    };

    if(vkCreatePipelineLayout(context.device, &graphics_layout, nullptr, &context.mesh_pipeline_layout) != VK_SUCCESS)
    {
        std::cerr << "Failed to create mesh graphics pipeline" << std::endl;
        exit(-1);
    }

    PipelineBuilder builder;
    builder.pipeline_layout = context.mesh_pipeline_layout;
    builder.set_shaders(mesh_vert_shader, mesh_frag_shader);
    builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    builder.set_multisampling_none();
    builder.disable_blending();
    builder.disable_depthtest();
    builder.set_color_attachment_format(context.draw_image.format);
    builder.set_depth_format(VK_FORMAT_UNDEFINED);

    context.mesh_pipeline = builder.build_pipeline(context.device);

    vkDestroyShaderModule(context.device, mesh_vert_shader, nullptr);
    vkDestroyShaderModule(context.device, mesh_frag_shader, nullptr);

}

AllocatedBuffer create_buffer(VulkanContext& context, size_t alloc_size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage)
{
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .size = alloc_size,
        .usage = usage
    };

    VmaAllocationCreateInfo vma_info = {
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = memory_usage
    };

    AllocatedBuffer new_buffer;

    if(vmaCreateBuffer(context.allocator, &buffer_info, &vma_info, &new_buffer.buffer, &new_buffer.allocation, &new_buffer.info) != VK_SUCCESS)
    {
        std::cerr << "Failed to create buffer" << std::endl;
        exit(-1);
    }

    return new_buffer;
}

void destroy_buffer(VulkanContext& context, const AllocatedBuffer& buffer)
{
    vmaDestroyBuffer(context.allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers upload_mesh(VulkanContext& context, std::span<uint32_t> indices, std::span<Vertex> vertices)
{
    const size_t vertex_buf_size = vertices.size() * sizeof(Vertex);
    const size_t index_buf_size = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers new_surface;

    new_surface.vertex_buffer = create_buffer(context, vertex_buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    
    VkBufferDeviceAddressInfo device_address_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = new_surface.vertex_buffer.buffer
    };
    new_surface.vertex_buffer_address = vkGetBufferDeviceAddress(context.device, &device_address_info);

    new_surface.index_buffer = create_buffer(context, index_buf_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);


    // Create a staging buffer that send the data to the GPU buffers we just made
    AllocatedBuffer staging_buffer = create_buffer(context, vertex_buf_size + index_buf_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    void* data = staging_buffer.allocation->GetMappedData();

    memcpy(data, vertices.data(), vertex_buf_size);
    memcpy((char*)data + vertex_buf_size, indices.data(), index_buf_size);

    //TODO: IMMEDIATELY SUBMIT THE DATA TO THE GPU (HAVEN'T IMPLEMENTED THIS YET)
    immediate_submit(context, [&](VkCommandBuffer cmd) {
        VkBufferCopy vertex_copy = {
            .srcOffset = 0,
            .dstOffset = 0,
            .size = vertex_buf_size
        };
        vkCmdCopyBuffer(cmd, staging_buffer.buffer, new_surface.vertex_buffer.buffer, 1, &vertex_copy);

        VkBufferCopy index_copy = {
            .srcOffset = vertex_buf_size,
            .dstOffset = 0,
            .size = index_buf_size
        };
        vkCmdCopyBuffer(cmd, staging_buffer.buffer, new_surface.index_buffer.buffer, 1, &index_copy);
    });

    destroy_buffer(context, staging_buffer);

    return new_surface;
}

void immediate_submit(VulkanContext& context, std::function<void(VkCommandBuffer cmd)>&& function)
{
	vkResetFences(context.device, 1, &context.immediate_submit.fence);
	vkResetCommandBuffer(context.immediate_submit.command_buffer, 0);

	VkCommandBuffer cmd = context.immediate_submit.command_buffer;

    VkCommandBufferBeginInfo cmd_buffer_info = {};
    cmd_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmd_buffer_info.pNext = nullptr;
    cmd_buffer_info.pInheritanceInfo = nullptr;
    cmd_buffer_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(cmd, &cmd_buffer_info);

	function(cmd);

	vkEndCommandBuffer(cmd);

	VkCommandBufferSubmitInfo cmdinfo = command_buffer_submit_info(cmd);
	VkSubmitInfo2 submit = submit_info(&cmdinfo, nullptr, nullptr);

	// submit command buffer to the queue and execute it.
	//  _renderFence will now block until the graphic commands finish execution
	vkQueueSubmit2(context.graphics_queue, 1, &submit, context.immediate_submit.fence);

	vkWaitForFences(context.device, 1, &context.immediate_submit.fence, true, 9999999999);
}