#include <iostream>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>
#include <vector>

const int WIN_WIDTH = 1920;
const int WIN_HEIGHT = 1080;

struct VulkanContext
{
    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkPhysicalDevice gpu;
    VkDevice device;
    VkSurfaceKHR surface;

    struct VulkanSwapchain
    {
        VkSwapchainKHR swapchain;
        VkFormat format;
        std::vector<VkImage> images;
        std::vector<VkImageView> image_views;
        VkExtent2D extent;
    };

    VulkanSwapchain swapchain;

};

void init_vulkan(VulkanContext& context, GLFWwindow* window);
void create_swapchain(uint32_t width, uint32_t height, VulkanContext& context);
void destroy_swapchain(VulkanContext& context);

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

    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
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

void destroy_swapchain(VulkanContext& context)
{
    vkDestroySwapchainKHR(context.device, context.swapchain.swapchain, NULL);

    for(const VkImageView& view : context.swapchain.image_views)
    {
        vkDestroyImageView(context.device, view, NULL);
    }
}