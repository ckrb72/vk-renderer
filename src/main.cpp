#include <iostream>
#include <GLFW/glfw3.h>

const int WIN_WIDTH = 1920;
const int WIN_HEIGHT = 1080;

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

    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}