cmake_minimum_required(VERSION 3.13)
project(vk-renderer)
set(CMAKE_CXX_STANDARD 17)


find_package(Vulkan REQUIRED)

set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")

set(GLFW_BUILD_DOCS OFF BOOL "GLFW lib only")
set(GLFW_INSTALL OFF CACHE BOOL "GLFW lib only")

add_subdirectory(glfw)
add_subdirectory(vk-bootstrap)

add_compile_options(-Wno-nullability-completeness)

file(GLOB_RECURSE SRC_CXX_FILES "${SOURCE_DIR}/*.cpp")

add_executable(renderer ${SRC_CXX_FILES})
target_link_libraries(renderer ${Vulkan_LIBRARIES} glfw vk-bootstrap::vk-bootstrap)