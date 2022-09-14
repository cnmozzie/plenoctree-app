cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
if (POLICY CMP0069)
    cmake_policy(SET CMP0069 NEW)
endif (POLICY CMP0069)
if (POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif (POLICY CMP0072)

project(plenoctree-app)

enable_language(CUDA)
message(STATUS "CUDA enabled")
set( CMAKE_CUDA_STANDARD 14 )
set( CMAKE_CUDA_STANDARD_REQUIRED ON)
set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -g -Xcudafe \"--display_error_number --diag_suppress=3057 --diag_suppress=3058 --diag_suppress=3059 --diag_suppress=3060\" -lineinfo -arch=sm_61")

set( INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" "/usr/include/python3.6")

if( MSVC )
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT /GLT /Ox")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler=\"/MT\"" )
endif()

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories (${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(plenoctree-app plenoctree-app.cu)
target_link_libraries(plenoctree-app PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(plenoctree-app PRIVATE "${INCLUDE_DIR}")
set_property(TARGET plenoctree-app PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET plenoctree-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:plenoctree-app>)
endif (MSVC)