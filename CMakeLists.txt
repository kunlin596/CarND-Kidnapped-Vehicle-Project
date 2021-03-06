project(PARTICLE_FILTER)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

file(GLOB HEADERS src/*.h)
file(GLOB HEADERS_HPP src/*.hpp)

set(sources src/particle_filter.cpp src/main.cpp ${HEADERS} ${HEADERS_HPP})



if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1*/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS log)
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    set(OpenMP_LIBS OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP Libs ${OpenMP_LIBS}")
endif()

add_executable(particle_filter ${sources})

target_link_libraries(particle_filter z ssl uv uWS Eigen3::Eigen Boost::log ${OpenMP_LIBS})

