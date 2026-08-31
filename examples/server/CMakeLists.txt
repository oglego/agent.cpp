cmake_minimum_required(VERSION 3.14)
project(server-example VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_BINARY_DIR}/agent-cpp)

add_executable(server-example server.cpp)

target_include_directories(server-example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared
    ${CMAKE_CURRENT_SOURCE_DIR}/../../deps/cpp-httplib
    ${LLAMA_SOURCE_DIR}/common
    ${LLAMA_SOURCE_DIR}/ggml/include
    ${LLAMA_SOURCE_DIR}/include
    ${LLAMA_SOURCE_DIR}/vendor
)

target_link_libraries(server-example PRIVATE agent-cpp::agent ${LLAMA_COMMON_TARGET} llama)
target_compile_features(server-example PRIVATE cxx_std_17)

find_package(Threads REQUIRED)
target_link_libraries(server-example PRIVATE Threads::Threads)

message(STATUS "Minimal agent server example configured.")
