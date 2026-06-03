set(TARGET Java_org_argeo_jjml_mtmd)

# 收集 mtmd 根目录源码，排除不需要的 cli 和 helper
file(GLOB MTMD_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../tp/llama.cpp/tools/mtmd/*.cpp)
list(FILTER MTMD_SOURCES EXCLUDE REGEX "mtmd-cli\\.cpp$")
list(FILTER MTMD_SOURCES EXCLUDE REGEX "mtmd-helper\\.cpp$")

# 收集 models 子目录下的源码（实现了各种 clip_graph）
file(GLOB MTMD_MODEL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../tp/llama.cpp/tools/mtmd/models/*.cpp)

add_library(${TARGET} SHARED
        jjml_mtmd.cpp
        jjml_mtmd_utils.cpp
        ${MTMD_SOURCES}
        ${MTMD_MODEL_SOURCES}
)

if(ggml_FOUND)
target_link_libraries(${TARGET} PRIVATE ggml::ggml ggml::ggml-base llama whisper)
else()
target_link_libraries(${TARGET} PRIVATE ggml ggml-base llama whisper)
endif()

a2_jni_target(${TARGET})

# 补充头文件搜索路径
target_include_directories(Java_org_argeo_jjml_mtmd PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../tp/llama.cpp/tools/mtmd
    ${CMAKE_CURRENT_SOURCE_DIR}/../tp/llama.cpp/tools/mtmd/models
    ${CMAKE_CURRENT_SOURCE_DIR}/../tp/llama.cpp/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../tp/llama.cpp/vendor
)
