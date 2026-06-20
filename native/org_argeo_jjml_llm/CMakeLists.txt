set(TARGET Java_org_argeo_jjml_llm)

add_library(${TARGET} SHARED 
	org_argeo_jjml_llm_.cpp
	jjml_llm.cpp
	jjml_llm_vocabulary.cpp
	jjml_llm_model.cpp
	jjml_llm_chat.cpp
	jjml_llm_context.cpp
	jjml_llm_sampling.cpp
	jjml_llm_batch_write.cpp
	jjml_llm_batch_read.cpp
	jjml_llm_speculative.cpp
	jjml_llm_embedding.cpp
)

if(TARGET llama-common)
set(LLAMA_COMMON_TARGET llama-common)
elseif(TARGET common)
set(LLAMA_COMMON_TARGET common)
else()
set(LLAMA_COMMON_TARGET common)
endif()

if(ggml_FOUND)
target_link_libraries(${TARGET} PRIVATE	ggml::ggml ggml::ggml-base llama ${LLAMA_COMMON_TARGET})
else()
target_link_libraries(${TARGET} PRIVATE	ggml ggml-base llama ${LLAMA_COMMON_TARGET})
endif()

# Include llama.cpp common/ directory for Jinja2 chat template support
target_include_directories(${TARGET} PRIVATE
	${CMAKE_SOURCE_DIR}/native/tp/llama.cpp/common
	${CMAKE_SOURCE_DIR}/native/tp/llama.cpp/vendor
)

# Jinja2 headers contain UTF-8 characters (e.g. ↵), MSVC needs /utf-8
if(MSVC)
target_compile_options(${TARGET} PRIVATE /utf-8)
endif()

# When breaking changes are not functionally critical
# we keep building against older, in order to extend the life
# of a given JJML branch. To be trimmed when a new branch
# is created.

# for example:
#if(LLAMA_BUILD_NUMBER LESS 6325)
## Breaks flash attention context parameters
#target_compile_definitions(${TARGET} PRIVATE JJML_PRE_LLAMA_0_0_6325)
#endif()

a2_jni_target(${TARGET})
