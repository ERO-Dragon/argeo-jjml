#ifndef _jjml_llama_h
#define _jjml_llama_h

#include <vector>

#include <jni.h>
#include <llama.h>

/*
 * SHARED UTILITIES
 */

/**
 * @brief Adds token to a batch.
 *
 * @param batch
 * @param id
 * @param pos
 * @param seq_ids
 * @param logits
 */
void jjml_llm_batch_add(struct llama_batch &batch, llama_token id,
		llama_pos pos, const std::vector<llama_seq_id> &seq_ids, bool logits);

/**
 * @brief Clears a batch.
 *
 * @param batch the batch to clear
 */
void jjml_llm_batch_clear(struct llama_batch &batch);

/**
 * @brief Converts a Java GGML type code to a llama.cpp KV cache type.
 *
 * Only GGML types supported by llama.cpp KV cache parameters are accepted.
 *
 * @param value the numeric GGML type code
 * @param param_name the parameter name to include in validation errors
 */
ggml_type jjml_llm_kv_cache_type_from_int(int value, const char *param_name);

/**
 * @brief Converts a Java flash-attention type code to llama.cpp enum.
 *
 * @param value the numeric llama_flash_attn_type code
 */
llama_flash_attn_type jjml_llm_flash_attn_type_from_int(int value);

/**
 * @brief Creates a Java LlamaCppDevice value for a ggml backend device.
 *
 * The returned local reference is owned by the caller.
 */
jobject jjml_llm_new_device(JNIEnv *env, ggml_backend_dev_t device);

#endif
