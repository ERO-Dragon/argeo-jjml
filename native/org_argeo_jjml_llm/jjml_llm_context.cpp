#include <cassert>
#include <stdexcept>
#include <string>
#include <thread>
#include <iostream>

#include <llama.h>

#include <argeo/jni/argeo_jni.h>

#include "org_argeo_jjml_llm_LlamaCppBackend.h" // IWYU pragma: keep
#include "org_argeo_jjml_llm_LlamaCppContext.h" // IWYU pragma: keep

#include "org_argeo_jjml_llm_.h"

static struct ggml_threadpool *threadpool = NULL;

static llama_context_type jjml_llm_context_type_from_int(jint value) {
	switch (value) {
	case LLAMA_CONTEXT_TYPE_DEFAULT:
		return LLAMA_CONTEXT_TYPE_DEFAULT;
	case LLAMA_CONTEXT_TYPE_MTP:
		return LLAMA_CONTEXT_TYPE_MTP;
	default:
		throw std::invalid_argument(
				"Invalid llama context type value: " + std::to_string(value));
	}
}

static llama_rope_scaling_type jjml_llm_rope_scaling_type_from_int(
		jint value) {
	switch (value) {
	case LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED:
		return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
	case LLAMA_ROPE_SCALING_TYPE_NONE:
		return LLAMA_ROPE_SCALING_TYPE_NONE;
	case LLAMA_ROPE_SCALING_TYPE_LINEAR:
		return LLAMA_ROPE_SCALING_TYPE_LINEAR;
	case LLAMA_ROPE_SCALING_TYPE_YARN:
		return LLAMA_ROPE_SCALING_TYPE_YARN;
	case LLAMA_ROPE_SCALING_TYPE_LONGROPE:
		return LLAMA_ROPE_SCALING_TYPE_LONGROPE;
	default:
		throw std::invalid_argument(
				"Invalid llama rope scaling type value: "
						+ std::to_string(value));
	}
}

static enum llama_pooling_type jjml_llm_pooling_type_from_int(jint value) {
	switch (value) {
	case LLAMA_POOLING_TYPE_UNSPECIFIED:
		return LLAMA_POOLING_TYPE_UNSPECIFIED;
	case LLAMA_POOLING_TYPE_NONE:
		return LLAMA_POOLING_TYPE_NONE;
	case LLAMA_POOLING_TYPE_MEAN:
		return LLAMA_POOLING_TYPE_MEAN;
	case LLAMA_POOLING_TYPE_CLS:
		return LLAMA_POOLING_TYPE_CLS;
	case LLAMA_POOLING_TYPE_LAST:
		return LLAMA_POOLING_TYPE_LAST;
	case LLAMA_POOLING_TYPE_RANK:
		return LLAMA_POOLING_TYPE_RANK;
	default:
		throw std::invalid_argument(
				"Invalid llama pooling type value: " + std::to_string(value));
	}
}

static llama_attention_type jjml_llm_attention_type_from_int(jint value) {
	switch (value) {
	case LLAMA_ATTENTION_TYPE_UNSPECIFIED:
		return LLAMA_ATTENTION_TYPE_UNSPECIFIED;
	case LLAMA_ATTENTION_TYPE_CAUSAL:
		return LLAMA_ATTENTION_TYPE_CAUSAL;
	case LLAMA_ATTENTION_TYPE_NON_CAUSAL:
		return LLAMA_ATTENTION_TYPE_NON_CAUSAL;
	default:
		throw std::invalid_argument(
				"Invalid llama attention type value: " + std::to_string(value));
	}
}

static llama_flash_attn_type jjml_llm_flash_attn_type_from_int(jint value) {
	switch (value) {
	case LLAMA_FLASH_ATTN_TYPE_AUTO:
		return LLAMA_FLASH_ATTN_TYPE_AUTO;
	case LLAMA_FLASH_ATTN_TYPE_DISABLED:
		return LLAMA_FLASH_ATTN_TYPE_DISABLED;
	case LLAMA_FLASH_ATTN_TYPE_ENABLED:
		return LLAMA_FLASH_ATTN_TYPE_ENABLED;
	default:
		throw std::invalid_argument(
				"Invalid llama flash attention type value: "
						+ std::to_string(value));
	}
}

static ggml_type jjml_llm_kv_cache_type_from_int(jint value,
		const char *param_name) {
	switch (value) {
	case GGML_TYPE_F32:
		return GGML_TYPE_F32;
	case GGML_TYPE_F16:
		return GGML_TYPE_F16;
	case GGML_TYPE_BF16:
		return GGML_TYPE_BF16;
	case GGML_TYPE_Q8_0:
		return GGML_TYPE_Q8_0;
	case GGML_TYPE_Q4_0:
		return GGML_TYPE_Q4_0;
	case GGML_TYPE_Q4_1:
		return GGML_TYPE_Q4_1;
	case GGML_TYPE_IQ4_NL:
		return GGML_TYPE_IQ4_NL;
	case GGML_TYPE_Q5_0:
		return GGML_TYPE_Q5_0;
	case GGML_TYPE_Q5_1:
		return GGML_TYPE_Q5_1;
	default:
		throw std::invalid_argument(
				std::string("Unsupported ") + param_name
						+ " cache type value: " + std::to_string(value));
	}
}

/*
 * STATE
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppContext_doGetStateSize(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	return static_cast<jlong>(llama_state_get_size(ctx));
	//return llama_get_state_size(ctx);// deprecated
}

JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_llm_LlamaCppContext_doGetStateDataAsBytes(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	size_t size = llama_state_get_size(ctx);
	jbyteArray res = env->NewByteArray(size);
	void *dst = env->GetPrimitiveArrayCritical(res, NULL);
	size_t n_bytes = llama_state_get_data(ctx, static_cast<uint8_t*>(dst),
			size);
	env->ReleasePrimitiveArrayCritical(res, dst, 0);
	// TODO check n_bytes
	return res;
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppContext_doGetStateData(
		JNIEnv *env, jobject obj, jobject buf, jint offset) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);

	size_t size = llama_state_get_size(ctx);
	void *dst = env->GetDirectBufferAddress(buf);
	if (dst == NULL)
		throw std::invalid_argument("Input is not a direct buffer");
	assert(env->GetDirectBufferCapacity(buf) >= offset + size);
	size_t n_bytes = llama_state_get_data(ctx, static_cast<uint8_t*>(dst),
			size);
	return n_bytes;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llm_LlamaCppContext_doSetStateDataBytes(
		JNIEnv *env, jobject obj, jbyteArray arr, jint offset, jint length) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	void *src = env->GetPrimitiveArrayCritical(arr, NULL);
	size_t n_bytes = llama_state_set_data(ctx,
			static_cast<uint8_t*>(src) + offset, length);
	env->ReleasePrimitiveArrayCritical(arr, src, 0);
	// TODO check n_bytes
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llm_LlamaCppContext_doSetStateData(
		JNIEnv *env, jobject obj, jobject buf, jint offset, jint length) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);

	void *src = env->GetDirectBufferAddress(buf);
	if (src == NULL)
		throw std::invalid_argument("Input is not a direct buffer");
	size_t n_bytes = llama_state_set_data(ctx,
			static_cast<uint8_t*>(src) + offset, length);
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llm_LlamaCppContext_doSaveStateFile(
		JNIEnv *env, jobject obj, jbyteArray path, jobject buf, jint offset,
		jint length) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	std::string p = argeo::jni::to_string(env, path);

	void *tokens_arr = env->GetDirectBufferAddress(buf);
	if (tokens_arr == NULL)
		throw std::invalid_argument("Input is not a direct buffer");
	assert(env->GetDirectBufferCapacity(buf) //
	>= (offset + length) * sizeof(llama_token));

	auto *tokens = static_cast<const llama_token*>(tokens_arr) + offset;
	llama_state_save_file(ctx, p.c_str(), tokens, length);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppContext_doLoadStateFile(
		JNIEnv *env, jobject obj, jbyteArray path, jobject buf, jint offset) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	std::string p = argeo::jni::to_string(env, path);

	void *tokens_arr = env->GetDirectBufferAddress(buf);
	if (tokens_arr == NULL)
		throw std::invalid_argument("Input is not a direct buffer");

	size_t capacity = env->GetDirectBufferCapacity(buf) / sizeof(llama_token)
			- offset;
	auto *tokens = static_cast<llama_token*>(tokens_arr) + offset;
	size_t n_token_count;
	llama_state_load_file(ctx, p.c_str(), tokens, capacity, &n_token_count);
	return n_token_count;
}

/*
 * PARAMETERS
 */
/** @brief Get context parameters from Java to native.*/
static void get_context_params(JNIEnv *env, jobject params,
		llama_context_params *ctx_params) {
	jclass clss = env->FindClass(JCLASS_CONTEXT_PARAMS.c_str());
	// integers
	ctx_params->n_ctx = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_ctx", "()I"));
	ctx_params->n_batch = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_batch", "()I"));
	ctx_params->n_ubatch = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_ubatch", "()I"));
	ctx_params->n_seq_max = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_seq_max", "()I"));
	ctx_params->n_rs_seq = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_rs_seq", "()I"));
	ctx_params->n_outputs_max = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_outputs_max", "()I"));
	ctx_params->n_threads = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_threads", "()I"));
	ctx_params->n_threads_batch = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_threads_batch", "()I"));

// enums
	ctx_params->ctx_type = jjml_llm_context_type_from_int(env->CallIntMethod(
			params, env->GetMethodID(clss, "ctx_type", "()I")));
	ctx_params->rope_scaling_type = jjml_llm_rope_scaling_type_from_int(
			env->CallIntMethod(params,
					env->GetMethodID(clss, "rope_scaling_type", "()I")));
	ctx_params->pooling_type = jjml_llm_pooling_type_from_int(
			env->CallIntMethod(params,
					env->GetMethodID(clss, "pooling_type", "()I")));
	ctx_params->attention_type = jjml_llm_attention_type_from_int(
			env->CallIntMethod(params,
					env->GetMethodID(clss, "attention_type", "()I")));
	ctx_params->flash_attn_type = jjml_llm_flash_attn_type_from_int(
			env->CallIntMethod(params,
					env->GetMethodID(clss, "flash_attn_type", "()I")));

	ctx_params->rope_freq_base = env->CallFloatMethod(params,
			env->GetMethodID(clss, "rope_freq_base", "()F"));
	ctx_params->rope_freq_scale = env->CallFloatMethod(params,
			env->GetMethodID(clss, "rope_freq_scale", "()F"));
	ctx_params->yarn_ext_factor = env->CallFloatMethod(params,
			env->GetMethodID(clss, "yarn_ext_factor", "()F"));
	ctx_params->yarn_attn_factor = env->CallFloatMethod(params,
			env->GetMethodID(clss, "yarn_attn_factor", "()F"));
	ctx_params->yarn_beta_fast = env->CallFloatMethod(params,
			env->GetMethodID(clss, "yarn_beta_fast", "()F"));
	ctx_params->yarn_beta_slow = env->CallFloatMethod(params,
			env->GetMethodID(clss, "yarn_beta_slow", "()F"));
	ctx_params->yarn_orig_ctx = env->CallIntMethod(params,
			env->GetMethodID(clss, "yarn_orig_ctx", "()I"));
	ctx_params->defrag_thold = env->CallFloatMethod(params,
			env->GetMethodID(clss, "defrag_thold", "()F"));

	ctx_params->type_k = jjml_llm_kv_cache_type_from_int(
			env->CallIntMethod(params, env->GetMethodID(clss, "type_k", "()I")),
			"type_k");
	ctx_params->type_v = jjml_llm_kv_cache_type_from_int(
			env->CallIntMethod(params, env->GetMethodID(clss, "type_v", "()I")),
			"type_v");

	// booleans
	ctx_params->embeddings = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "embeddings", "()Z"));
	ctx_params->offload_kqv = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "offload_kqv", "()Z"));
	ctx_params->no_perf = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "no_perf", "()Z"));
	ctx_params->op_offload = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "op_offload", "()Z"));
	ctx_params->swa_full = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "swa_full", "()Z"));
	ctx_params->kv_unified = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "kv_unified", "()Z"));
}

JNIEXPORT jobject JNICALL Java_org_argeo_jjml_llm_LlamaCppBackend_newContextParams(
		JNIEnv *env, jclass) {
	llama_context_params ctx_params = llama_context_default_params();
	jobject res = env->NewObject(
			argeo::jni::find_jclass(env, JCLASS_CONTEXT_PARAMS), //
			ContextParams__init, //
			ctx_params.n_ctx, //
			ctx_params.n_batch, //
			ctx_params.n_ubatch, //
			ctx_params.n_seq_max, //
			ctx_params.n_rs_seq, //
			ctx_params.n_outputs_max, //
			ctx_params.n_threads, //
			ctx_params.n_threads_batch, //
			ctx_params.ctx_type, //
			ctx_params.rope_scaling_type, //
			ctx_params.pooling_type, //
			ctx_params.attention_type, //
			ctx_params.flash_attn_type, //
			ctx_params.rope_freq_base, //
			ctx_params.rope_freq_scale, //
			ctx_params.yarn_ext_factor, //
			ctx_params.yarn_attn_factor, //
			ctx_params.yarn_beta_fast, //
			ctx_params.yarn_beta_slow, //
			ctx_params.yarn_orig_ctx, //
			ctx_params.defrag_thold, //
			ctx_params.type_k, //
			ctx_params.type_v, //
			ctx_params.embeddings, //
			ctx_params.offload_kqv, //
			ctx_params.no_perf, //
			ctx_params.op_offload, //
			ctx_params.swa_full, //
			ctx_params.kv_unified //
			);
	return res;
}

/*
 * LIFECYCLE
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppContext_doInit(
		JNIEnv *env, jclass, jobject modelObj, jobject contextParams) {
	try {
		auto *model = argeo::jni::as_pointer<llama_model*>(env, modelObj);

		llama_context_params ctx_params = llama_context_default_params();
		get_context_params(env, contextParams, &ctx_params);

		llama_context *ctx = llama_init_from_model(model, ctx_params);
		if (ctx == NULL) {
			throw std::runtime_error("Failed to create llama.cpp context");
		}

		// Thread pool
		auto *reg = ggml_backend_dev_backend_reg(
				ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
		auto *ggml_threadpool_new_fn =
				(decltype(ggml_threadpool_new)*) ggml_backend_reg_get_proc_address(
						reg, "ggml_threadpool_new");
		auto *ggml_threadpool_free_fn =
				(decltype(ggml_threadpool_free)*) ggml_backend_reg_get_proc_address(
						reg, "ggml_threadpool_free");

		unsigned int n_threads_os = std::thread::hardware_concurrency();
//		struct ggml_threadpool_params tpp_batch;
//	    ggml_threadpool_params_init(&tpp_batch, n_threads);
		struct ggml_threadpool_params tpp;
		ggml_threadpool_params_init(&tpp, n_threads_os);

		//set_process_priority(params.cpuparams.priority);

//		struct ggml_threadpool *threadpool_batch = NULL;
//		if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
//			threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
//			if (!threadpool_batch) {
//				// FIXME throw exception
//			}
//
//			// Start the non-batch threadpool in the paused state
//			tpp.paused = true;
//		}

//		struct ggml_threadpool *threadpool = ggml_threadpool_new_fn(&tpp);
		if (!threadpool) {
			threadpool = ggml_threadpool_new_fn(&tpp);
			if (!threadpool) {
				// FIXME throw exception
			}
		}

		llama_attach_threadpool(ctx, threadpool, NULL);

		return (jlong) ctx;
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
		return 0;
	}
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llm_LlamaCppContext_doDestroy(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);

	llama_detach_threadpool(ctx);
	llama_free(ctx);
}

/*
 * ACCESSORS
 */
JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppContext_doGetPoolingType(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	return llama_pooling_type(ctx);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppContext_doGetContextSize(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	return llama_n_ctx(ctx);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppContext_doGetBatchSize(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	return llama_n_batch(ctx);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppContext_doGetPhysicalBatchSize(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	return llama_n_ubatch(ctx);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppContext_doGetMaxSequenceCount(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	return llama_n_seq_max(ctx);
}
