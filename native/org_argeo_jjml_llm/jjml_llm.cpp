#include "jjml_llm.h"

#include <stddef.h>
#include <stdexcept>
#include <string>

#include <argeo/jni/argeo_jni.h>

#include "org_argeo_jjml_llm_.h"

/*
 * SHARED PLAIN C++ UTILITIES
 */
void jjml_llm_batch_add(struct llama_batch &batch, llama_token id,
		llama_pos pos, const std::vector<llama_seq_id> &seq_ids, bool logits) {
	batch.token[batch.n_tokens] = id;
	batch.pos[batch.n_tokens] = pos;
	batch.n_seq_id[batch.n_tokens] = seq_ids.size();
	for (size_t i = 0; i < seq_ids.size(); ++i) {
		batch.seq_id[batch.n_tokens][i] = seq_ids[i];
	}
	batch.logits[batch.n_tokens] = logits;

	batch.n_tokens++;
}

void jjml_llm_batch_clear(struct llama_batch &batch) {
	batch.n_tokens = 0;
}

ggml_type jjml_llm_kv_cache_type_from_int(int value,
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

llama_flash_attn_type jjml_llm_flash_attn_type_from_int(int value) {
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

jobject jjml_llm_new_device(JNIEnv *env, ggml_backend_dev_t device) {
	struct ggml_backend_dev_props props;
	ggml_backend_dev_get_props(device, &props);
	ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);

	jstring backend = env->NewStringUTF(
			reg != nullptr && ggml_backend_reg_name(reg) != nullptr ?
					ggml_backend_reg_name(reg) : "");
	jstring name = env->NewStringUTF(props.name != nullptr ? props.name : "");
	jstring description = props.description != nullptr ?
			env->NewStringUTF(props.description) : nullptr;
	jstring deviceId = props.device_id != nullptr ?
			env->NewStringUTF(props.device_id) : nullptr;

	jobject result = env->NewObject(argeo::jni::find_jclass(env, JCLASS_DEVICE),
			LlamaCppDevice__init, //
			backend, name, description, deviceId, //
			static_cast<jint>(props.type), //
			static_cast<jlong>(props.memory_free), //
			static_cast<jlong>(props.memory_total));

	env->DeleteLocalRef(backend);
	env->DeleteLocalRef(name);
	if (description != nullptr)
		env->DeleteLocalRef(description);
	if (deviceId != nullptr)
		env->DeleteLocalRef(deviceId);

	return result;
}
