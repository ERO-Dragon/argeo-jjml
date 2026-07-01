#include <stddef.h>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <llama.h>

#include "../tp/llama.cpp/src/llama-ext.h"

#include <argeo/jni/argeo_jni.h>

#include "org_argeo_jjml_llm_LlamaCppModel.h" // IWYU pragma: keep
#include "org_argeo_jjml_llm_LlamaCppBackend.h" // IWYU pragma: keep

#include "jjml_llm.h"
#include "org_argeo_jjml_llm_.h"

// CONSTANTS
static const size_t META_BUFFER_SIZE = 1024;
static const size_t META_BIG_BUFFER_SIZE = 20480;

namespace {
struct jjml_llm_model_state {
	std::vector<float> tensor_split;
};

std::map<llama_model*, jjml_llm_model_state> model_states;
std::mutex model_states_mutex;
}

static llama_split_mode jjml_llm_split_mode_from_int(jint value) {
	switch (value) {
	case LLAMA_SPLIT_MODE_NONE:
		return LLAMA_SPLIT_MODE_NONE;
	case LLAMA_SPLIT_MODE_LAYER:
		return LLAMA_SPLIT_MODE_LAYER;
	case LLAMA_SPLIT_MODE_ROW:
		return LLAMA_SPLIT_MODE_ROW;
	case LLAMA_SPLIT_MODE_TENSOR:
		return LLAMA_SPLIT_MODE_TENSOR;
	default:
		throw std::invalid_argument(
				"Invalid llama split mode value: " + std::to_string(value));
	}
}

static std::vector<float> jjml_llm_parse_tensor_split(
		const std::string &tensor_split) {
	std::vector<float> result;
	if (tensor_split.empty())
		return result;

	std::string token;
	std::stringstream ss(tensor_split);
	while (std::getline(ss, token, ',')) {
		size_t slash = 0;
		while ((slash = token.find('/')) != std::string::npos) {
			if (slash > 0)
				result.push_back(std::stof(token.substr(0, slash)));
			token.erase(0, slash + 1);
		}
		if (!token.empty())
			result.push_back(std::stof(token));
	}

	if (result.size() >= llama_max_devices())
		throw std::invalid_argument(
				"Tensor split has " + std::to_string(result.size())
						+ " entries, but llama.cpp supports "
						+ std::to_string(llama_max_devices()) + " devices");
	return result;
}

static void jjml_llm_keep_model_state(llama_model *model,
		jjml_llm_model_state &&state) {
	if (!state.tensor_split.empty()) {
		std::lock_guard<std::mutex> lock(model_states_mutex);
		model_states.emplace(model, std::move(state));
	}
}

static void jjml_llm_drop_model_state(llama_model *model) {
	std::lock_guard<std::mutex> lock(model_states_mutex);
	model_states.erase(model);
}

/*
 * PARAMETERS
 */
/** @brief Get model parameters from Java to native.*/
static void get_model_params(JNIEnv *env, jobject params,
		llama_model_params *mparams, jjml_llm_model_state &model_state) {
	jclass clss = env->FindClass(JCLASS_MODEL_PARAMS.c_str());
	mparams->n_gpu_layers = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_gpu_layers", "()I"));
	mparams->split_mode = jjml_llm_split_mode_from_int(env->CallIntMethod(params,
			env->GetMethodID(clss, "split_mode", "()I")));
	mparams->main_gpu = env->CallIntMethod(params,
			env->GetMethodID(clss, "main_gpu", "()I"));

	jstring tensor_split_string = static_cast<jstring>(env->CallObjectMethod(
			params, env->GetMethodID(clss, "tensor_split", "()Ljava/lang/String;")));
	if (tensor_split_string != nullptr) {
		const char *tensor_split_cstr = env->GetStringUTFChars(
				tensor_split_string, nullptr);
		model_state.tensor_split = jjml_llm_parse_tensor_split(tensor_split_cstr);
		env->ReleaseStringUTFChars(tensor_split_string, tensor_split_cstr);
		env->DeleteLocalRef(tensor_split_string);
		if (!model_state.tensor_split.empty())
			mparams->tensor_split = model_state.tensor_split.data();
	}

	mparams->vocab_only = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "vocab_only", "()Z"));
	mparams->use_mmap = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "use_mmap", "()Z"));
	mparams->use_direct_io = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "use_direct_io", "()Z"));
	mparams->use_mlock = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "use_mlock", "()Z"));
	mparams->check_tensors = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "check_tensors", "()Z"));
	mparams->use_extra_bufts = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "use_extra_bufts", "()Z"));
	mparams->no_host = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "no_host", "()Z"));
	mparams->no_alloc = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "no_alloc", "()Z"));
}

static ggml_backend_dev_t jjml_llm_find_device(const std::string &device_selector) {
	if (device_selector.empty())
		return nullptr;

	if (device_selector[0] == '#') {
		if (device_selector.length() == 1)
			return nullptr;
		size_t pos = 0;
		size_t index;
		try {
			index = static_cast<size_t>(std::stoul(device_selector.substr(1), &pos));
		} catch (const std::exception&) {
			return nullptr;
		}
		if (pos != device_selector.length() - 1)
			return nullptr;
		if (index < ggml_backend_dev_count())
			return ggml_backend_dev_get(index);
		return nullptr;
	}

	for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
		ggml_backend_dev_t device = ggml_backend_dev_get(i);
		if (device == nullptr)
			continue;

		ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
		const char *backend = reg != nullptr ? ggml_backend_reg_name(reg) : nullptr;
		const char *name = ggml_backend_dev_name(device);
		if (name != nullptr && device_selector == name)
			return device;
		if (backend != nullptr && name != nullptr
				&& device_selector == std::string(backend) + ":" + name)
			return device;

		struct ggml_backend_dev_props props;
		ggml_backend_dev_get_props(device, &props);
		if (props.device_id != nullptr && device_selector == props.device_id)
			return device;
		if (props.description != nullptr && device_selector == props.description)
			return device;
	}

	return nullptr;
}

JNIEXPORT jobject JNICALL Java_org_argeo_jjml_llm_LlamaCppBackend_newModelParams(
		JNIEnv *env, jclass) {
	llama_model_params mparams = llama_model_default_params();

	jobject res = env->NewObject(
			argeo::jni::find_jclass(env, JCLASS_MODEL_PARAMS), //
			ModelParams__init, //
			mparams.n_gpu_layers, //
			nullptr, //
			mparams.split_mode, //
			mparams.main_gpu, //
			nullptr, //
			mparams.vocab_only, //
			mparams.use_mmap, //
			mparams.use_direct_io, //
			mparams.use_mlock, //
			mparams.check_tensors, //
			mparams.use_extra_bufts, //
			mparams.no_host, //
			mparams.no_alloc //
			);
	//set_model_params(env, res, default_mparams);
	return res;
}

/*
 * LIFECYCLE
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doInit(
		JNIEnv *env, jclass, jstring localPath, jobject modelParams,
		jobject progressCallback) {
	const char *path_model = env->GetStringUTFChars(localPath, nullptr);
	try {
		llama_model_params mparams = llama_model_default_params();
		jjml_llm_model_state model_state;
		get_model_params(env, modelParams, &mparams, model_state);

		ggml_backend_load_all();

		std::unique_ptr<ggml_backend_dev_t[]> selected_devices;
		{
			jclass clss = env->FindClass(JCLASS_MODEL_PARAMS.c_str());
			jstring device = static_cast<jstring>(env->CallObjectMethod(modelParams,
					env->GetMethodID(clss, "device", "()Ljava/lang/String;")));
			if (device != nullptr) {
				const char *device_cstr = env->GetStringUTFChars(device, nullptr);
				std::string device_selector(device_cstr);
				env->ReleaseStringUTFChars(device, device_cstr);
				ggml_backend_dev_t selected = jjml_llm_find_device(device_selector);
				env->DeleteLocalRef(device);
				if (!device_selector.empty() && selected == nullptr)
					throw std::invalid_argument(
							"No ggml backend device matches '" + device_selector + "'");

				if (selected != nullptr) {
					selected_devices = std::make_unique<ggml_backend_dev_t[]>(2);
					selected_devices[0] = selected;
					selected_devices[1] = nullptr;
					mparams.devices = selected_devices.get();
				}
			}
		}

		// progress callback
		argeo::jni::java_callback progress_data { };
		if (progressCallback != nullptr) {
			progress_data.callback = env->NewGlobalRef(progressCallback);
			progress_data.method = DoublePredicate__test;
			env->GetJavaVM(&progress_data.jvm);
			mparams.progress_callback_user_data = &progress_data;

			mparams.progress_callback = [](float progress,
					void *user_data) -> bool {
				return argeo::jni::exec_boolean_callback(
						static_cast<argeo::jni::java_callback*>(user_data),
						static_cast<jdouble>(progress));
			};
		}

		llama_model *model = llama_model_load_from_file(path_model, mparams);
		if (!model) {
			if (progress_data.callback != nullptr)
				env->DeleteGlobalRef(progress_data.callback);
			throw std::runtime_error("Cannot load model");
		}
		jjml_llm_keep_model_state(model, std::move(model_state));

		// free callback global reference
		if (progress_data.callback != nullptr)
			env->DeleteGlobalRef(progress_data.callback);

		env->ReleaseStringUTFChars(localPath, path_model);
		return (jlong) model;
	} catch (const std::exception &ex) {
		env->ReleaseStringUTFChars(localPath, path_model);
		argeo::jni::throw_to_java(env, ex);
		return 0;
	}
}

JNIEXPORT jobjectArray JNICALL Java_org_argeo_jjml_llm_LlamaCppBackend_doGetDevices(
		JNIEnv *env, jclass) {
	try {
		ggml_backend_load_all();

		const jsize count = static_cast<jsize>(ggml_backend_dev_count());
		jclass deviceClass = argeo::jni::find_jclass(env, JCLASS_DEVICE);
		jobjectArray res = env->NewObjectArray(count, deviceClass, nullptr);

		for (jsize i = 0; i < count; i++) {
			ggml_backend_dev_t device = ggml_backend_dev_get(i);
			jobject deviceObj = jjml_llm_new_device(env, device);
			env->SetObjectArrayElement(res, i, deviceObj);
			env->DeleteLocalRef(deviceObj);
		}
		return res;
	} catch (const std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doDestroy(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	jjml_llm_drop_model_state(model);
	llama_model_free(model);
}

/*
 * ACCESSORS
 */
JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetVocabularySize(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	const llama_vocab *vocab = llama_model_get_vocab(model);
	return llama_vocab_n_tokens(vocab);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetContextTrainingSize(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return llama_model_n_ctx_train(model);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetEmbeddingSize(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return llama_model_n_embd_out(model);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetLayerCount(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return llama_model_n_layer(model);
}

/** Gather metadata keys or values. */
static jobjectArray jjml_lama_get_meta(JNIEnv *env, llama_model *model,
		std::function<int32_t(int32_t, char*, size_t)> supplier) {
	try {
		int32_t meta_count = llama_model_meta_count(model);

		jobjectArray res = env->NewObjectArray(meta_count, env->FindClass("[B"),
				nullptr);
		for (int32_t i = 0; i < meta_count; i++) {
			try {

				char buf[META_BUFFER_SIZE];
				int32_t length = supplier(i, buf, META_BUFFER_SIZE);
				if (length == -1)
					throw std::runtime_error(
							"Cannot read model metadata " + std::to_string(i));
				std::string u8_res;
				if (length > META_BUFFER_SIZE) { // chat templates can be quite big
					char big_buf[META_BIG_BUFFER_SIZE];
					// +1 for null terminator: snprintf needs buf_size to include
					// the null terminator, but 'length' is the string length without it
					length = supplier(i, big_buf, length + 1);
					u8_res = std::string(big_buf, length);
				} else {
					u8_res = std::string(buf, length);
				}
				jbyteArray str = env->NewByteArray(u8_res.length());
				env->SetObjectArrayElement(res, i, str);
				env->SetByteArrayRegion(str, 0, u8_res.length(),
						(jbyte*) u8_res.c_str());
			} catch (std::exception &ex) {
				// ignore
				std::cerr << "Cannot read metadata " << i << ": " << ex.what()
						<< ". Ignoring it." << std::endl;
			}
		}
		return res;
	} catch (std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
}

JNIEXPORT jobjectArray JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetMetadataKeys(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return jjml_lama_get_meta(env, model,
			[model](int32_t i, char *buf, size_t buf_size) {
				return llama_model_meta_key_by_index(model, i, buf, buf_size);
			});
}

JNIEXPORT jobjectArray JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetMetadataValues(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return jjml_lama_get_meta(env, model,
			[model](int32_t i, char *buf, size_t buf_size) {
				return llama_model_meta_val_str_by_index(model, i, buf,
						buf_size);
			});
}

JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetDescription(
		JNIEnv *env, jobject obj) {
	try {
		auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
		char buf[META_BUFFER_SIZE];
		int32_t length = llama_model_desc(model, buf, META_BUFFER_SIZE);
		if (length == -1)
			throw std::runtime_error("Cannot read model description ");
		std::string u8_res;
		if (length > META_BUFFER_SIZE) { // big description
			char big_buf[META_BIG_BUFFER_SIZE];
			length = llama_model_desc(model, big_buf, length + 1);
			u8_res = std::string(big_buf, length);
		} else {
			u8_res = std::string(buf, length);
		}
		jbyteArray res = env->NewByteArray(u8_res.length());
		env->SetByteArrayRegion(res, 0, u8_res.length(),
				(jbyte*) u8_res.c_str());
		return res;
	} catch (std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetModelSize(
		JNIEnv *env, jobject obj) {
	static_assert(sizeof(jlong) >= sizeof(uint64_t));
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return llama_model_size(model);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doEstimateKvCacheBytesPerToken(
		JNIEnv *env, jobject obj, jint typeK, jint typeV,
		jint flashAttentionType) {
	try {
		const ggml_type ggml_type_k = jjml_llm_kv_cache_type_from_int(typeK,
				"type_k");
		const ggml_type ggml_type_v = jjml_llm_kv_cache_type_from_int(typeV,
				"type_v");
		llama_flash_attn_type flash_attn_type =
				jjml_llm_flash_attn_type_from_int(flashAttentionType);
		auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
		return static_cast<jlong>(
				llama_model_estimate_kv_cache_bytes_per_token(model, ggml_type_k,
						ggml_type_v, flash_attn_type));
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
		return 0;
	}
}

JNIEXPORT jobjectArray JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetDevices(
		JNIEnv *env, jobject obj) {
	try {
		auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
		const jsize count = static_cast<jsize>(llama_model_n_devices(model));
		jclass deviceClass = argeo::jni::find_jclass(env, JCLASS_DEVICE);
		jobjectArray res = env->NewObjectArray(count, deviceClass, nullptr);

		for (jsize i = 0; i < count; i++) {
			ggml_backend_dev_t device = llama_model_get_device(model, i);
			if (device == nullptr)
				continue;
			jobject deviceObj = jjml_llm_new_device(env, device);
			env->SetObjectArrayElement(res, i, deviceObj);
			env->DeleteLocalRef(deviceObj);
		}
		return res;
	} catch (const std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetEndOfGenerationToken(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	const llama_vocab *vocab = llama_model_get_vocab(model);
	llama_token eot = llama_vocab_eot(vocab);
	return eot == -1 ? llama_vocab_eos(vocab) : eot;
}
