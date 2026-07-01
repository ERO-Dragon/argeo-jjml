#include <llama.h>

#include <argeo/jni/argeo_jni.h>

#include "org_argeo_jjml_llm_LlamaCppNativeSampler.h" // IWYU pragma: keep
#include "org_argeo_jjml_llm_LlamaCppSamplerChain.h" // IWYU pragma: keep
#include "org_argeo_jjml_llm_LlamaCppSamplers.h" // IWYU pragma: keep

#include "org_argeo_jjml_llm_.h"

#include "common.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

static std::string jjml_sampling_bytes(JNIEnv *env, jbyteArray bytes) {
	void *arr = env->GetPrimitiveArrayCritical(bytes, 0);
	std::string value(static_cast<char*>(arr), env->GetArrayLength(bytes));
	env->ReleasePrimitiveArrayCritical(bytes, arr, 0);
	return value;
}

static std::string jjml_sampling_bytes(JNIEnv *env, jobjectArray values,
		jsize index) {
	jbyteArray bytes = static_cast<jbyteArray>(env->GetObjectArrayElement(
			values, index));
	std::string value = jjml_sampling_bytes(env, bytes);
	env->DeleteLocalRef(bytes);
	return value;
}

} // namespace

/*
 * STANDARD SAMPLERS
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplers_doInitGreedy(
		JNIEnv*, jclass) {
	llama_sampler *smpl = llama_sampler_init_greedy();
	return reinterpret_cast<jlong>(smpl);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplers_doInitPenalties(
		JNIEnv *env, jclass, jint penalty_last_n, jfloat penalty_repeat,
		jfloat penalty_freq, jfloat penalty_present, jboolean penalize_nl,
		jboolean ignore_eos) {
	llama_sampler *smpl = llama_sampler_init_penalties(penalty_last_n,
			penalty_repeat, penalty_freq, penalty_present);
	return reinterpret_cast<jlong>(smpl);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplers_doInitTopK(
		JNIEnv*, jclass, jint top_k) {
	return reinterpret_cast<jlong>(llama_sampler_init_top_k(top_k));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplers_doInitTopP(
		JNIEnv*, jclass, jfloat top_p, jlong min_keep) {
	return reinterpret_cast<jlong>(llama_sampler_init_top_p(top_p, min_keep));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplers_doInitMinP(
		JNIEnv*, jclass, jfloat min_p, jlong min_keep) {
	return reinterpret_cast<jlong>(llama_sampler_init_min_p(min_p, min_keep));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplers_doInitTypicalP(
		JNIEnv*, jclass, jfloat typ_p, jlong min_keep) {
	return reinterpret_cast<jlong>(llama_sampler_init_typical(typ_p, min_keep));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplers_doInitTempExt(
		JNIEnv*, jclass, jfloat temp, jfloat dynatemp_ext,
		jfloat dynatemp_exponent) {
	return reinterpret_cast<jlong>(llama_sampler_init_temp_ext(temp,
			dynatemp_ext, dynatemp_exponent));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplers_doInitTemp(
		JNIEnv*, jclass, jfloat temp) {
	return reinterpret_cast<jlong>(llama_sampler_init_temp(temp));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplers_doInitDist__(
		JNIEnv*, jclass) {
	return reinterpret_cast<jlong>(llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplers_doInitDist__I(
		JNIEnv*, jclass, jint seed) {
	return reinterpret_cast<jlong>(llama_sampler_init_dist(seed));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplers_doInitGrammar(
		JNIEnv *env, jclass, jobject modelObj, jbyteArray grammarStr,
		jbyteArray rootStr) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, modelObj);
	const llama_vocab *vocab = llama_model_get_vocab(model);

	void *u8_grammar_arr = env->GetPrimitiveArrayCritical(grammarStr, 0);
	std::string u8_grammar(static_cast<char*>(u8_grammar_arr),
			env->GetArrayLength(grammarStr));

	void *u8_root_arr = env->GetPrimitiveArrayCritical(rootStr, 0);
	std::string u8_root(static_cast<char*>(u8_root_arr),
			env->GetArrayLength(rootStr));

	llama_sampler *smpl = llama_sampler_init_grammar(vocab, //
			u8_grammar.c_str(), u8_root.c_str());

	// clean up
	env->ReleasePrimitiveArrayCritical(grammarStr, u8_grammar_arr, 0);
	env->ReleasePrimitiveArrayCritical(rootStr, u8_root_arr, 0);

	return reinterpret_cast<jlong>(smpl);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplers_doInitGrammarLazy(
		JNIEnv *env, jclass, jobject modelObj, jbyteArray grammarStr,
		jbyteArray rootStr, jintArray triggerTypes, jobjectArray triggerValues,
		jintArray triggerTokens) {
	try {
		auto *model = argeo::jni::as_pointer<llama_model*>(env, modelObj);
		const llama_vocab *vocab = llama_model_get_vocab(model);
		std::string u8_grammar = jjml_sampling_bytes(env, grammarStr);
		std::string u8_root = jjml_sampling_bytes(env, rootStr);

		jsize n_triggers = env->GetArrayLength(triggerTypes);
		if (env->GetArrayLength(triggerValues) != n_triggers
				|| env->GetArrayLength(triggerTokens) != n_triggers)
			throw std::invalid_argument("Grammar trigger arrays have different sizes");

		jint *types = env->GetIntArrayElements(triggerTypes, nullptr);
		jint *tokens = env->GetIntArrayElements(triggerTokens, nullptr);
		if (types == nullptr || tokens == nullptr)
			throw std::runtime_error("Failed to access grammar trigger arrays");

		std::vector<std::string> trigger_patterns;
		std::vector<llama_token> trigger_tokens;
		try {
			for (jsize i = 0; i < n_triggers; ++i) {
				std::string value = jjml_sampling_bytes(env, triggerValues, i);
				switch (types[i]) {
				case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
					trigger_patterns.push_back(regex_escape(value));
					break;
				case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
					trigger_patterns.push_back(value);
					break;
				case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
				{
					std::string anchored = "^$";
					if (!value.empty()) {
						anchored = (value.front() != '^' ? "^" : "")
								+ value + (value.back() != '$' ? "$" : "");
					}
					trigger_patterns.push_back(anchored);
					break;
				}
				case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
					trigger_tokens.push_back(static_cast<llama_token>(tokens[i]));
					break;
				default:
					throw std::invalid_argument(
							"Unsupported grammar trigger type: "
									+ std::to_string(types[i]));
				}
			}
		} catch (...) {
			env->ReleaseIntArrayElements(triggerTypes, types, JNI_ABORT);
			env->ReleaseIntArrayElements(triggerTokens, tokens, JNI_ABORT);
			throw;
		}
		env->ReleaseIntArrayElements(triggerTypes, types, JNI_ABORT);
		env->ReleaseIntArrayElements(triggerTokens, tokens, JNI_ABORT);

		std::vector<const char*> trigger_patterns_c;
		trigger_patterns_c.reserve(trigger_patterns.size());
		for (const std::string &pattern : trigger_patterns)
			trigger_patterns_c.push_back(pattern.c_str());

		llama_sampler *smpl = llama_sampler_init_grammar_lazy_patterns(vocab,
				u8_grammar.c_str(), u8_root.c_str(),
				trigger_patterns_c.data(), trigger_patterns_c.size(),
				trigger_tokens.data(), trigger_tokens.size());
		if (smpl == nullptr)
			throw std::runtime_error("Failed to initialize lazy grammar sampler");
		return reinterpret_cast<jlong>(smpl);
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
		return 0;
	}
}

/*
 * JAVA SAMPLER
 */
struct jjml_llm_sampler_java {
	const jobject obj; //
	const jmethodID applyMethod; //
	const jmethodID acceptMethod; //
	const jmethodID resetMethod; //
	const char *name; //
	JavaVM *jvm; //
};

void jjml_llm_sampler_java_apply(struct llama_sampler *smpl,
		llama_token_data_array *cur_p) {
	const auto *ctx = static_cast<jjml_llm_sampler_java*>(smpl->ctx);
	JNIEnv *env;
	argeo::jni::load_thread_jnienv(ctx->jvm, &env);

	jobject obj = env->NewLocalRef(ctx->obj);
	jobject buf = env->NewDirectByteBuffer(cur_p->data,
			cur_p->size * sizeof(llama_token_data));
	jlong selected = env->CallLongMethod(obj, ctx->applyMethod, buf,
			cur_p->size, cur_p->selected, cur_p->sorted);

	if (selected >= 0) {
		cur_p->selected = static_cast<int64_t>(selected);
	}
}

void jjml_llm_sampler_java_accept(struct llama_sampler *smpl,
		llama_token token) {
	const auto *ctx = static_cast<jjml_llm_sampler_java*>(smpl->ctx);
	JNIEnv *env;
	argeo::jni::load_thread_jnienv(ctx->jvm, &env);

	jobject obj = env->NewLocalRef(ctx->obj);
	env->CallVoidMethod(obj, ctx->acceptMethod, token);
}

void jjml_llm_sampler_java_reset(struct llama_sampler *smpl) {
	const auto *ctx = static_cast<jjml_llm_sampler_java*>(smpl->ctx);
	JNIEnv *env;
	argeo::jni::load_thread_jnienv(ctx->jvm, &env);

	jobject obj = env->NewLocalRef(ctx->obj);
	env->CallVoidMethod(obj, ctx->resetMethod);
}

void jjml_llm_sampler_java_free(struct llama_sampler *smpl) {
	const auto *ctx = static_cast<jjml_llm_sampler_java*>(smpl->ctx);
	JNIEnv *env;
	argeo::jni::load_thread_jnienv(ctx->jvm, &env);

	// TODO call a close method on the Java object?
	env->DeleteGlobalRef(ctx->obj);
	delete ctx;
}

static const char* jjml_llm_sampler_java_name(
		const struct llama_sampler *smpl) {
	//return "java";
	const auto *ctx = static_cast<jjml_llm_sampler_java*>(smpl->ctx);
	return ctx->name;
}

static struct llama_sampler_i jjml_llm_sampler_java_i = {
/* .name   = */jjml_llm_sampler_java_name,
/* .accept = */jjml_llm_sampler_java_accept,
/* .apply  = */jjml_llm_sampler_java_apply,
/* .reset  = */jjml_llm_sampler_java_reset,
/* .clone  = */nullptr,
/* .free   = */jjml_llm_sampler_java_free, };

struct llama_sampler* jjml_llm_sampler_init_java(JNIEnv *env, jobject obj) {
//	jclass clss = argeo::jni::find_jclass(env, JCLASS_JAVA_SAMPLER);
	jjml_llm_sampler_java *ctx = new jjml_llm_sampler_java { env->NewGlobalRef(
			obj), //
	LlamaCppJavaSampler__apply, //
			LlamaCppJavaSampler__accept, //
			LlamaCppJavaSampler__reset, //
			"java" //
			};
	env->GetJavaVM(&ctx->jvm);
	return new llama_sampler {
	/* .iface = */&jjml_llm_sampler_java_i,
	/* .ctx   = */ctx, //
	};
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplers_doInitJavaSampler(
		JNIEnv *env, jclass, jobject javaSamplerObj) {
	return reinterpret_cast<jlong>(jjml_llm_sampler_init_java(env,
			javaSamplerObj));
}
/*
 * SAMPLER CHAIN
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplerChain_doInit(
		JNIEnv*, jclass) {
	auto sparams = llama_sampler_chain_default_params();
	llama_sampler *smpl = llama_sampler_chain_init(sparams);
	return reinterpret_cast<jlong>(smpl);
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplerChain_doAddSampler(
		JNIEnv *env, jobject obj, jobject child) {
	auto *chain = argeo::jni::as_pointer<llama_sampler*>(env, obj);
	auto *smpl = argeo::jni::as_pointer<llama_sampler*>(env, child);
	llama_sampler_chain_add(chain, smpl);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplerChain_doRemoveSampler(
		JNIEnv *env, jobject obj, jint index) {
	auto *chain = argeo::jni::as_pointer<llama_sampler*>(env, obj);
	auto *smpl = llama_sampler_chain_remove(chain, index);
	return reinterpret_cast<jlong>(smpl);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplerChain_doGetSampler(
		JNIEnv *env, jobject obj, jint index) {
	auto *chain = argeo::jni::as_pointer<llama_sampler*>(env, obj);
	auto *smpl = llama_sampler_chain_get(chain, index);
	return reinterpret_cast<jlong>(smpl);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppSamplerChain_doGetSize(
		JNIEnv *env, jobject obj) {
	auto *chain = argeo::jni::as_pointer<llama_sampler*>(env, obj);
	return llama_sampler_chain_n(chain);
}

/*
 * GENERIC SAMPLER
 */
JNIEXPORT void JNICALL Java_org_argeo_jjml_llm_LlamaCppNativeSampler_doReset(
		JNIEnv *env, jobject obj) {
	auto *smpl = argeo::jni::as_pointer<llama_sampler*>(env, obj);
	llama_sampler_reset(smpl);
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llm_LlamaCppNativeSampler_doDestroy(
		JNIEnv *env, jobject obj) {
	auto *smpl = argeo::jni::as_pointer<llama_sampler*>(env, obj);
	llama_sampler_free(smpl);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppNativeSampler_doClone(
		JNIEnv *env, jobject obj) {
	auto *smpl = argeo::jni::as_pointer<llama_sampler*>(env, obj);
	auto *cloned = llama_sampler_clone(smpl);
	return reinterpret_cast<jlong>(cloned);
}
