#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <ggml.h>
#include <llama.h>

#include <argeo/jni/argeo_jni.h>

#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "../tp/llama.cpp/src/llama-ext.h"

#include "org_argeo_jjml_llm_LlamaCppSpeculativeProcessor.h" // IWYU pragma: keep

namespace {

static constexpr llama_seq_id JJML_SPEC_SEQ_ID = 0;
static constexpr size_t JJML_SPEC_META_BUFFER_SIZE = 1024;

struct jjml_speculative_stats {
	int64_t prompt_tokens = 0;
	int64_t generated_tokens = 0;
	int64_t drafted_tokens = 0;
	int64_t accepted_draft_tokens = 0;
	int64_t decode_nanos = 0;
};

struct jjml_speculative_engine {
	llama_context *ctx_tgt = nullptr;
	llama_context *ctx_dft = nullptr;
	common_speculative *spec = nullptr;

	llama_tokens prompt;
	llama_tokens pending_output;
	std::vector<char> pending_output_origin;
	llama_token id_last = LLAMA_TOKEN_NULL;
	llama_pos n_past = 0;
	bool begun = false;
	bool has_eog = false;

	int32_t n_draft_max = 0;
	jjml_speculative_stats stats;

	~jjml_speculative_engine() {
		if (spec != nullptr) {
			common_speculative_free(spec);
			spec = nullptr;
		}
		if (ctx_dft != nullptr) {
			llama_free(ctx_dft);
			ctx_dft = nullptr;
		}
	}
};

static ggml_type jjml_spec_ggml_type_from_int(jint value,
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

static std::string jjml_spec_model_meta(const llama_model *model,
		const char *key) {
	char buf[JJML_SPEC_META_BUFFER_SIZE];
	int32_t length = llama_model_meta_val_str(model, key, buf, sizeof(buf));
	if (length < 0)
		return "";
	if (static_cast<size_t>(length) >= sizeof(buf)) {
		std::vector<char> big(static_cast<size_t>(length) + 1);
		length = llama_model_meta_val_str(model, key, big.data(), big.size());
		if (length < 0)
			return "";
		return std::string(big.data(), static_cast<size_t>(length));
	}
	return std::string(buf, static_cast<size_t>(length));
}

static int32_t jjml_spec_parse_non_negative_int(const std::string &value) {
	if (value.empty())
		return 0;
	char *end = nullptr;
	long parsed = std::strtol(value.c_str(), &end, 10);
	if (end == value.c_str() || parsed <= 0)
		return 0;
	return static_cast<int32_t>(parsed);
}

static bool jjml_spec_ends_with(const std::string &value,
		const std::string &suffix) {
	return value.size() >= suffix.size()
			&& value.compare(value.size() - suffix.size(), suffix.size(),
					suffix) == 0;
}

static int32_t jjml_spec_model_nextn_predict_layers(
		const llama_model *model, std::string &metadata_key) {
	const std::string arch = jjml_spec_model_meta(model,
			"general.architecture");
	if (!arch.empty()) {
		metadata_key = arch + ".nextn_predict_layers";
		int32_t value = jjml_spec_parse_non_negative_int(
				jjml_spec_model_meta(model, metadata_key.c_str()));
		if (value > 0)
			return value;
	}

	const std::string suffix = ".nextn_predict_layers";
	int32_t count = llama_model_meta_count(model);
	for (int32_t i = 0; i < count; ++i) {
		char key_buf[JJML_SPEC_META_BUFFER_SIZE];
		int32_t key_len = llama_model_meta_key_by_index(model, i, key_buf,
				sizeof(key_buf));
		if (key_len < 0)
			continue;
		std::string key;
		if (static_cast<size_t>(key_len) >= sizeof(key_buf)) {
			std::vector<char> big(static_cast<size_t>(key_len) + 1);
			key_len = llama_model_meta_key_by_index(model, i, big.data(),
					big.size());
			if (key_len < 0)
				continue;
			key.assign(big.data(), static_cast<size_t>(key_len));
		} else {
			key.assign(key_buf, static_cast<size_t>(key_len));
		}
		if (!jjml_spec_ends_with(key, suffix))
			continue;
		int32_t value = jjml_spec_parse_non_negative_int(
				jjml_spec_model_meta(model, key.c_str()));
		if (value > 0) {
			metadata_key = key;
			return value;
		}
	}

	return 0;
}

static common_params_speculative jjml_spec_params_from_java(JNIEnv *env,
		jobject params) {
	jclass clss = env->GetObjectClass(params);

	jint type_code = env->CallIntMethod(params,
			env->GetMethodID(clss, "typeCode", "()I"));
	if (env->ExceptionCheck())
		throw std::runtime_error("Cannot read speculative type");

	common_params_speculative spec_params;
	spec_params.types.clear();

	switch (type_code) {
	case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
		spec_params.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
		break;
	default:
		throw std::invalid_argument(
				"Unsupported speculative type code: "
						+ std::to_string(type_code));
	}

	spec_params.draft.n_max = env->CallIntMethod(params,
			env->GetMethodID(clss, "draftMax", "()I"));
	spec_params.draft.n_min = env->CallIntMethod(params,
			env->GetMethodID(clss, "draftMin", "()I"));
	spec_params.draft.p_split = env->CallFloatMethod(params,
			env->GetMethodID(clss, "draftSplitProbability", "()F"));
	spec_params.draft.p_min = env->CallFloatMethod(params,
			env->GetMethodID(clss, "draftMinProbability", "()F"));
	spec_params.draft.cache_type_k = jjml_spec_ggml_type_from_int(
			env->CallIntMethod(params,
					env->GetMethodID(clss, "cacheTypeKCode", "()I")),
			"speculative cacheTypeK");
	spec_params.draft.cache_type_v = jjml_spec_ggml_type_from_int(
			env->CallIntMethod(params,
					env->GetMethodID(clss, "cacheTypeVCode", "()I")),
			"speculative cacheTypeV");

	if (env->ExceptionCheck())
		throw std::runtime_error("Cannot read speculative parameters");
	if (spec_params.draft.n_max <= 0)
		throw std::invalid_argument("Speculative draftMax must be positive");
	if (spec_params.draft.n_min < 0
			|| spec_params.draft.n_min > spec_params.draft.n_max)
		throw std::invalid_argument("Invalid speculative draftMin");

	return spec_params;
}

static void jjml_spec_batch_add(llama_batch &batch, llama_token id,
		llama_pos pos, bool logits) {
	common_batch_add(batch, id, pos, { JJML_SPEC_SEQ_ID }, logits);
}

static void jjml_spec_validate_batch(const llama_batch &batch,
		int32_t n_tokens_alloc, const char *usage) {
	if (n_tokens_alloc <= 0)
		throw std::runtime_error(
				std::string("Invalid MTP batch capacity for ") + usage);
	if (batch.token == nullptr || batch.pos == nullptr
			|| batch.n_seq_id == nullptr || batch.seq_id == nullptr
			|| batch.logits == nullptr) {
		throw std::runtime_error(
				std::string("Failed to allocate MTP token batch for ")
						+ usage);
	}
	for (int32_t i = 0; i < n_tokens_alloc; ++i) {
		if (batch.seq_id[i] == nullptr) {
			throw std::runtime_error(
					std::string("Failed to allocate MTP batch seq_id rows for ")
							+ usage);
		}
	}
}

static void jjml_spec_decode_or_throw(llama_context *ctx,
		const llama_batch &batch, common_speculative *spec) {
	const int rc = llama_decode(ctx, batch);
	if (rc != 0)
		throw std::runtime_error(
				"llama_decode failed with code " + std::to_string(rc));
	if (spec != nullptr && !common_speculative_process(spec, batch))
		throw std::runtime_error("Speculative processor rejected decoded batch");
}

static void jjml_spec_seq_rm(llama_context *ctx, llama_seq_id seq_id,
		llama_pos p0, llama_pos p1) {
	if (!llama_memory_seq_rm(llama_get_memory(ctx), seq_id, p0, p1)) {
		throw std::runtime_error(
				"Failed to roll back speculative sequence state from position "
						+ std::to_string(p0)
						+ ". Recreate the target context with n_rs_seq >= draftMax "
						+ "if this model uses recurrent state.");
	}
}

static void jjml_spec_reserve_graph(llama_context *ctx, uint32_t n_tokens,
		uint32_t n_outputs, const char *ctx_name) {
	if (llama_graph_reserve(ctx, std::max<uint32_t>(1, n_tokens),
			std::max<uint32_t>(1, llama_n_seq_max(ctx)),
			std::max<uint32_t>(1, n_outputs)) == nullptr) {
		throw std::runtime_error(
				std::string("Failed to reserve MTP graph for ") + ctx_name);
	}
}

static bool jjml_spec_trace_output(int64_t index) {
	const char *enabled = std::getenv("JJML_SPEC_TRACE_OUTPUT");
	if (enabled == nullptr || enabled[0] == '\0'
			|| std::strcmp(enabled, "0") == 0)
		return false;
	const char *from_env = std::getenv("JJML_SPEC_TRACE_FROM");
	const char *to_env = std::getenv("JJML_SPEC_TRACE_TO");
	int64_t from = from_env != nullptr ? std::strtoll(from_env, nullptr, 10) : 0;
	int64_t to = to_env != nullptr ? std::strtoll(to_env, nullptr, 10)
			: INT64_MAX;
	return index >= from && index <= to;
}

static void jjml_spec_trace_emit(const jjml_speculative_engine *engine,
		char origin, llama_token id) {
	const int64_t index = engine->stats.generated_tokens;
	if (!jjml_spec_trace_output(index))
		return;
	std::fprintf(stderr,
			"JJML_SPEC_TOKEN index=%lld origin=%c token=%d n_past=%d prompt=%zu\n",
			static_cast<long long>(index), origin, static_cast<int>(id),
			static_cast<int>(engine->n_past), engine->prompt.size());
}

static void jjml_spec_prefill(jjml_speculative_engine *engine,
		llama_sampler *sampler, const llama_tokens &tokens) {
	if (tokens.empty())
		throw std::invalid_argument("Prompt token list is empty");
	if ((uint32_t) tokens.size() >= llama_n_ctx(engine->ctx_tgt))
		throw std::invalid_argument("Prompt exceeds the target context size");

	engine->prompt.clear();
	engine->prompt.reserve(llama_n_ctx(engine->ctx_tgt));
	engine->pending_output.clear();
	engine->pending_output_origin.clear();
	engine->id_last = LLAMA_TOKEN_NULL;
	engine->n_past = 0;
	engine->has_eog = false;

	const uint32_t n_batch = std::max<uint32_t>(1,
			std::min<uint32_t>(llama_n_batch(engine->ctx_tgt),
					llama_n_ubatch(engine->ctx_tgt)));
	size_t pos = 0;
	while (pos < tokens.size()) {
		const size_t remaining = tokens.size() - pos;
		const int32_t n_eval = static_cast<int32_t>(
				std::min<size_t>(remaining, n_batch));
		llama_batch batch = llama_batch_init(n_eval, 0, 1);
		try {
			jjml_spec_validate_batch(batch, n_eval, "prompt prefill");
			for (int32_t i = 0; i < n_eval; ++i) {
				const bool logits = pos + static_cast<size_t>(i)
						== tokens.size() - 1;
				jjml_spec_batch_add(batch, tokens[pos + i],
						static_cast<llama_pos>(pos + i), logits);
			}
			jjml_spec_decode_or_throw(engine->ctx_tgt, batch, engine->spec);
		} catch (...) {
			llama_batch_free(batch);
			throw;
		}
		llama_batch_free(batch);
		pos += n_eval;
	}

	llama_sampler_reset(sampler);

	engine->prompt = tokens;
	engine->n_past = static_cast<llama_pos>(tokens.size());

	const llama_token first = llama_sampler_sample(sampler, engine->ctx_tgt, -1);
	engine->pending_output.push_back(first);
	engine->pending_output_origin.push_back('P');
	engine->id_last = first;

	common_speculative_begin(engine->spec, JJML_SPEC_SEQ_ID, engine->prompt);
	engine->stats.prompt_tokens = static_cast<int64_t>(tokens.size());
	engine->begun = true;
}

static std::vector<llama_token> jjml_sample_and_accept_n(llama_sampler *sampler,
		llama_context *ctx, const llama_tokens &draft) {
	std::vector<llama_token> result;
	result.reserve(draft.size() + 1);
	size_t i = 0;
	for (; i < draft.size(); i++) {
		const llama_token id = llama_sampler_sample(sampler, ctx,
				static_cast<int32_t>(i));
		result.push_back(id);
		if (draft[i] != id)
			break;
	}
	if (i == draft.size()) {
		const llama_token id = llama_sampler_sample(sampler, ctx,
				static_cast<int32_t>(i));
		result.push_back(id);
	}
	return result;
}

static int jjml_spec_generate_with_sampler(jjml_speculative_engine *engine,
		llama_sampler *sampler, llama_token *out, int max_tokens) {
	if (!engine->begun)
		throw std::runtime_error("Speculative generation has not begun");
	if (max_tokens < 0)
		throw std::invalid_argument("max_tokens must not be negative");
	if (max_tokens == 0 || engine->has_eog)
		return 0;

	const llama_vocab *vocab = llama_model_get_vocab(
			llama_get_model(engine->ctx_tgt));

	int written = 0;
	const auto begin = std::chrono::steady_clock::now();
	while (written < max_tokens && !engine->pending_output.empty()
			&& !engine->has_eog) {
		const llama_token id = engine->pending_output.front();
		const char origin = engine->pending_output_origin.front();
		jjml_spec_trace_emit(engine, origin, id);
		engine->pending_output.erase(engine->pending_output.begin());
		engine->pending_output_origin.erase(
				engine->pending_output_origin.begin());
		out[written++] = id;
		engine->stats.generated_tokens++;
		if (llama_vocab_is_eog(vocab, id))
			engine->has_eog = true;
	}

	while (written < max_tokens && !engine->has_eog) {
		llama_tokens draft;
		draft.reserve(engine->n_draft_max);

		const int remaining = max_tokens - written;
		if (remaining > 1) {
			common_speculative_get_draft_params(engine->spec, JJML_SPEC_SEQ_ID) = {
				/* .drafting   = */ true,
				/* .n_max      = */ std::min(engine->n_draft_max,
						remaining - 1),
				/* .n_past     = */ engine->n_past,
				/* .id_last    = */ engine->id_last,
				/* .prompt     = */ &engine->prompt,
				/* .result     = */ &draft,
			};
			common_speculative_draft(engine->spec);

			// Drafting advances ctx_dft speculatively. Before the target verifies
			// [id_last, draft...], ctx_dft must be back at the same boundary so
			// common_speculative_process can mirror that verified batch.
			jjml_spec_seq_rm(engine->ctx_dft, JJML_SPEC_SEQ_ID,
					engine->n_past, -1);
		}
		const size_t n_draft = draft.size();
		engine->stats.drafted_tokens += static_cast<int64_t>(n_draft);

		llama_batch batch = llama_batch_init(
				static_cast<int32_t>(1 + draft.size()), 0, 1);
		try {
			jjml_spec_validate_batch(batch,
					static_cast<int32_t>(1 + draft.size()),
					"speculative verification");
			jjml_spec_batch_add(batch, engine->id_last, engine->n_past,
					true);
			for (size_t i = 0; i < draft.size(); ++i) {
				jjml_spec_batch_add(batch, draft[i],
						engine->n_past + static_cast<llama_pos>(i) + 1,
						true);
			}
			jjml_spec_decode_or_throw(engine->ctx_tgt, batch,
					engine->spec);
		} catch (...) {
			llama_batch_free(batch);
			throw;
		}
		llama_batch_free(batch);

		std::vector<llama_token> ids = jjml_sample_and_accept_n(sampler,
				engine->ctx_tgt, draft);
		if (ids.empty())
			throw std::runtime_error("Sampling produced no token");

		auto eog = std::find_if(ids.begin(), ids.end(),
				[vocab](llama_token id) {
					return llama_vocab_is_eog(vocab, id);
				});
		if (eog != ids.end()) {
			ids.resize(static_cast<size_t>(eog - ids.begin()) + 1);
		}

		const size_t accepted_draft = ids.size() - 1;
		if (n_draft > 0)
			common_speculative_accept(engine->spec, JJML_SPEC_SEQ_ID,
					static_cast<uint16_t>(accepted_draft));

		engine->stats.accepted_draft_tokens += static_cast<int64_t>(
				accepted_draft);

		engine->prompt.push_back(engine->id_last);
		engine->id_last = ids.back();
		engine->n_past += static_cast<llama_pos>(ids.size());

		engine->pending_output.insert(engine->pending_output.end(),
				ids.begin(), ids.end());
		for (size_t i = 0; i < ids.size(); ++i)
			engine->pending_output_origin.push_back(
					i < accepted_draft ? 'A' : 'S');

		engine->prompt.insert(engine->prompt.end(), ids.begin(), ids.end() - 1);

		jjml_spec_seq_rm(engine->ctx_tgt, JJML_SPEC_SEQ_ID, engine->n_past,
				-1);
		jjml_spec_seq_rm(engine->ctx_dft, JJML_SPEC_SEQ_ID, engine->n_past,
				-1);

		while (written < max_tokens && !engine->pending_output.empty()
				&& !engine->has_eog) {
			const llama_token id = engine->pending_output.front();
			const char origin = engine->pending_output_origin.front();
			jjml_spec_trace_emit(engine, origin, id);
			engine->pending_output.erase(engine->pending_output.begin());
			engine->pending_output_origin.erase(
					engine->pending_output_origin.begin());
			out[written++] = id;
			engine->stats.generated_tokens++;
			if (llama_vocab_is_eog(vocab, id))
				engine->has_eog = true;
		}

		if (n_draft == 0 && ids.empty())
			throw std::runtime_error("Speculative decoder made no progress");
	}

	const auto end = std::chrono::steady_clock::now();
	engine->stats.decode_nanos += std::chrono::duration_cast<
			std::chrono::nanoseconds>(end - begin).count();
	return written;
}

} // namespace

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppSpeculativeProcessor_doInit(
		JNIEnv *env, jclass, jlong contextPointer, jobject params) {
	try {
		auto *ctx_tgt = argeo::jni::as_pointer<llama_context*>(contextPointer);
		if (ctx_tgt == nullptr)
			throw std::invalid_argument("Target context pointer is null");

		common_params_speculative spec_params = jjml_spec_params_from_java(env,
				params);
		spec_params.draft.backend_sampling = false;

		const llama_model *model_tgt = llama_get_model(ctx_tgt);
		std::string nextn_key;
		int32_t nextn_layers = jjml_spec_model_nextn_predict_layers(model_tgt,
				nextn_key);
		if (nextn_layers <= 0) {
			throw std::runtime_error(
					"MTP requires a GGUF with NextN/MTP layers. Metadata "
							+ (nextn_key.empty() ?
									std::string("<arch>.nextn_predict_layers") :
									nextn_key)
							+ " is missing or zero.");
		}

		llama_context_params cparams = llama_context_default_params();
		cparams.n_ctx = llama_n_ctx(ctx_tgt);
		cparams.n_batch = llama_n_batch(ctx_tgt);
		cparams.n_ubatch = llama_n_ubatch(ctx_tgt);
		cparams.n_seq_max = llama_n_seq_max(ctx_tgt);
		cparams.n_outputs_max = 1;
		cparams.n_threads = llama_n_threads(ctx_tgt);
		cparams.n_threads_batch = llama_n_threads_batch(ctx_tgt);
		cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
		cparams.type_k = spec_params.draft.cache_type_k;
		cparams.type_v = spec_params.draft.cache_type_v;
		cparams.n_rs_seq = spec_params.draft.n_max;
		cparams.ctx_other = ctx_tgt;

		std::unique_ptr<jjml_speculative_engine> engine(
				new jjml_speculative_engine());
		engine->ctx_tgt = ctx_tgt;
		engine->ctx_dft = llama_init_from_model(
				const_cast<llama_model*>(model_tgt), cparams);
		if (engine->ctx_dft == nullptr)
			throw std::runtime_error(
					"Failed to create MTP context for model with "
							+ std::to_string(nextn_layers)
							+ " NextN/MTP layer(s). Check that the GGUF contains the matching MTP tensors and that the Vulkan backend can allocate the extra draft context.");

		spec_params.draft.ctx_tgt = ctx_tgt;
		spec_params.draft.ctx_dft = engine->ctx_dft;
		engine->n_draft_max = spec_params.draft.n_max;
		engine->spec = common_speculative_init(spec_params, 1);
		if (engine->spec == nullptr)
			throw std::runtime_error("Failed to initialize speculative decoding");

		const uint32_t n_tgt_tokens = std::max<uint32_t>(1,
				std::min<uint32_t>(llama_n_ubatch(ctx_tgt),
						llama_n_ctx(ctx_tgt)));
		jjml_spec_reserve_graph(ctx_tgt, n_tgt_tokens,
				static_cast<uint32_t>(spec_params.draft.n_max + 1),
				"target context");
		jjml_spec_reserve_graph(engine->ctx_dft, llama_n_seq_max(engine->ctx_dft),
				llama_n_seq_max(engine->ctx_dft), "MTP draft context");

		return reinterpret_cast<jlong>(engine.release());
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
		return 0;
	}
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llm_LlamaCppSpeculativeProcessor_doDestroy(
		JNIEnv *env, jobject obj) {
	try {
		auto *engine = argeo::jni::as_pointer<jjml_speculative_engine*>(env,
				obj);
		delete engine;
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
	}
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llm_LlamaCppSpeculativeProcessor_doBegin(
		JNIEnv *env, jobject obj, jlong samplerChainPointer,
		jintArray promptTokens, jint offset, jint length) {
	try {
		auto *engine = argeo::jni::as_pointer<jjml_speculative_engine*>(env,
				obj);
		auto *sampler = argeo::jni::as_pointer<llama_sampler*>(
				samplerChainPointer);
		if (engine == nullptr)
			throw std::invalid_argument("Speculative engine pointer is null");
		if (sampler == nullptr)
			throw std::invalid_argument("Sampler chain pointer is null");
		if (length <= 0)
			throw std::invalid_argument("Prompt token list is empty");

		jint *arr = env->GetIntArrayElements(promptTokens, nullptr);
		try {
			llama_tokens tokens;
			tokens.reserve(length);
			for (int i = 0; i < length; ++i)
				tokens.push_back(static_cast<llama_token>(arr[offset + i]));
			jjml_spec_prefill(engine, sampler, tokens);
		} catch (...) {
			env->ReleaseIntArrayElements(promptTokens, arr, JNI_ABORT);
			throw;
		}
		env->ReleaseIntArrayElements(promptTokens, arr, JNI_ABORT);
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
	}
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppSpeculativeProcessor_doRead(
		JNIEnv *env, jobject obj, jlong samplerChainPointer, jobject output,
		jint offset, jint length) {
	try {
		auto *engine = argeo::jni::as_pointer<jjml_speculative_engine*>(env,
				obj);
		auto *sampler = argeo::jni::as_pointer<llama_sampler*>(
				samplerChainPointer);
		void *output_arr = env->GetDirectBufferAddress(output);
		if (output_arr == nullptr)
			throw std::invalid_argument("Output is not a direct buffer");
		auto *tokens = static_cast<llama_token*>(output_arr) + offset;
		return jjml_spec_generate_with_sampler(engine, sampler, tokens, length);
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
		return 0;
	}
}

JNIEXPORT jintArray JNICALL Java_org_argeo_jjml_llm_LlamaCppSpeculativeProcessor_doReadArray(
		JNIEnv *env, jobject obj, jlong samplerChainPointer, jint maxTokens) {
	try {
		auto *engine = argeo::jni::as_pointer<jjml_speculative_engine*>(env,
				obj);
		auto *sampler = argeo::jni::as_pointer<llama_sampler*>(
				samplerChainPointer);
		std::vector<llama_token> output(maxTokens);
		int read = jjml_spec_generate_with_sampler(engine, sampler, output.data(),
				maxTokens);
		jintArray result = env->NewIntArray(read);
		env->SetIntArrayRegion(result, 0, read,
				reinterpret_cast<const jint*>(output.data()));
		return result;
	} catch (const std::exception &ex) {
		return reinterpret_cast<jintArray>(argeo::jni::throw_to_java(env, ex));
	}
}

JNIEXPORT jlongArray JNICALL Java_org_argeo_jjml_llm_LlamaCppSpeculativeProcessor_doGetStats(
		JNIEnv *env, jobject obj) {
	try {
		auto *engine = argeo::jni::as_pointer<jjml_speculative_engine*>(env,
				obj);
		jlong values[5] = { engine->stats.prompt_tokens,
				engine->stats.generated_tokens, engine->stats.drafted_tokens,
				engine->stats.accepted_draft_tokens,
				engine->stats.decode_nanos };
		jlongArray result = env->NewLongArray(5);
		env->SetLongArrayRegion(result, 0, 5, values);
		return result;
	} catch (const std::exception &ex) {
		return reinterpret_cast<jlongArray>(argeo::jni::throw_to_java(env, ex));
	}
}
