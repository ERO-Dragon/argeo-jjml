#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <llama.h>

#include <argeo/jni/argeo_jni.h>

#include "org_argeo_jjml_llm_LlamaCppEmbeddingProcessor.h" // IWYU pragma: keep

#include "jjml_llm.h"

namespace {
class int_array_elements {
public:
	int_array_elements(JNIEnv *env, jintArray array) :
			env(env), array(array) {
		if (array != nullptr) {
			data = env->GetIntArrayElements(array, nullptr);
			if (data == nullptr)
				throw std::runtime_error("Failed to access int array");
		}
	}

	~int_array_elements() {
		if (data != nullptr)
			env->ReleaseIntArrayElements(array, data, JNI_ABORT);
	}

	jint* get() const {
		return data;
	}

private:
	JNIEnv *env;
	jintArray array;
	jint *data = nullptr;
};

class llama_batch_owner {
public:
	llama_batch_owner(int32_t n_tokens, int32_t embd, int32_t n_seq_max) :
			batch(llama_batch_init(n_tokens, embd, n_seq_max)) {
	}

	~llama_batch_owner() {
		llama_batch_free(batch);
	}

	llama_batch& get() {
		return batch;
	}

private:
	llama_batch batch;
};

// from llama.cpp's common llama_embd_normalize
static void embd_normalize(const float *inp, float *out, int n,
		int embd_norm) {
	double sum = 0.0;

	switch (embd_norm) {
	case -1: // no normalisation
		sum = 1.0;
		break;
	case 0: // max absolute
		for (int i = 0; i < n; i++) {
			if (sum < std::abs(inp[i]))
				sum = std::abs(inp[i]);
		}
		sum /= 32760.0; // make an int16 range
		break;
	case 2: // euclidean
		for (int i = 0; i < n; i++) {
			sum += inp[i] * inp[i];
		}
		sum = std::sqrt(sum);
		break;
	default: // p-norm (euclidean is p-norm p=2)
		for (int i = 0; i < n; i++) {
			sum += std::pow(std::abs(inp[i]), embd_norm);
		}
		sum = std::pow(sum, 1.0 / embd_norm);
		break;
	}

	const float norm = sum > 0.0 ? 1.0f / sum : 0.0f;

	for (int i = 0; i < n; i++) {
		out[i] = inp[i] * norm;
	}
}

static void embedding_batch_add_seq(llama_batch &batch,
		const llama_token *tokens, int32_t length, llama_seq_id seq_id) {
	for (int32_t i = 0; i < length; i++) {
		batch.token[batch.n_tokens] = tokens[i];
		batch.pos[batch.n_tokens] = i;
		batch.n_seq_id[batch.n_tokens] = 1;
		batch.seq_id[batch.n_tokens][0] = seq_id;
		batch.logits[batch.n_tokens] = true;
		batch.n_tokens++;
	}
}

static void embedding_batch_decode(llama_context *ctx, llama_batch &batch,
		float *output, int n_seq, int n_embd, int embd_norm) {
	if (batch.n_tokens == 0)
		return;

	const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

	llama_memory_clear(llama_get_memory(ctx), true);

	const int decode_status = llama_decode(ctx, batch);
	if (decode_status == 1)
		throw std::runtime_error(
				"Failed to compute embeddings: llama.cpp could not find a KV slot for the batch");
	if (decode_status == 2)
		throw std::runtime_error(
				"Failed to compute embeddings: llama.cpp evaluation was aborted");
	if (decode_status < 0)
		throw std::runtime_error(
				"Failed to compute embeddings, llama_decode returned "
						+ std::to_string(decode_status));

	if (pooling_type != LLAMA_POOLING_TYPE_NONE
			&& n_seq <= 0)
		throw std::invalid_argument(
				"Sequence embeddings require at least one sequence");

	for (int i = 0; i < batch.n_tokens; i++) {
		if (!batch.logits[i])
			continue;

		const float *embd = nullptr;
		int embd_pos = 0;

		if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
			embd = llama_get_embeddings_ith(ctx, i);
			embd_pos = i;
			if (embd == nullptr)
				throw std::runtime_error("Failed to get token embeddings");
		} else {
			const llama_seq_id seq_id = batch.seq_id[i][0];
			if (seq_id < 0 || seq_id >= n_seq)
				throw std::runtime_error(
						"Invalid embedding sequence id "
								+ std::to_string(seq_id));
			embd = llama_get_embeddings_seq(ctx, seq_id);
			embd_pos = seq_id;
			if (embd == nullptr)
				throw std::runtime_error("Failed to get sequence embeddings");
		}

		float *out = output + embd_pos * n_embd;
		embd_normalize(embd, out, n_embd, embd_norm);
	}
}

static void validate_embedding_inputs(JNIEnv *env, llama_context *ctx,
		jobjectArray token_lists, jintArray offsets, jintArray lengths,
		const std::vector<jint> &offset_values,
		const std::vector<jint> &length_values) {
	const int n_prompts = env->GetArrayLength(token_lists);
	if (env->GetArrayLength(offsets) != n_prompts
			|| env->GetArrayLength(lengths) != n_prompts)
		throw std::invalid_argument(
				"Token list, offset, and length arrays must have the same size");

	const int n_batch = llama_n_batch(ctx);
	for (int i = 0; i < n_prompts; i++) {
		jint length = length_values[i];
		if (length <= 0)
			throw std::invalid_argument(
					"Embedding token list " + std::to_string(i)
							+ " is empty");
		if (length > n_batch)
			throw std::invalid_argument(
					"Embedding token list " + std::to_string(i) + " has "
							+ std::to_string(length)
							+ " tokens, exceeding context batch size "
							+ std::to_string(n_batch));
		if (offset_values[i] < 0)
			throw std::out_of_range(
					"Embedding token list " + std::to_string(i)
					+ " has invalid offset/length");

		jintArray token_list = static_cast<jintArray>(env->GetObjectArrayElement(
				token_lists, i));
		if (token_list == nullptr)
			throw std::invalid_argument(
					"Embedding token list " + std::to_string(i)
							+ " cannot be null");
		jint offset = offset_values[i];
		jint array_length = env->GetArrayLength(token_list);
		env->DeleteLocalRef(token_list);
		if (offset < 0 || length < 0 || offset + length > array_length)
			throw std::out_of_range(
					"Embedding token list " + std::to_string(i)
					+ " has invalid offset/length");
	}
}

static std::vector<jint> copy_int_array(JNIEnv *env, jintArray array,
		const char *name) {
	if (array == nullptr)
		throw std::invalid_argument(std::string(name) + " cannot be null");
	const int length = env->GetArrayLength(array);
	std::vector<jint> res(length);
	if (length > 0)
		env->GetIntArrayRegion(array, 0, length, res.data());
	return res;
}
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llm_LlamaCppEmbeddingProcessor_doProcessEmbeddings(
		JNIEnv *env, jclass, jlong contextPointer, jobjectArray tokenLists,
		jintArray offsets, jintArray lengths, jint maxSequences,
		jfloatArray res) {
	try {
		auto *ctx = argeo::jni::as_pointer<llama_context*>(contextPointer);
		if (ctx == nullptr)
			throw std::invalid_argument("Context pointer cannot be null");

		const struct llama_model *model = llama_get_model(ctx);
		if (llama_model_has_encoder(model) && llama_model_has_decoder(model))
			throw std::runtime_error(
					"Computing embeddings in encoder-decoder models is not supported");

		// TODO make normalization configurable.
		const int embd_normalize = -1;

		const int n_embd = llama_model_n_embd_out(model);
		const int n_batch = llama_n_batch(ctx);
		const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

		const int n_prompts = env->GetArrayLength(tokenLists);
		if (n_prompts == 0)
			return;

		std::vector<jint> offset_values = copy_int_array(env, offsets,
				"Embedding offsets");
		std::vector<jint> length_values = copy_int_array(env, lengths,
				"Embedding lengths");
		validate_embedding_inputs(env, ctx, tokenLists, offsets, lengths,
				offset_values, length_values);

		int n_embd_count = 0;
		if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
			for (jint length : length_values)
				n_embd_count += length;
		} else {
			n_embd_count = n_prompts;
		}

		const int usable_seq_max = std::max(1,
				std::min(static_cast<int>(maxSequences),
						static_cast<int>(llama_max_parallel_sequences())));
		llama_batch_owner batch_owner(n_batch, 0, 1);
		llama_batch &batch = batch_owner.get();

		const int output_length = env->GetArrayLength(res);
		const int expected_output_length = n_embd_count * n_embd;
		if (output_length != expected_output_length)
			throw std::invalid_argument(
					"Embedding output array has length "
							+ std::to_string(output_length)
							+ ", expected "
							+ std::to_string(expected_output_length));
		std::vector<float> embeddings(output_length, 0.0f);

		int e = 0; // number of embeddings already stored
		int s = 0; // number of prompts in current batch
		for (int k = 0; k < n_prompts; k++) {
			const int n_toks = length_values[k];

			if (batch.n_tokens > 0
					&& (batch.n_tokens + n_toks > n_batch
							|| s >= usable_seq_max)) {
				float *out = embeddings.data() + e * n_embd;
				embedding_batch_decode(ctx, batch, out, s, n_embd,
						embd_normalize);
				e += pooling_type == LLAMA_POOLING_TYPE_NONE ?
						batch.n_tokens : s;
				s = 0;
				jjml_llm_batch_clear(batch);
			}

			jintArray token_list =
					static_cast<jintArray>(env->GetObjectArrayElement(
							tokenLists, k));
			{
				int_array_elements token_values(env, token_list);
				const llama_token *tokens =
						reinterpret_cast<const llama_token*>(token_values.get())
								+ offset_values[k];
				embedding_batch_add_seq(batch, tokens, n_toks, s);
			}
			env->DeleteLocalRef(token_list);

			s += 1;
		}

		if (batch.n_tokens > 0) {
			float *out = embeddings.data() + e * n_embd;
			embedding_batch_decode(ctx, batch, out, s, n_embd,
					embd_normalize);
		}
		if (output_length > 0)
			env->SetFloatArrayRegion(res, 0, output_length,
					embeddings.data());
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
	}
}
