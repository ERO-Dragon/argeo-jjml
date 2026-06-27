#include <stddef.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <llama.h>

#include <argeo/jni/argeo_jni.h>

#include "org_argeo_jjml_llm_LlamaCppBatchProcessor.h" // IWYU pragma: keep

#include "jjml_llm.h"

namespace {
struct TokenSpan {
	const llama_token *tokens = nullptr;
	int32_t length = 0;
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

static std::vector<llama_seq_id> copy_sequence_ids(JNIEnv *env,
		jintArray sequence_ids) {
	std::vector<jint> values = copy_int_array(env, sequence_ids,
			"Sequence ids");
	std::vector<llama_seq_id> res(values.size());
	for (size_t i = 0; i < values.size(); i++)
		res[i] = static_cast<llama_seq_id>(values[i]);
	return res;
}

static std::vector<int32_t> copy_output_ids(JNIEnv *env, jintArray output_ids) {
	std::vector<jint> values = copy_int_array(env, output_ids, "Output ids");
	std::vector<int32_t> res(values.size());
	for (size_t i = 0; i < values.size(); i++)
		res[i] = static_cast<int32_t>(values[i]);
	return res;
}

static void write_output_ids(JNIEnv *env, jintArray output_ids,
		const std::vector<int32_t> &values) {
	if (values.empty())
		return;
	env->SetIntArrayRegion(output_ids, 0, static_cast<jsize>(values.size()),
			reinterpret_cast<const jint*>(values.data()));
}

static void validate_spans(const std::vector<TokenSpan> &inputs) {
	if (inputs.empty())
		throw std::invalid_argument("At least one input buffer is required");
	for (size_t i = 0; i < inputs.size(); i++) {
		if (inputs[i].tokens == nullptr)
			throw std::invalid_argument(
					"Input buffer " + std::to_string(i) + " cannot be null");
		if (inputs[i].length <= 0)
			throw std::invalid_argument(
					"Input buffer " + std::to_string(i) + " is empty");
	}
}

static void check_array_windows(JNIEnv *env, jobjectArray arrays,
		const std::vector<jint> &offsets, const std::vector<jint> &lengths) {
	const int count = env->GetArrayLength(arrays);
	if (static_cast<int>(offsets.size()) != count
			|| static_cast<int>(lengths.size()) != count)
		throw std::invalid_argument(
				"Input, offset, and length arrays must have the same size");
	for (int i = 0; i < count; i++) {
		jintArray arr = static_cast<jintArray>(env->GetObjectArrayElement(
				arrays, i));
		if (arr == nullptr)
			throw std::invalid_argument(
					"Input array " + std::to_string(i) + " cannot be null");
		const jint offset = offsets[i];
		const jint length = lengths[i];
		const jint array_length = env->GetArrayLength(arr);
		env->DeleteLocalRef(arr);
		if (offset < 0 || length <= 0 || offset + length > array_length)
			throw std::out_of_range(
					"Input array " + std::to_string(i)
							+ " has invalid offset/length");
	}
}

static std::vector<std::vector<llama_token>> copy_token_windows(JNIEnv *env,
		jobjectArray arrays, const std::vector<jint> &offsets,
		const std::vector<jint> &lengths) {
	check_array_windows(env, arrays, offsets, lengths);

	const int count = env->GetArrayLength(arrays);
	std::vector<std::vector<llama_token>> storage(count);
	for (int i = 0; i < count; i++) {
		jintArray arr = static_cast<jintArray>(env->GetObjectArrayElement(
				arrays, i));
		storage[i].resize(lengths[i]);
		env->GetIntArrayRegion(arr, offsets[i], lengths[i],
				reinterpret_cast<jint*>(storage[i].data()));
		env->DeleteLocalRef(arr);
	}
	return storage;
}

static std::vector<TokenSpan> spans_from_storage(
		const std::vector<std::vector<llama_token>> &storage) {
	std::vector<TokenSpan> spans(storage.size());
	for (size_t i = 0; i < storage.size(); i++) {
		spans[i].tokens = storage[i].data();
		spans[i].length = static_cast<int32_t>(storage[i].size());
	}
	return spans;
}

static std::vector<TokenSpan> direct_spans(JNIEnv *env, jobjectArray buffers,
		const std::vector<jint> &offsets, const std::vector<jint> &lengths) {
	const int count = env->GetArrayLength(buffers);
	if (static_cast<int>(offsets.size()) != count
			|| static_cast<int>(lengths.size()) != count)
		throw std::invalid_argument(
				"Input, offset, and length arrays must have the same size");

	std::vector<TokenSpan> spans(count);
	for (int i = 0; i < count; i++) {
		jobject buffer = env->GetObjectArrayElement(buffers, i);
		if (buffer == nullptr)
			throw std::invalid_argument(
					"Input buffer " + std::to_string(i) + " cannot be null");
		void *address = env->GetDirectBufferAddress(buffer);
		env->DeleteLocalRef(buffer);
		if (address == nullptr)
			throw std::invalid_argument(
					"Input buffer " + std::to_string(i)
							+ " is not a direct buffer");
		if (offsets[i] < 0 || lengths[i] <= 0)
			throw std::out_of_range(
					"Input buffer " + std::to_string(i)
							+ " has invalid offset/length");
		spans[i].tokens = static_cast<llama_token*>(address) + offsets[i];
		spans[i].length = lengths[i];
	}
	return spans;
}

static llama_pos write_spans(llama_context *ctx, llama_sampler *smpl,
		llama_pos cur_pos, const std::vector<TokenSpan> &inputs,
		const std::vector<llama_seq_id> &sequence_ids,
		std::vector<int32_t> &output_ids, bool last_logits) {
	validate_spans(inputs);
	const int inputs_count = static_cast<int>(inputs.size());
	const int n_parallel = static_cast<int>(sequence_ids.size());
	if (n_parallel <= 0)
		throw std::invalid_argument("At least one sequence id is required");

	const llama_model *model = llama_get_model(ctx);
	const llama_vocab *vocab = llama_model_get_vocab(model);

	PERF_BEGIN();
	if (inputs_count == 1) {
		const TokenSpan &input = inputs[0];

		llama_batch_owner batch_owner(
				std::max(input.length, n_parallel), 0, n_parallel);
		llama_batch &batch = batch_owner.get();

		for (int32_t i = 0; i < input.length; i++) {
			batch.token[batch.n_tokens] = input.tokens[i];
			batch.pos[batch.n_tokens] = cur_pos + i;
			batch.n_seq_id[batch.n_tokens] = n_parallel;
			for (int j = 0; j < n_parallel; j++)
				batch.seq_id[batch.n_tokens][j] = sequence_ids[j];
			batch.logits[batch.n_tokens] = false;
			batch.n_tokens++;
		}

		if (llama_model_has_encoder(model)) {
			if (llama_encode(ctx, batch) != 0)
				throw std::runtime_error("Encode failed");

			llama_token decoder_start_token_id =
					llama_model_decoder_start_token(model);
			if (decoder_start_token_id == -1)
				decoder_start_token_id = llama_vocab_bos(vocab);

			jjml_llm_batch_clear(batch);
			jjml_llm_batch_add(batch, decoder_start_token_id, cur_pos,
					sequence_ids, false);
		}

		if (last_logits) {
			batch.logits[batch.n_tokens - 1] = true;
			for (int i = 0; i < n_parallel; i++)
				output_ids[i] = batch.n_tokens - 1;
		}

		const int decode_status = llama_decode(ctx, batch);
		if (decode_status != 0)
			throw std::runtime_error(
					"Decode failed with status "
							+ std::to_string(decode_status));

		for (int i = 0; i < batch.n_tokens; i++)
			llama_sampler_accept(smpl, batch.token[i]);

		cur_pos = cur_pos + batch.n_tokens;
	} else {
		if (inputs_count != n_parallel)
			throw std::invalid_argument(
					"Input count must be one or match sequence count");

		int total_tokens = 0;
		for (const TokenSpan &input : inputs)
			total_tokens += input.length;

		llama_batch_owner batch_owner(total_tokens, 0, n_parallel);
		llama_batch &batch = batch_owner.get();

		for (int j = 0; j < n_parallel; j++) {
			const TokenSpan &input = inputs[j];
			for (int32_t i = 0; i < input.length; i++) {
				batch.token[batch.n_tokens] = input.tokens[i];
				batch.pos[batch.n_tokens] = cur_pos;
				batch.n_seq_id[batch.n_tokens] = 1;
				batch.seq_id[batch.n_tokens][0] = sequence_ids[j];
				batch.logits[batch.n_tokens] = false;
				batch.n_tokens++;
				cur_pos++;
			}

			if (last_logits) {
				batch.logits[batch.n_tokens - 1] = true;
				output_ids[j] = batch.n_tokens - 1;
			}
		}

		const int decode_status = llama_decode(ctx, batch);
		if (decode_status != 0)
			throw std::runtime_error(
					"Decode failed with status "
							+ std::to_string(decode_status));
	}
	PERF_END(__func__);
	return cur_pos;
}
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppBatchProcessor_doWrite(
		JNIEnv *env, jclass, jlong contextPointer, jlong samplerChainPointer,
		jint contextPosition, jobjectArray inputBuffers, jintArray offsets,
		jintArray lengths, jintArray sequenceIds, jintArray outputIds,
		jboolean lastLogits) {
	try {
		auto *ctx = argeo::jni::as_pointer<llama_context*>(contextPointer);
		auto *smpl = argeo::jni::as_pointer<llama_sampler*>(
				samplerChainPointer);
		std::vector<jint> offset_values = copy_int_array(env, offsets,
				"Input offsets");
		std::vector<jint> length_values = copy_int_array(env, lengths,
				"Input lengths");
		std::vector<TokenSpan> inputs = direct_spans(env, inputBuffers,
				offset_values, length_values);
		std::vector<llama_seq_id> sequence_ids = copy_sequence_ids(env,
				sequenceIds);
		std::vector<int32_t> output_ids = copy_output_ids(env, outputIds);

		const llama_pos new_position = write_spans(ctx, smpl, contextPosition,
				inputs, sequence_ids, output_ids, lastLogits);
		write_output_ids(env, outputIds, output_ids);
		return new_position;
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
		return contextPosition;
	}
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppBatchProcessor_doWriteArrays(
		JNIEnv *env, jclass, jlong contextPointer, jlong samplerChainPointer,
		jint contextPosition, jobjectArray inputArrays, jintArray offsets,
		jintArray lengths, jintArray sequenceIds, jintArray outputIds,
		jboolean lastLogits) {
	try {
		auto *ctx = argeo::jni::as_pointer<llama_context*>(contextPointer);
		auto *smpl = argeo::jni::as_pointer<llama_sampler*>(
				samplerChainPointer);
		std::vector<jint> offset_values = copy_int_array(env, offsets,
				"Input offsets");
		std::vector<jint> length_values = copy_int_array(env, lengths,
				"Input lengths");
		std::vector<std::vector<llama_token>> storage = copy_token_windows(env,
				inputArrays, offset_values, length_values);
		std::vector<TokenSpan> inputs = spans_from_storage(storage);
		std::vector<llama_seq_id> sequence_ids = copy_sequence_ids(env,
				sequenceIds);
		std::vector<int32_t> output_ids = copy_output_ids(env, outputIds);

		const llama_pos new_position = write_spans(ctx, smpl, contextPosition,
				inputs, sequence_ids, output_ids, lastLogits);
		write_output_ids(env, outputIds, output_ids);
		return new_position;
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
		return contextPosition;
	}
}
