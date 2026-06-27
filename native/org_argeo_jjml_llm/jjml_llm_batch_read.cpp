#include <math.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <llama.h>

#include <argeo/jni/argeo_jni.h>

#include "org_argeo_jjml_llm_LlamaCppBatchProcessor.h" // IWYU pragma: keep
#include "jjml_llm.h"
#include "org_argeo_jjml_llm_.h"

namespace {
struct TokenOutput {
	llama_token *tokens = nullptr;
	int32_t capacity = 0;
	jintArray java_array = nullptr;
	jint java_offset = 0;
	bool copy_to_java = false;
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

class output_refs_owner {
public:
	output_refs_owner(JNIEnv *env, std::vector<TokenOutput> &outputs) :
			env(env), outputs(outputs) {
	}

	~output_refs_owner() {
		for (TokenOutput &output : outputs) {
			if (output.java_array != nullptr) {
				env->DeleteLocalRef(output.java_array);
				output.java_array = nullptr;
			}
		}
	}

private:
	JNIEnv *env;
	std::vector<TokenOutput> &outputs;
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
	if (!values.empty())
		env->SetIntArrayRegion(output_ids, 0, static_cast<jsize>(values.size()),
				reinterpret_cast<const jint*>(values.data()));
}

static void write_output_id(JNIEnv *env, jintArray output_ids, int index,
		int32_t value) {
	jint java_value = static_cast<jint>(value);
	env->SetIntArrayRegion(output_ids, index, 1, &java_value);
}

static std::vector<TokenOutput> direct_outputs(JNIEnv *env,
		jobjectArray buffers, const std::vector<jint> &offsets,
		const std::vector<jint> &lengths) {
	const int count = env->GetArrayLength(buffers);
	if (static_cast<int>(offsets.size()) != count
			|| static_cast<int>(lengths.size()) != count)
		throw std::invalid_argument(
				"Output, offset, and length arrays must have the same size");

	std::vector<TokenOutput> outputs(count);
	for (int i = 0; i < count; i++) {
		jobject buffer = env->GetObjectArrayElement(buffers, i);
		if (buffer == nullptr) {
			outputs[i] = { nullptr, 0, nullptr, 0, false };
			continue;
		}
		void *address = env->GetDirectBufferAddress(buffer);
		env->DeleteLocalRef(buffer);
		if (address == nullptr)
			throw std::invalid_argument(
					"Output buffer " + std::to_string(i)
							+ " is not a direct buffer");
		if (offsets[i] < 0 || lengths[i] < 0)
			throw std::out_of_range(
					"Output buffer " + std::to_string(i)
							+ " has invalid offset/length");
		outputs[i].tokens = static_cast<llama_token*>(address) + offsets[i];
		outputs[i].capacity = lengths[i];
	}
	return outputs;
}

static std::vector<std::vector<llama_token>> allocate_output_storage(
		int count, const std::vector<jint> &lengths) {
	std::vector<std::vector<llama_token>> storage(count);
	for (int i = 0; i < count; i++)
		if (lengths[i] > 0)
			storage[i].resize(lengths[i]);
	return storage;
}

static std::vector<TokenOutput> outputs_from_storage(JNIEnv *env,
		jobjectArray arrays, const std::vector<jint> &offsets,
		std::vector<std::vector<llama_token>> &storage) {
	std::vector<TokenOutput> outputs(storage.size());
	for (size_t i = 0; i < storage.size(); i++) {
		outputs[i].tokens = storage[i].empty() ? nullptr : storage[i].data();
		outputs[i].capacity = static_cast<int32_t>(storage[i].size());
		if (!storage[i].empty()) {
			outputs[i].java_array = static_cast<jintArray>(
					env->GetObjectArrayElement(arrays, i));
			outputs[i].java_offset = offsets[i];
			outputs[i].copy_to_java = true;
		}
	}
	return outputs;
}

static void validate_output_arrays(JNIEnv *env, jobjectArray arrays,
		const std::vector<jint> &offsets, const std::vector<jint> &lengths) {
	const int count = env->GetArrayLength(arrays);
	if (static_cast<int>(offsets.size()) != count
			|| static_cast<int>(lengths.size()) != count)
		throw std::invalid_argument(
				"Output, offset, and length arrays must have the same size");
	for (int i = 0; i < count; i++) {
		jintArray arr = static_cast<jintArray>(env->GetObjectArrayElement(
				arrays, i));
		if (arr == nullptr)
			throw std::invalid_argument(
					"Output array " + std::to_string(i) + " cannot be null");
		const jint array_length = env->GetArrayLength(arr);
		env->DeleteLocalRef(arr);
		if (offsets[i] < 0 || lengths[i] < 0
				|| offsets[i] + lengths[i] > array_length)
			throw std::out_of_range(
					"Output array " + std::to_string(i)
							+ " has invalid offset/length");
	}
}

static std::vector<llama_token_data> jjml_get_logits(llama_context *ctx,
		int idx) {
	const auto *logits = llama_get_logits_ith(ctx, idx);

	const llama_vocab *vocab = llama_model_get_vocab(llama_get_model(ctx));
	const int n_vocab = llama_vocab_n_tokens(vocab);

	std::vector<llama_token_data> cur;
	cur.resize(n_vocab);

	for (llama_token token_id = 0; token_id < n_vocab; token_id++)
		cur[token_id] = llama_token_data { token_id, logits[token_id], 0.0f };

	return cur;
}

static llama_token jjml_check_grammar(llama_context *ctx, int idx,
		llama_sampler *chain, llama_sampler *grmr, llama_token id) {
	{
		llama_token_data single_token_data = { id, 1.0f, 0.0f };
		llama_token_data_array single_token_data_array = { &single_token_data,
				1, -1, false };

		llama_sampler_apply(grmr, &single_token_data_array);

		const bool is_valid = single_token_data_array.data[0].logit
				!= -INFINITY;
		if (is_valid)
			return id;
	}

	std::vector<llama_token_data> cur = jjml_get_logits(ctx, idx);
	llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false, };

	llama_sampler_apply(grmr, &cur_p);
	llama_sampler_apply(chain, &cur_p);

	if (cur_p.selected == -1)
		throw std::runtime_error(
				"No selected token during grammar re-sampling");

	return cur_p.data[cur_p.selected].id;
}

static void notify_completed(JNIEnv *env, jobject completion_handler,
		int result, int sequence_index) {
	jclass Integer = argeo::jni::find_jclass(env, "java/lang/Integer");
	jobject completionHandlerResult = env->CallStaticObjectMethod(Integer,
			Integer__valueOf, result);
	jobject completionHandlerAttachment = env->CallStaticObjectMethod(Integer,
			Integer__valueOf, sequence_index);
	env->CallVoidMethod(completion_handler, CompletionHandler__completed,
			completionHandlerResult, completionHandlerAttachment);
	env->DeleteLocalRef(completionHandlerResult);
	env->DeleteLocalRef(completionHandlerAttachment);
	env->DeleteLocalRef(Integer);
	if (env->ExceptionCheck())
		throw std::runtime_error("Java completion handler failed");
}

static void publish_output(JNIEnv *env, const TokenOutput &output,
		int written) {
	if (!output.copy_to_java || written <= 0)
		return;
	env->SetIntArrayRegion(output.java_array, output.java_offset, written,
			reinterpret_cast<const jint*>(output.tokens));
	if (env->ExceptionCheck())
		throw std::runtime_error("Failed to copy generated tokens to Java");
}

static llama_pos read_outputs(llama_context *ctx, llama_sampler *smpl,
		llama_sampler *grmr, llama_pos cur_pos,
		const std::vector<TokenOutput> &outputs,
		const std::vector<llama_seq_id> &sequence_ids,
		std::vector<int32_t> &output_ids, std::vector<int> &written,
		JNIEnv *env, jintArray output_ids_array, jobject completion_handler) {
	const llama_vocab *vocab = llama_model_get_vocab(llama_get_model(ctx));
	const uint32_t no_output_id = llama_n_batch(ctx);

	const int n_parallel = static_cast<int>(sequence_ids.size());
	if (n_parallel <= 0)
		throw std::invalid_argument("At least one sequence id is required");
	if (static_cast<int>(outputs.size()) != n_parallel
			|| static_cast<int>(output_ids.size()) != n_parallel)
		throw std::invalid_argument(
				"Output count must match sequence count");

	int max_decodes = 0;
	for (const TokenOutput &output : outputs)
		if (output.capacity > max_decodes)
			max_decodes = output.capacity;

	PERF_BEGIN();
	int next_idx = 0;
	std::vector<bool> completed(outputs.size(), false);
	llama_batch_owner batch_owner(n_parallel, 0, 1);
	llama_batch &batch = batch_owner.get();

	while (true) {
		jjml_llm_batch_clear(batch);

		for (int32_t i = 0; i < n_parallel; ++i) {
			if (outputs[i].tokens == nullptr)
				continue;
			if (completed[i])
				continue;
			if (output_ids[i] == static_cast<int32_t>(no_output_id))
				continue;
			if (next_idx >= outputs[i].capacity) {
				completed[i] = true;
				written[i] = next_idx;
				publish_output(env, outputs[i], next_idx);
				write_output_id(env, output_ids_array, i, output_ids[i]);
				notify_completed(env, completion_handler, next_idx, i);
				continue;
			}

			PERF_BEGIN();
			llama_token new_token_id;
			if (grmr == nullptr) {
				new_token_id = llama_sampler_sample(smpl, ctx, output_ids[i]);
			} else {
				std::vector<llama_token_data> cur = jjml_get_logits(ctx,
						output_ids[i]);
				llama_token_data_array cur_p = { cur.data(), cur.size(), -1,
						false, };

				llama_sampler_apply(smpl, &cur_p);
				if (cur_p.selected == -1)
					throw std::runtime_error("Sampling produced no token");
				llama_token candidate = cur_p.data[cur_p.selected].id;
				new_token_id = jjml_check_grammar(ctx, output_ids[i], smpl,
						grmr, candidate);

				llama_sampler_accept(grmr, new_token_id);
				llama_sampler_accept(smpl, new_token_id);
			}
			PERF_END("sampling");

			const bool is_eog = llama_vocab_is_eog(vocab, new_token_id);

			if (is_eog) {
				if (is_eog)
					output_ids[i] = no_output_id;
				completed[i] = true;
				written[i] = next_idx;
				publish_output(env, outputs[i], next_idx);
				write_output_id(env, output_ids_array, i, output_ids[i]);
				notify_completed(env, completion_handler, next_idx, i);
				continue;
			}

			outputs[i].tokens[next_idx] = new_token_id;
			output_ids[i] = batch.n_tokens;
			jjml_llm_batch_add(batch, new_token_id, cur_pos,
					{ sequence_ids[i] }, true);
		}
		next_idx++;

		if (batch.n_tokens == 0)
			break;

		cur_pos += 1;
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

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppBatchProcessor_doRead(
		JNIEnv *env, jclass, jlong contextPointer, jlong samplerPtr,
		jlong grammarSamplerPtr, jint contextPosition,
		jobjectArray outputBuffers, jintArray offsets, jintArray lengths,
		jintArray sequenceIds, jintArray outputIds, jobject completionHandler) {
	try {
		auto *ctx = argeo::jni::as_pointer<llama_context*>(contextPointer);
		auto *smpl = argeo::jni::as_pointer<llama_sampler*>(samplerPtr);
		auto *grmr =
				grammarSamplerPtr != 0 ?
						argeo::jni::as_pointer<llama_sampler*>(
								grammarSamplerPtr) :
						nullptr;
		std::vector<jint> offset_values = copy_int_array(env, offsets,
				"Output offsets");
		std::vector<jint> length_values = copy_int_array(env, lengths,
				"Output lengths");
		std::vector<TokenOutput> outputs = direct_outputs(env, outputBuffers,
				offset_values, length_values);
		std::vector<llama_seq_id> sequence_ids = copy_sequence_ids(env,
				sequenceIds);
		std::vector<int32_t> output_ids = copy_output_ids(env, outputIds);
		std::vector<int> written(outputs.size(), 0);

		const llama_pos new_position = read_outputs(ctx, smpl, grmr,
				contextPosition, outputs, sequence_ids, output_ids, written,
				env, outputIds, completionHandler);
		write_output_ids(env, outputIds, output_ids);
		return new_position;
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
		return contextPosition;
	}
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppBatchProcessor_doReadToArrays(
		JNIEnv *env, jclass, jlong contextPointer, jlong samplerPtr,
		jlong grammarSamplerPtr, jint contextPosition,
		jobjectArray outputArrays, jintArray offsets, jintArray lengths,
		jintArray sequenceIds, jintArray outputIds, jobject completionHandler) {
	try {
		auto *ctx = argeo::jni::as_pointer<llama_context*>(contextPointer);
		auto *smpl = argeo::jni::as_pointer<llama_sampler*>(samplerPtr);
		auto *grmr =
				grammarSamplerPtr != 0 ?
						argeo::jni::as_pointer<llama_sampler*>(
								grammarSamplerPtr) :
						nullptr;
		std::vector<jint> offset_values = copy_int_array(env, offsets,
				"Output offsets");
		std::vector<jint> length_values = copy_int_array(env, lengths,
				"Output lengths");
		validate_output_arrays(env, outputArrays, offset_values,
				length_values);
		std::vector<std::vector<llama_token>> storage =
				allocate_output_storage(env->GetArrayLength(outputArrays),
						length_values);
		std::vector<TokenOutput> outputs = outputs_from_storage(env,
				outputArrays, offset_values, storage);
		output_refs_owner output_refs(env, outputs);
		std::vector<llama_seq_id> sequence_ids = copy_sequence_ids(env,
				sequenceIds);
		std::vector<int32_t> output_ids = copy_output_ids(env, outputIds);
		std::vector<int> written(outputs.size(), 0);

		const llama_pos new_position = read_outputs(ctx, smpl, grmr,
				contextPosition, outputs, sequence_ids, output_ids, written,
				env, outputIds, completionHandler);
		write_output_ids(env, outputIds, output_ids);
		return new_position;
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
		return contextPosition;
	}
}
