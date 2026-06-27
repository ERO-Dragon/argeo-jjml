package org.argeo.jjml.llm;

import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import org.argeo.jjml.llm.params.PoolingType;

/** Computes embeddings. */
public class LlamaCppEmbeddingProcessor {
	private final LlamaCppContext context;

	public LlamaCppEmbeddingProcessor(LlamaCppContext context) {
		this.context = context;
	}

	private static native void doProcessEmbeddings(long contextPointer, int[][] tokens, int[] offsets, int[] lengths,
			int maxSequences, float[] emb);

	public float[][] processEmbeddings(List<String> prompts) {
		Objects.requireNonNull(prompts, "Prompts cannot be null");
		IntBuffer[] tokenLists = new IntBuffer[prompts.size()];
		for (int i = 0; i < prompts.size(); i++) {
			String prompt = prompts.get(i);
			IntBuffer tokenList = context.getModel().getVocabulary().tokenize(prompt, true, true);
			tokenLists[i] = tokenList;
		}
		return processEmbeddings(tokenLists);
	}

	public float[][] processEmbeddings(IntBuffer[] inputs) {
		Objects.requireNonNull(inputs, "Embedding inputs cannot be null");
		if (!context.getInitParams().embeddings())
			throw new IllegalStateException("Context was not initialized for embeddings");
		if (inputs.length == 0)
			return new float[0][];

		EmbeddingInput[] embeddingInputs = prepareInputs(inputs);

		// logic taken from llama.cpp's examples/embedding
		PoolingType poolingType = context.getPoolingType();
		int n_embd_count = 0;
		if (PoolingType.LLAMA_POOLING_TYPE_NONE.equals(poolingType)) {
			for (EmbeddingInput input : embeddingInputs)
				n_embd_count += input.length;
		} else {
			n_embd_count = inputs.length;
		}

		int n_embd = context.getModel().getEmbeddingSize();

		float[] emb = new float[n_embd_count * n_embd];
		Arrays.fill(emb, 0);

		int[][] tokens = new int[inputs.length][];
		int[] offsets = new int[inputs.length];
		int[] lengths = new int[inputs.length];
		for (int i = 0; i < inputs.length; i++) {
			EmbeddingInput input = embeddingInputs[i];
			tokens[i] = input.tokens;
			offsets[i] = input.offset;
			lengths[i] = input.length;
		}
		int maxSequences = context.getInitParams().kv_unified() ? LlamaCppBackend.maxParallelSequences()
				: context.getMaxSequenceCount();
		doProcessEmbeddings(context.getAsLong(), tokens, offsets, lengths, maxSequences, emb);

		// TODO optimize storage, returned values, and copy
		float[][] res = new float[n_embd_count][];
		for (int j = 0;;) { // at least one iteration (one prompt)
			float[] arr = new float[n_embd];
			for (int i = 0;;) { // at least one iteration (n_embd > 0)
				arr[i] = emb[j * n_embd + i];
				i++;
				if (i == n_embd)
					break;
			}
			res[j] = arr;
			j++;
			if (j == n_embd_count)
				break;
		}
		return res;
	}

	private EmbeddingInput[] prepareInputs(IntBuffer[] inputs) {
		EmbeddingInput[] res = new EmbeddingInput[inputs.length];
		int batchSize = context.getBatchSize();
		for (int i = 0; i < inputs.length; i++) {
			IntBuffer input = Objects.requireNonNull(inputs[i], "Embedding input " + i + " cannot be null");
			int length = input.remaining();
			if (length == 0)
				throw new IllegalArgumentException("Embedding input " + i + " is empty");
			if (length > batchSize)
				throw new IllegalArgumentException(
						"Embedding input " + i + " has " + length + " tokens, exceeding batch size " + batchSize);
			if (input.hasArray() && !input.isReadOnly()) {
				res[i] = new EmbeddingInput(input.array(), input.arrayOffset() + input.position(), length);
			} else {
				int[] copy = new int[length];
				IntBuffer duplicate = input.duplicate();
				duplicate.get(copy);
				res[i] = new EmbeddingInput(copy, 0, length);
			}
		}
		return res;
	}

	private static final class EmbeddingInput {
		private final int[] tokens;
		private final int offset;
		private final int length;

		private EmbeddingInput(int[] tokens, int offset, int length) {
			this.tokens = tokens;
			this.offset = offset;
			this.length = length;
		}
	}

	/*
	 * ACCESSORS
	 */
	protected LlamaCppContext getContext() {
		return context;
	}
}
