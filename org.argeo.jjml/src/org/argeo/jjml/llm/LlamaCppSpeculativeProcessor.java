package org.argeo.jjml.llm;

import java.nio.IntBuffer;
import java.util.Objects;
import java.util.function.LongSupplier;

/**
 * Processor using llama.cpp speculative decoding. Currently exposes MTP draft
 * decoding as an explicit opt-in path.
 */
public class LlamaCppSpeculativeProcessor implements LongSupplier, AutoCloseable {
	private final LlamaCppContext context;
	private final LlamaCppModel draftModel;
	private final LlamaCppSamplerChain samplerChain;
	private final long pointer;

	private boolean closed = false;
	private boolean begun = false;

	public LlamaCppSpeculativeProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain,
			SpeculativeParams params) {
		this(context, null, samplerChain, params);
	}

	/**
	 * Create a speculative processor with an explicit draft model.
	 * <p>
	 * This is required by models such as Gemma 4 Assistant MTP where the draft
	 * head is distributed as a separate GGUF next to the target model.
	 */
	public LlamaCppSpeculativeProcessor(LlamaCppContext context, LlamaCppModel draftModel,
			LlamaCppSamplerChain samplerChain, SpeculativeParams params) {
		this.context = Objects.requireNonNull(context);
		this.draftModel = draftModel;
		this.samplerChain = Objects.requireNonNull(samplerChain);
		params = Objects.requireNonNull(params);
		if (!params.isEnabled())
			throw new IllegalArgumentException("Speculative decoding is disabled by the given parameters");
		if (params.isMtp() && draftModel == null && !context.getModel().supportsMtp())
			throw new IllegalArgumentException("Model does not expose MTP/NextN layers. Expected GGUF metadata "
					+ context.getModel().getMetadata().getOrDefault("general.architecture", "<arch>")
					+ ".nextn_predict_layers to be greater than zero.");
		if (params.isMtp()) {
			int requiredOutputs = params.requiredTargetOutputs();
			int actualOutputs = context.getInitParams().n_outputs_max();
			if (actualOutputs != 0 && actualOutputs < requiredOutputs)
				throw new IllegalArgumentException(
						"MTP target context requires n_outputs_max >= " + requiredOutputs
								+ ". Use SpeculativeParams.adjustTargetContextParams(...) to lift a single-output baseline "
								+ "(for example n_outputs_max = 1).");
			if (context.getBatchSize() < requiredOutputs)
				throw new IllegalArgumentException("MTP target context requires n_batch >= " + requiredOutputs
						+ ". Create the context with SpeculativeParams.adjustTargetContextParams(...).");
			if (context.getInitParams().n_ubatch() < requiredOutputs)
				throw new IllegalArgumentException("MTP target context requires n_ubatch >= " + requiredOutputs
						+ ". Create the context with SpeculativeParams.adjustTargetContextParams(...).");
			if (context.getInitParams().n_rs_seq() < params.draftMax())
				throw new IllegalArgumentException("MTP target context requires n_rs_seq >= " + params.draftMax()
						+ ". Create the context with SpeculativeParams.adjustTargetContextParams(...).");
		}
		this.pointer = doInit(context.getAsLong(), draftModel != null ? draftModel.getAsLong() : 0L, params);
	}

	private static native long doInit(long contextPointer, long draftModelPointer, SpeculativeParams params);

	private native void doDestroy();

	private native void doBegin(long samplerChainPointer, int[] promptTokens, int offset, int length);

	private native void doBeginFromRestoredTarget(long samplerChainPointer, int[] promptTokens, int offset, int length,
			int restoredTokenCount);

	private native int doRead(long samplerChainPointer, IntBuffer output, int offset, int length);

	private native int[] doReadArray(long samplerChainPointer, int maxTokens);

	private native long[] doGetStats();

	public synchronized void begin(IntBuffer promptTokens) {
		Objects.requireNonNull(promptTokens);
		checkOpen();
		if (begun)
			throw new IllegalStateException("Speculative generation has already begun");
		TokenArrayView tokens = consumeTokens(promptTokens);
		doBegin(samplerChain.getAsLong(), tokens.array, tokens.offset, tokens.length);
		begun = true;
	}

	/**
	 * Begin speculative generation after the target context has already been restored
	 * to contain the first {@code restoredTokenCount} tokens from {@code promptTokens}.
	 * The remaining tokens are replayed through the target context so that the MTP
	 * draft context and hidden-state bridge are rebuilt without replaying the whole
	 * prompt. Callers should keep a short overlap after the saved context position;
	 * an empty tail cannot rebuild the MTP draft state.
	 */
	public synchronized void beginFromRestoredTarget(IntBuffer promptTokens, int restoredTokenCount) {
		Objects.requireNonNull(promptTokens);
		checkOpen();
		if (begun)
			throw new IllegalStateException("Speculative generation has already begun");
		TokenArrayView tokens = consumeTokens(promptTokens);
		if (restoredTokenCount < 0)
			throw new IllegalArgumentException("restoredTokenCount must not be negative");
		if (restoredTokenCount >= tokens.length)
			throw new IllegalArgumentException("restoredTokenCount must leave at least one token to replay");
		doBeginFromRestoredTarget(samplerChain.getAsLong(), tokens.array, tokens.offset, tokens.length,
				restoredTokenCount);
		begun = true;
	}

	public synchronized int read(IntBuffer output) {
		Objects.requireNonNull(output);
		checkOpen();
		checkBegun();
		if (output.isReadOnly())
			throw new IllegalArgumentException("Output buffer is read-only");
		if (!output.isDirect()) {
			int[] arr = doReadArray(samplerChain.getAsLong(), output.remaining());
			output.put(arr);
			return arr.length;
		}
		int offset = output.position();
		int read = doRead(samplerChain.getAsLong(), output, offset, output.remaining());
		output.position(offset + read);
		return read;
	}

	public synchronized int[] readArray(int maxTokens) {
		checkOpen();
		checkBegun();
		if (maxTokens < 0)
			throw new IllegalArgumentException("maxTokens must not be negative");
		return doReadArray(samplerChain.getAsLong(), maxTokens);
	}

	public synchronized SpeculativeStats getStats() {
		checkOpen();
		long[] stats = doGetStats();
		return new SpeculativeStats(stats[0], stats[1], stats[2], stats[3], stats[4]);
	}

	@Override
	public long getAsLong() {
		checkOpen();
		return pointer;
	}

	public LlamaCppContext getContext() {
		return context;
	}

	/**
	 * Return the explicit draft model, or {@code null} when the target model itself
	 * supplies the MTP layers.
	 */
	public LlamaCppModel getDraftModel() {
		return draftModel;
	}

	@Override
	public synchronized void close() {
		if (!closed) {
			doDestroy();
			closed = true;
		}
	}

	private void checkOpen() {
		if (closed)
			throw new IllegalStateException("Speculative processor is closed");
	}

	private void checkBegun() {
		if (!begun)
			throw new IllegalStateException("Call begin(...) before reading generated tokens");
	}

	private static TokenArrayView consumeTokens(IntBuffer tokens) {
		int length = tokens.remaining();
		if (tokens.hasArray() && !tokens.isReadOnly()) {
			int[] arr = tokens.array();
			int offset = tokens.arrayOffset() + tokens.position();
			tokens.position(tokens.limit());
			return new TokenArrayView(arr, offset, length);
		}
		int[] arr = new int[length];
		tokens.get(arr);
		return new TokenArrayView(arr, 0, length);
	}

	private static final class TokenArrayView {
		final int[] array;
		final int offset;
		final int length;

		TokenArrayView(int[] array, int offset, int length) {
			this.array = array;
			this.offset = offset;
			this.length = length;
		}
	}
}
