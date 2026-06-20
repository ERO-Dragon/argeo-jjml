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
	private final LlamaCppSamplerChain samplerChain;
	private final long pointer;

	private boolean closed = false;
	private boolean begun = false;

	public LlamaCppSpeculativeProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain,
			SpeculativeParams params) {
		this.context = Objects.requireNonNull(context);
		this.samplerChain = Objects.requireNonNull(samplerChain);
		params = Objects.requireNonNull(params);
		if (!params.isEnabled())
			throw new IllegalArgumentException("Speculative decoding is disabled by the given parameters");
		if (params.isMtp() && !context.getModel().supportsMtp())
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
		this.pointer = doInit(context.getAsLong(), params);
	}

	private static native long doInit(long contextPointer, SpeculativeParams params);

	private native void doDestroy();

	private native void doBegin(long samplerChainPointer, int[] promptTokens, int offset, int length);

	private native int doRead(long samplerChainPointer, IntBuffer output, int offset, int length);

	private native int[] doReadArray(long samplerChainPointer, int maxTokens);

	private native long[] doGetStats();

	public synchronized void begin(IntBuffer promptTokens) {
		Objects.requireNonNull(promptTokens);
		checkOpen();
		if (begun)
			throw new IllegalStateException("Speculative generation has already begun");
		int[] arr;
		int offset;
		int length = promptTokens.remaining();
		if (promptTokens.hasArray() && !promptTokens.isReadOnly()) {
			arr = promptTokens.array();
			offset = promptTokens.arrayOffset() + promptTokens.position();
			promptTokens.position(promptTokens.limit());
		} else {
			arr = new int[length];
			promptTokens.get(arr);
			offset = 0;
		}
		doBegin(samplerChain.getAsLong(), arr, offset, length);
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
}
