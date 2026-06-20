package org.argeo.jjml.mtmd;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.nio.IntBuffer;
import java.util.Objects;

import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppSamplerChain;

public class MtmdProcessor {
	private final LlamaCppContext context;
	private final LlamaCppSamplerChain chain;
	private final MtmdContext mtmdContext;

	public MtmdProcessor(LlamaCppContext context, LlamaCppSamplerChain chain, MtmdContext mtmdContext) {
		this.context = Objects.requireNonNull(context);
		this.chain = Objects.requireNonNull(chain);
		this.mtmdContext = Objects.requireNonNull(mtmdContext);
	}

	private static native int[] doSingleTurn(long contextPointer, long samplerPointer, long mtmdContextPointer,
			byte[] prompt, MtmdBitmap[] bitmaps);

	public String transcribe(String prompt, MtmdBitmap[] bitmaps) {
		Objects.requireNonNull(prompt);
		if (bitmaps == null)
			bitmaps = new MtmdBitmap[0];
		int[] tokens = doSingleTurn(context.getAsLong(), chain.getAsLong(), mtmdContext.getAsLong(),
				prompt.getBytes(UTF_8), bitmaps);
		String response = context.getModel().getVocabulary().deTokenize(IntBuffer.wrap(tokens));
		return response;
	}
}
