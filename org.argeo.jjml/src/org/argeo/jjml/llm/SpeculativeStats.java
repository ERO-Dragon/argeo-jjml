package org.argeo.jjml.llm;

/** Runtime counters for speculative decoding. */
public record SpeculativeStats(long promptTokens, long generatedTokens, long draftedTokens, long acceptedDraftTokens,
		long decodeNanos) {
	public double acceptanceRate() {
		return draftedTokens == 0 ? 0.0 : acceptedDraftTokens / (double) draftedTokens;
	}

	public double tokensPerSecond() {
		return decodeNanos == 0 ? 0.0 : generatedTokens * 1_000_000_000.0 / decodeNanos;
	}
}
