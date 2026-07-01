package org.argeo.jjml.llm;

/** Trigger used by llama.cpp lazy grammar samplers. */
public record LlamaCppGrammarTrigger(int type, String value, int token) {
	public static final int TOKEN = 0;
	public static final int WORD = 1;
	public static final int PATTERN = 2;
	public static final int PATTERN_FULL = 3;

	public LlamaCppGrammarTrigger {
		value = value != null ? value : "";
	}
}
