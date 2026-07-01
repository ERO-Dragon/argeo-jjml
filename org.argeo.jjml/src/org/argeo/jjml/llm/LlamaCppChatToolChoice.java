package org.argeo.jjml.llm;

/** Tool-call policy understood by llama.cpp's OpenAI-compatible formatter. */
public enum LlamaCppChatToolChoice {
	AUTO("auto"), //
	REQUIRED("required"), //
	NONE("none");

	private final String llamaName;

	LlamaCppChatToolChoice(String llamaName) {
		this.llamaName = llamaName;
	}

	String llamaName() {
		return llamaName;
	}
}
