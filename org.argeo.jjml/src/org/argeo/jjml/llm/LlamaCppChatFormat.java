package org.argeo.jjml.llm;

import java.util.Objects;

/** Result of applying a llama.cpp chat template. */
public record LlamaCppChatFormat(String prompt, String grammar, boolean grammarLazy, String generationPrompt,
		boolean supportsThinking, String thinkingStartTag, String thinkingEndTag, LlamaCppGrammarTrigger[] grammarTriggers,
		String parser) {
	public LlamaCppChatFormat {
		Objects.requireNonNull(prompt);
		grammar = grammar != null ? grammar : "";
		generationPrompt = generationPrompt != null ? generationPrompt : "";
		thinkingStartTag = thinkingStartTag != null ? thinkingStartTag : "";
		thinkingEndTag = thinkingEndTag != null ? thinkingEndTag : "";
		grammarTriggers = grammarTriggers != null ? grammarTriggers : new LlamaCppGrammarTrigger[0];
		parser = parser != null ? parser : "";
	}
}
