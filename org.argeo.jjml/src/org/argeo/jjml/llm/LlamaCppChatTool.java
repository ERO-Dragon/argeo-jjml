package org.argeo.jjml.llm;

import java.util.Objects;

/** OpenAI-compatible function tool definition for llama.cpp chat templates. */
public record LlamaCppChatTool(String name, String description, String parametersJson) {
	public LlamaCppChatTool {
		Objects.requireNonNull(name);
		Objects.requireNonNull(description);
		Objects.requireNonNull(parametersJson);
		if (name.isBlank())
			throw new IllegalArgumentException("Tool name must not be blank");
		if (parametersJson.isBlank())
			throw new IllegalArgumentException("Tool parameters JSON must not be blank");
	}
}
