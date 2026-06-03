package org.argeo.jjml.llm.util;

/**
 * Controls the thinking/reasoning mode for chat templates that support it
 * (e.g. Qwen3, DeepSeek-R1).
 * <p>
 * When the model's template does not support thinking, this setting has no
 * effect and thinking is implicitly disabled.
 * </p>
 */
public enum ThinkingMode {
	/** Enable thinking only if the model's template supports it. */
	AUTO,
	/** Pass enable_thinking=true to the chat template. */
	ENABLED,
	/** Force thinking off. */
	DISABLED,
}
