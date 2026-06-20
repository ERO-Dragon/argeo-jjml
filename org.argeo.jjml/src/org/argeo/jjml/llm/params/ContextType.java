package org.argeo.jjml.llm.params;

import java.util.function.IntSupplier;

/**
 * Context type. (see enum <code>llama_context_type</code>, in llama.h)
 */
public enum ContextType implements IntSupplier {
	LLAMA_CONTEXT_TYPE_DEFAULT(0), //
	LLAMA_CONTEXT_TYPE_MTP(1), //
	;

	private final int code;

	private ContextType(int code) {
		this.code = code;
	}

	@Override
	public int getAsInt() {
		return code;
	}

	public static ContextType byCode(int code) throws IllegalArgumentException {
		for (ContextType type : values())
			if (type.code == code)
				return type;
		throw new IllegalArgumentException("Unknown context type code : " + code);
	}
}
