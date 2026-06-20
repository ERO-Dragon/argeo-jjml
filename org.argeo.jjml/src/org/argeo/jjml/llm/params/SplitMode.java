package org.argeo.jjml.llm.params;

import java.util.function.IntSupplier;

/**
 * Model split mode. (see enum <code>llama_split_mode</code>, in llama.h)
 */
public enum SplitMode implements IntSupplier {
	LLAMA_SPLIT_MODE_NONE(0), //
	LLAMA_SPLIT_MODE_LAYER(1), //
	LLAMA_SPLIT_MODE_ROW(2), //
	LLAMA_SPLIT_MODE_TENSOR(3), //
	;

	private final int code;

	private SplitMode(int code) {
		this.code = code;
	}

	@Override
	public int getAsInt() {
		return code;
	}

	public static SplitMode byCode(int code) throws IllegalArgumentException {
		for (SplitMode mode : values())
			if (mode.code == code)
				return mode;
		throw new IllegalArgumentException("Unknown split mode code : " + code);
	}
}
