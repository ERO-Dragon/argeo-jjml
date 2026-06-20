package org.argeo.jjml.llm.params;

import java.util.function.IntSupplier;

/**
 * Flash Attention mode. (see enum <code>llama_flash_attn_type</code>, in
 * llama.h)
 */
public enum FlashAttentionType implements IntSupplier {
	LLAMA_FLASH_ATTN_TYPE_AUTO(-1), //
	LLAMA_FLASH_ATTN_TYPE_DISABLED(0), //
	LLAMA_FLASH_ATTN_TYPE_ENABLED(1), //
	;

	private final int code;

	private FlashAttentionType(int code) {
		this.code = code;
	}

	@Override
	public int getAsInt() {
		return code;
	}

	public static FlashAttentionType byCode(int code) throws IllegalArgumentException {
		for (FlashAttentionType type : values())
			if (type.code == code)
				return type;
		throw new IllegalArgumentException("Unknown flash attention type code : " + code);
	}
}
