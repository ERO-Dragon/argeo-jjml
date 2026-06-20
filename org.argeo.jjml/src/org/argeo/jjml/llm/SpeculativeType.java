package org.argeo.jjml.llm;

/** Speculative decoding implementations exposed by JJML. */
enum SpeculativeType {
	NONE(0),
	DRAFT_MTP(3);

	private final int code;

	SpeculativeType(int code) {
		this.code = code;
	}

	int code() {
		return code;
	}
}
