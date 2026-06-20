package org.argeo.jjml.llm.params;

/** Supported parameters. */
public enum ContextParam {
	n_ctx, //
	n_batch, //
	n_ubatch, //
	n_seq_max, //
	n_rs_seq, //
	n_outputs_max, //
	n_threads, //
	n_threads_batch, //
	ctx_type, //
	rope_scaling_type, //
	pooling_type, //
	attention_type, //
	flash_attn_type, //
	rope_freq_base, //
	rope_freq_scale, //
	yarn_ext_factor, //
	yarn_attn_factor, //
	yarn_beta_fast, //
	yarn_beta_slow, //
	yarn_orig_ctx, //
	defrag_thold, //
	type_k, // only Q4_0 and Q8_0 supported at this stage
	type_v, // only Q4_0 and Q8_0 supported at this stage
	embeddings, //
	offload_kqv, //
	flash_attn, // legacy boolean alias for flash_attn_type
	no_perf, //
	op_offload, //
	swa_full, //
	kv_unified, //
	;

	/** System property to set the default context size. */
	final static String SYSTEM_PROPERTY_CONTEXT_PARAM_PREFIX = "jjml.llm.context.";

	/** As a system property used to override default value. */
	public String asSystemProperty() {
		return SYSTEM_PROPERTY_CONTEXT_PARAM_PREFIX + name();
	}

}
