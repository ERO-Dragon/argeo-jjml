package org.argeo.jjml.llm.params;

/** Names of the supported model parameters. */
public enum ModelParam {
	n_gpu_layers, //
	device, //
	split_mode, //
	main_gpu, //
	tensor_split, //
	vocab_only, //
	use_mmap, //
	use_direct_io, //
	use_mlock, //
	check_tensors, //
	use_extra_bufts, //
	no_host, //
	no_alloc, //
	;

	/**
	 * System property to set model parameters, such as the default number of layers
	 * offloaded to GPU.
	 */
	final static String SYSTEM_PROPERTY_MODEL_PARAM_PREFIX = "jjml.llm.model.";

	/** As a system property used to override default value. */
	public String asSystemProperty() {
		return SYSTEM_PROPERTY_MODEL_PARAM_PREFIX + name();
	}
}
