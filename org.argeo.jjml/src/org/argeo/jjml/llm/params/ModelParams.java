package org.argeo.jjml.llm.params;

import static java.lang.Boolean.parseBoolean;
import static java.lang.Integer.parseInt;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.function.IntSupplier;

import org.argeo.jjml.llm.LlamaCppModel;

/**
 * Initialization parameters of a model. New instance should be created by using
 * the {@link #with(Map)} methods on {@link LlamaCppModel#defaultModelParams()},
 * with the default values populated by the shared library.
 * 
 * Note: it provides record-style getters, in order to ease transition to Java
 * records in the future.
 * 
 * (see <code>llama_model_params</code>, in llama.h)
 */
public class ModelParams {
	private final int n_gpu_layers;
	private final String device;
	private final int split_mode;
	private final int main_gpu;
	private final String tensor_split;
	private final boolean vocab_only;
	private final boolean use_mmap;
	private final boolean use_direct_io;
	private final boolean use_mlock;
	private final boolean check_tensors;
	private final boolean use_extra_bufts;
	private final boolean no_host;
	private final boolean no_alloc;

	/**
	 * Record-like full constructor. Will be called by the native side to provide
	 * the defaults.
	 */
	ModelParams( //
			int n_gpu_layers, //
			String device, //
			int split_mode, //
			int main_gpu, //
			String tensor_split, //
			boolean vocab_only, //
			boolean use_mmap, //
			boolean use_direct_io, //
			boolean use_mlock, //
			boolean check_tensors, //
			boolean use_extra_bufts, //
			boolean no_host, //
			boolean no_alloc //
	) {
		this.n_gpu_layers = n_gpu_layers;
		this.device = device;
		this.split_mode = split_mode;
		this.main_gpu = main_gpu;
		this.tensor_split = tensor_split;
		this.vocab_only = vocab_only;
		this.use_mmap = use_mmap;
		this.use_direct_io = use_direct_io;
		this.use_mlock = use_mlock;
		this.check_tensors = check_tensors;
		this.use_extra_bufts = use_extra_bufts;
		this.no_host = no_host;
		this.no_alloc = no_alloc;
	}

	public ModelParams with(ModelParam key, Object value) {
		Objects.requireNonNull(key);
		Objects.requireNonNull(value);
		String str;
		if (value instanceof IntSupplier) {
			str = Integer.toString(((IntSupplier) value).getAsInt());
		} else {
			str = value.toString();
		}
		return with(Collections.singletonMap(key, str));
	}

	public ModelParams with(Map<ModelParam, String> p) {
		return new ModelParams( //
				parseInt(p.getOrDefault(ModelParam.n_gpu_layers, Integer.toString(this.n_gpu_layers))), //
				p.getOrDefault(ModelParam.device, this.device), //
				parseInt(p.getOrDefault(ModelParam.split_mode, Integer.toString(this.split_mode))), //
				parseInt(p.getOrDefault(ModelParam.main_gpu, Integer.toString(this.main_gpu))), //
				p.getOrDefault(ModelParam.tensor_split, this.tensor_split), //
				parseBoolean(p.getOrDefault(ModelParam.vocab_only, Boolean.toString(this.vocab_only))), //
				parseBoolean(p.getOrDefault(ModelParam.use_mmap, Boolean.toString(this.use_mmap))), //
				parseBoolean(p.getOrDefault(ModelParam.use_direct_io, Boolean.toString(this.use_direct_io))), //
				parseBoolean(p.getOrDefault(ModelParam.use_mlock, Boolean.toString(this.use_mlock))), //
				parseBoolean(p.getOrDefault(ModelParam.check_tensors, Boolean.toString(this.check_tensors))), //
				parseBoolean(p.getOrDefault(ModelParam.use_extra_bufts, Boolean.toString(this.use_extra_bufts))), //
				parseBoolean(p.getOrDefault(ModelParam.no_host, Boolean.toString(this.no_host))), //
				parseBoolean(p.getOrDefault(ModelParam.no_alloc, Boolean.toString(this.no_alloc))) //
		);
	}

	public int n_gpu_layers() {
		return n_gpu_layers;
	}

	public String device() {
		return device;
	}

	public int split_mode() {
		return split_mode;
	}

	public int main_gpu() {
		return main_gpu;
	}

	public String tensor_split() {
		return tensor_split;
	}

	public boolean vocab_only() {
		return vocab_only;
	}

	public boolean use_mmap() {
		return use_mmap;
	}

	public boolean use_direct_io() {
		return use_direct_io;
	}

	public boolean use_mlock() {
		return use_mlock;
	}

	public boolean check_tensors() {
		return check_tensors;
	}

	public boolean use_extra_bufts() {
		return use_extra_bufts;
	}

	public boolean no_host() {
		return no_host;
	}

	public boolean no_alloc() {
		return no_alloc;
	}
}
