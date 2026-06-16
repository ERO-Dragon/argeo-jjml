package org.argeo.jjml.llm;

import java.util.Objects;

/** A ggml backend device that can be used for model offload. */
public class LlamaCppDevice {
	private final String backend;
	private final String name;
	private final String description;
	private final String deviceId;
	private final int type;
	private final long memoryFree;
	private final long memoryTotal;

	LlamaCppDevice(String backend, String name, String description, String deviceId, int type, long memoryFree,
			long memoryTotal) {
		this.backend = Objects.requireNonNull(backend);
		this.name = Objects.requireNonNull(name);
		this.description = description;
		this.deviceId = deviceId;
		this.type = type;
		this.memoryFree = memoryFree;
		this.memoryTotal = memoryTotal;
	}

	public String backend() {
		return backend;
	}

	public String name() {
		return name;
	}

	public String description() {
		return description;
	}

	public String deviceId() {
		return deviceId;
	}

	public int type() {
		return type;
	}

	public long memoryFree() {
		return memoryFree;
	}

	public long memoryTotal() {
		return memoryTotal;
	}

	@Override
	public String toString() {
		return backend + ":" + name + (deviceId != null ? " (" + deviceId + ")" : "");
	}
}
