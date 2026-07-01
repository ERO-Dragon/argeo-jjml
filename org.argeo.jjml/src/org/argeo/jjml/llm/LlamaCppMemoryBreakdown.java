package org.argeo.jjml.llm;

import java.util.Objects;

/** Memory allocated by llama.cpp for a backend buffer type. */
public class LlamaCppMemoryBreakdown {
	private final String bufferType;
	private final LlamaCppDevice device;
	private final boolean host;
	private final long modelBytes;
	private final long contextBytes;
	private final long computeBytes;

	LlamaCppMemoryBreakdown(String bufferType, LlamaCppDevice device, boolean host, long modelBytes, long contextBytes,
			long computeBytes) {
		this.bufferType = Objects.requireNonNull(bufferType);
		this.device = device;
		this.host = host;
		this.modelBytes = modelBytes;
		this.contextBytes = contextBytes;
		this.computeBytes = computeBytes;
	}

	public String bufferType() {
		return bufferType;
	}

	public LlamaCppDevice device() {
		return device;
	}

	public boolean host() {
		return host;
	}

	public long modelBytes() {
		return modelBytes;
	}

	public long contextBytes() {
		return contextBytes;
	}

	public long computeBytes() {
		return computeBytes;
	}

	public long totalBytes() {
		return modelBytes + contextBytes + computeBytes;
	}

	@Override
	public String toString() {
		return bufferType + "[model=" + modelBytes + ", context=" + contextBytes + ", compute=" + computeBytes
				+ "]";
	}
}
