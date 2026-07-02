package org.argeo.jjml.llm;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;

import org.argeo.jjml.llm.params.ContextParams;
import org.argeo.jjml.llm.params.ModelParams;
import org.argeo.jjml.llm.params.PoolingType;

/**
 * Context-level preflight facts collected before creating the real runtime
 * context.
 * <p>
 * The plan contains model-level metadata plus effective context parameters and
 * the simulated native memory layout for the requested context. It is a fact
 * snapshot for upper layers; it does not choose budgets, safety margins, or
 * context policies.
 */
public final class LlamaCppContextPlan {
	private final LlamaCppModelInfo modelInfo;
	private final ContextParams requestedContextParams;
	private final ContextParams contextParams;
	private final int contextSize;
	private final long kvCacheBytesPerToken;
	private final LlamaCppMemoryBreakdown[] memoryBreakdown;

	LlamaCppContextPlan(LlamaCppModelInfo modelInfo, ContextParams requestedContextParams, ContextParams contextParams,
			int contextSize, long kvCacheBytesPerToken, LlamaCppMemoryBreakdown[] memoryBreakdown) {
		this.modelInfo = Objects.requireNonNull(modelInfo);
		this.requestedContextParams = Objects.requireNonNull(requestedContextParams);
		this.contextParams = Objects.requireNonNull(contextParams);
		if (contextSize < 0)
			throw new IllegalArgumentException("contextSize cannot be negative");
		if (kvCacheBytesPerToken < 0)
			throw new IllegalArgumentException("kvCacheBytesPerToken cannot be negative");
		this.contextSize = contextSize;
		this.kvCacheBytesPerToken = kvCacheBytesPerToken;
		this.memoryBreakdown = memoryBreakdown == null ? new LlamaCppMemoryBreakdown[0] : memoryBreakdown.clone();
	}

	public LlamaCppModelInfo modelInfo() {
		return modelInfo;
	}

	public Path localPath() {
		return modelInfo.localPath();
	}

	public ModelParams requestedModelParams() {
		return modelInfo.requestedModelParams();
	}

	public ModelParams planningModelParams() {
		return modelInfo.inspectionModelParams();
	}

	public ContextParams requestedContextParams() {
		return requestedContextParams;
	}

	public ContextParams contextParams() {
		return contextParams;
	}

	public int contextSize() {
		return contextSize;
	}

	public int contextTrainingSize() {
		return modelInfo.contextTrainingSize();
	}

	public int vocabularySize() {
		return modelInfo.vocabularySize();
	}

	public int embeddingSize() {
		return modelInfo.embeddingSize();
	}

	public int layerCount() {
		return modelInfo.layerCount();
	}

	public Map<String, String> metadata() {
		return modelInfo.metadata();
	}

	public PoolingType defaultPoolingType() {
		return modelInfo.defaultPoolingType();
	}

	public int mtpLayerCount() {
		return modelInfo.mtpLayerCount();
	}

	public boolean supportsMtp() {
		return modelInfo.supportsMtp();
	}

	public String description() {
		return modelInfo.description();
	}

	public long modelSize() {
		return modelInfo.modelSize();
	}

	public int endOfGenerationToken() {
		return modelInfo.endOfGenerationToken();
	}

	public boolean supportsEnableThinking() {
		return modelInfo.supportsEnableThinking();
	}

	public LlamaCppChatTemplateCapabilities chatTemplateCapabilities() {
		return modelInfo.chatTemplateCapabilities();
	}

	public long kvCacheBytesPerToken() {
		return kvCacheBytesPerToken;
	}

	public long kvCacheBytes() {
		return saturatedMultiply(kvCacheBytesPerToken, contextSize);
	}

	public LlamaCppMemoryBreakdown[] memoryBreakdown() {
		return memoryBreakdown.clone();
	}

	public LlamaCppDevice[] devices() {
		return modelInfo.devices();
	}

	public long modelBytes() {
		return sum(MemoryPart.MODEL);
	}

	public long contextBytes() {
		return sum(MemoryPart.CONTEXT);
	}

	public long computeBytes() {
		return sum(MemoryPart.COMPUTE);
	}

	public long hostBytes() {
		return sumHost(true);
	}

	public long deviceBytes() {
		return sumHost(false);
	}

	public long totalBytes() {
		long total = 0L;
		for (LlamaCppMemoryBreakdown breakdown : memoryBreakdown) {
			if (breakdown != null)
				total = saturatedAdd(total, breakdown.totalBytes());
		}
		return total;
	}

	@Override
	public String toString() {
		return "LlamaCppContextPlan[path=" + localPath() + ", contextSize=" + contextSize + ", modelBytes="
				+ modelBytes() + ", contextBytes=" + contextBytes() + ", computeBytes=" + computeBytes()
				+ ", totalBytes=" + totalBytes() + ", devices=" + Arrays.toString(devices()) + "]";
	}

	private long sum(MemoryPart part) {
		long total = 0L;
		for (LlamaCppMemoryBreakdown breakdown : memoryBreakdown) {
			if (breakdown == null)
				continue;
			switch (part) {
			case MODEL:
				total = saturatedAdd(total, breakdown.modelBytes());
				break;
			case CONTEXT:
				total = saturatedAdd(total, breakdown.contextBytes());
				break;
			case COMPUTE:
				total = saturatedAdd(total, breakdown.computeBytes());
				break;
			default:
				break;
			}
		}
		return total;
	}

	private long sumHost(boolean host) {
		long total = 0L;
		for (LlamaCppMemoryBreakdown breakdown : memoryBreakdown) {
			if (breakdown != null && breakdown.host() == host)
				total = saturatedAdd(total, breakdown.totalBytes());
		}
		return total;
	}

	private static long saturatedAdd(long a, long b) {
		if (b <= 0)
			return a;
		if (Long.MAX_VALUE - a < b)
			return Long.MAX_VALUE;
		return a + b;
	}

	private static long saturatedMultiply(long a, long b) {
		if (a <= 0 || b <= 0)
			return 0L;
		if (Long.MAX_VALUE / a < b)
			return Long.MAX_VALUE;
		return a * b;
	}

	private enum MemoryPart {
		MODEL, CONTEXT, COMPUTE
	}
}
