package org.argeo.jjml.llm;

import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import org.argeo.jjml.llm.params.ModelParams;
import org.argeo.jjml.llm.params.PoolingType;

/**
 * Model-level facts collected through llama.cpp's no-allocation loading path.
 * <p>
 * This object is safe to keep after the native model used for inspection has
 * been released. It contains only metadata and capability facts, not runtime
 * policy decisions.
 */
public final class LlamaCppModelInfo {
	private final Path localPath;
	private final ModelParams requestedModelParams;
	private final ModelParams inspectionModelParams;
	private final int vocabularySize;
	private final int contextTrainingSize;
	private final int embeddingSize;
	private final int layerCount;
	private final Map<String, String> metadata;
	private final PoolingType defaultPoolingType;
	private final int mtpLayerCount;
	private final String description;
	private final long modelSize;
	private final int endOfGenerationToken;
	private final boolean supportsEnableThinking;
	private final LlamaCppChatTemplateCapabilities chatTemplateCapabilities;
	private final LlamaCppDevice[] devices;

	LlamaCppModelInfo(Path localPath, ModelParams requestedModelParams, ModelParams inspectionModelParams,
			int vocabularySize, int contextTrainingSize, int embeddingSize, int layerCount, Map<String, String> metadata,
			PoolingType defaultPoolingType, int mtpLayerCount, String description, long modelSize,
			int endOfGenerationToken, boolean supportsEnableThinking,
			LlamaCppChatTemplateCapabilities chatTemplateCapabilities, LlamaCppDevice[] devices) {
		this.localPath = Objects.requireNonNull(localPath);
		this.requestedModelParams = Objects.requireNonNull(requestedModelParams);
		this.inspectionModelParams = Objects.requireNonNull(inspectionModelParams);
		if (vocabularySize < 0)
			throw new IllegalArgumentException("vocabularySize cannot be negative");
		if (contextTrainingSize < 0)
			throw new IllegalArgumentException("contextTrainingSize cannot be negative");
		if (embeddingSize < 0)
			throw new IllegalArgumentException("embeddingSize cannot be negative");
		if (layerCount < 0)
			throw new IllegalArgumentException("layerCount cannot be negative");
		if (mtpLayerCount < 0)
			throw new IllegalArgumentException("mtpLayerCount cannot be negative");
		if (modelSize < 0)
			throw new IllegalArgumentException("modelSize cannot be negative");
		this.vocabularySize = vocabularySize;
		this.contextTrainingSize = contextTrainingSize;
		this.embeddingSize = embeddingSize;
		this.layerCount = layerCount;
		this.metadata = Collections.unmodifiableMap(new LinkedHashMap<>(Objects.requireNonNull(metadata)));
		this.defaultPoolingType = Objects.requireNonNull(defaultPoolingType);
		this.mtpLayerCount = mtpLayerCount;
		this.description = Objects.requireNonNull(description);
		this.modelSize = modelSize;
		this.endOfGenerationToken = endOfGenerationToken;
		this.supportsEnableThinking = supportsEnableThinking;
		this.chatTemplateCapabilities = Objects.requireNonNull(chatTemplateCapabilities);
		this.devices = devices == null ? new LlamaCppDevice[0] : devices.clone();
	}

	public Path localPath() {
		return localPath;
	}

	public ModelParams requestedModelParams() {
		return requestedModelParams;
	}

	public ModelParams inspectionModelParams() {
		return inspectionModelParams;
	}

	public int vocabularySize() {
		return vocabularySize;
	}

	public int contextTrainingSize() {
		return contextTrainingSize;
	}

	public int embeddingSize() {
		return embeddingSize;
	}

	public int layerCount() {
		return layerCount;
	}

	public Map<String, String> metadata() {
		return metadata;
	}

	public PoolingType defaultPoolingType() {
		return defaultPoolingType;
	}

	public int mtpLayerCount() {
		return mtpLayerCount;
	}

	public boolean supportsMtp() {
		return mtpLayerCount > 0;
	}

	public String description() {
		return description;
	}

	public long modelSize() {
		return modelSize;
	}

	public int endOfGenerationToken() {
		return endOfGenerationToken;
	}

	public boolean supportsEnableThinking() {
		return supportsEnableThinking;
	}

	public LlamaCppChatTemplateCapabilities chatTemplateCapabilities() {
		return chatTemplateCapabilities;
	}

	public LlamaCppDevice[] devices() {
		return devices.clone();
	}

	@Override
	public String toString() {
		return "LlamaCppModelInfo[path=" + localPath + ", description=" + description + ", layers=" + layerCount
				+ ", contextTrainingSize=" + contextTrainingSize + ", modelSize=" + modelSize + "]";
	}
}
