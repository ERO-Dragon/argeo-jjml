package org.argeo.jjml.llm;

import java.util.Objects;

import org.argeo.jjml.ggml.params.GgmlType;
import org.argeo.jjml.llm.params.ContextParam;
import org.argeo.jjml.llm.params.ContextParams;

/** Parameters for llama.cpp speculative decoding support. */
public final class SpeculativeParams {
	private static final int DEFAULT_MTP_DRAFT_MAX = 3;
	private static final int MAX_DRAFT_MAX = 0xffff;
	private static final float DEFAULT_DRAFT_SPLIT_PROBABILITY = 0.1f;
	private static final float DEFAULT_DRAFT_MIN_PROBABILITY = 0.0f;

	private final SpeculativeType type;
	private final int draftMax;
	private final int draftMin;
	private final float draftSplitProbability;
	private final float draftMinProbability;
	private final GgmlType cacheTypeK;
	private final GgmlType cacheTypeV;

	private SpeculativeParams(SpeculativeType type, int draftMax, int draftMin, float draftSplitProbability,
			float draftMinProbability, GgmlType cacheTypeK, GgmlType cacheTypeV) {
		this.type = Objects.requireNonNull(type);
		this.cacheTypeK = Objects.requireNonNull(cacheTypeK);
		this.cacheTypeV = Objects.requireNonNull(cacheTypeV);
		this.draftMax = draftMax;
		this.draftMin = draftMin;
		this.draftSplitProbability = draftSplitProbability;
		this.draftMinProbability = draftMinProbability;
		validate();
	}

	private void validate() {
		Objects.requireNonNull(type);
		Objects.requireNonNull(cacheTypeK);
		Objects.requireNonNull(cacheTypeV);
		if (type != SpeculativeType.NONE) {
			if (draftMax <= 0)
				throw new IllegalArgumentException("draftMax must be positive");
			if (draftMax > MAX_DRAFT_MAX)
				throw new IllegalArgumentException("draftMax must be <= " + MAX_DRAFT_MAX);
			if (draftMin < 0)
				throw new IllegalArgumentException("draftMin must not be negative");
			if (draftMin > draftMax)
				throw new IllegalArgumentException("draftMin cannot be greater than draftMax");
			if (draftSplitProbability < 0 || draftSplitProbability > 1)
				throw new IllegalArgumentException("draftSplitProbability must be between 0 and 1");
			if (draftMinProbability < 0 || draftMinProbability > 1)
				throw new IllegalArgumentException("draftMinProbability must be between 0 and 1");
		}
	}

	/**
	 * Disabled speculative decoding configuration. Most callers should simply not
	 * create a {@link LlamaCppSpeculativeProcessor} when this is selected.
	 */
	public static SpeculativeParams none() {
		return new SpeculativeParams(SpeculativeType.NONE, 0, 0, 0.0f, 0.0f, GgmlType.GGML_TYPE_F16,
				GgmlType.GGML_TYPE_F16);
	}

	/**
	 * Create a starter MTP configuration.
	 * <p>
	 * Prefer {@link #draftMtp(int)} in production so the upper layer can benchmark
	 * and choose {@code draftMax} explicitly for the current model, GPU and driver.
	 */
	public static SpeculativeParams draftMtp() {
		return draftMtp(DEFAULT_MTP_DRAFT_MAX);
	}

	/**
	 * Create the common MTP speculative configuration with a caller-chosen draft
	 * window.
	 *
	 * @param draftMax the number of draft tokens to speculate per request
	 */
	public static SpeculativeParams draftMtp(int draftMax) {
		return draftMtp(draftMax, DEFAULT_DRAFT_SPLIT_PROBABILITY, DEFAULT_DRAFT_MIN_PROBABILITY);
	}

	/**
	 * Advanced MTP configuration. The stable public tuning knob is still
	 * {@code draftMax}; the probability values are exposed for benchmark tooling and
	 * expert experiments.
	 */
	public static SpeculativeParams draftMtp(int draftMax, float draftSplitProbability, float draftMinProbability) {
		return new SpeculativeParams(SpeculativeType.DRAFT_MTP, draftMax, 0, draftSplitProbability,
				draftMinProbability, GgmlType.GGML_TYPE_F16, GgmlType.GGML_TYPE_F16);
	}

	/** Whether this configuration enables speculative decoding. */
	public boolean isEnabled() {
		return type != SpeculativeType.NONE;
	}

	/** Whether this configuration uses llama.cpp's MTP/NextN draft path. */
	public boolean isMtp() {
		return type == SpeculativeType.DRAFT_MTP;
	}

	/** Runtime draft-window size to try for this request. */
	public int draftMax() {
		return draftMax;
	}

	/**
	 * Minimum target context output slots needed by this configuration.
	 */
	public int requiredTargetOutputs() {
		return isMtp() ? requiredMtpTargetOutputs(draftMax) : 1;
	}

	/** Advanced MTP split probability. */
	public float draftSplitProbability() {
		return draftSplitProbability;
	}

	/** Advanced MTP minimum probability threshold. */
	public float draftMinProbability() {
		return draftMinProbability;
	}

	/**
	 * Minimum target context output slots needed by an MTP draft-window size.
	 */
	public static int requiredMtpTargetOutputs(int draftMax) {
		return validateDraftMax(draftMax) + 1;
	}

	/**
	 * Prepare target-context parameters for a planned maximum MTP draft window.
	 * <p>
	 * Use this when the upper layer wants to create one target context and then
	 * benchmark several per-request {@code draftMax} values against it.
	 */
	public static ContextParams adjustTargetContextParams(ContextParams params, int maxDraftMax) {
		Objects.requireNonNull(params);
		int draftMax = validateDraftMax(maxDraftMax);
		int requiredOutputs = requiredMtpTargetOutputs(draftMax);
		ContextParams adjusted = params;
		if (adjusted.n_rs_seq() < draftMax)
			adjusted = adjusted.with(ContextParam.n_rs_seq, draftMax);
		if (adjusted.n_outputs_max() != 0 && adjusted.n_outputs_max() < requiredOutputs)
			adjusted = adjusted.with(ContextParam.n_outputs_max, requiredOutputs);
		if (adjusted.n_batch() < requiredOutputs)
			adjusted = adjusted.with(ContextParam.n_batch, requiredOutputs);
		if (adjusted.n_ubatch() < requiredOutputs)
			adjusted = adjusted.with(ContextParam.n_ubatch, requiredOutputs);
		return adjusted;
	}

	/**
	 * Prepare the target-context parameters for batched MTP.
	 * <p>
	 * The upper layer may keep a plain single-output baseline, for example
	 * {@code n_outputs_max = 1}. When MTP is actually enabled, JJML lifts the
	 * target context to the minimum output/batch sizes required by the draft
	 * window. This is a runtime sizing rule; it is unrelated to the model's
	 * {@code nextn_predict_layers} metadata.
	 */
	public ContextParams adjustTargetContextParams(ContextParams params) {
		Objects.requireNonNull(params);
		if (type != SpeculativeType.DRAFT_MTP)
			return params;
		return adjustTargetContextParams(params, draftMax);
	}

	private static int validateDraftMax(int draftMax) {
		if (draftMax <= 0)
			throw new IllegalArgumentException("draftMax must be positive");
		if (draftMax > MAX_DRAFT_MAX)
			throw new IllegalArgumentException("draftMax must be <= " + MAX_DRAFT_MAX);
		return draftMax;
	}

	int typeCode() {
		return type.code();
	}

	int draftMin() {
		return draftMin;
	}

	int cacheTypeKCode() {
		return cacheTypeK.getAsInt();
	}

	int cacheTypeVCode() {
		return cacheTypeV.getAsInt();
	}
}
