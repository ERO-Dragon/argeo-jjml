package org.argeo.jjml.ggml;

/** Cooperative execution controls for the GGML Vulkan backend. */
public final class GgmlVulkanScheduler {
	private static final int DEFAULT_TIME_WINDOW_US = 20_000;

	private static final Profile[] PROFILES = { //
			new Profile(true, 12, 16L * 1024 * 1024, DEFAULT_TIME_WINDOW_US, 1_000, 1, 1_000), //
			new Profile(true, 16, 24L * 1024 * 1024, DEFAULT_TIME_WINDOW_US, 2_000, 1, 1_000), //
			new Profile(true, 20, 32L * 1024 * 1024, DEFAULT_TIME_WINDOW_US, 4_000, 1, 1_000), //
			new Profile(true, 28, 48L * 1024 * 1024, DEFAULT_TIME_WINDOW_US, 6_000, 1, 1_000), //
			new Profile(true, 36, 64L * 1024 * 1024, DEFAULT_TIME_WINDOW_US, 8_000, 1, 1_000), //
			new Profile(true, 48, 80L * 1024 * 1024, DEFAULT_TIME_WINDOW_US, 10_000, 1, 1_000), //
			new Profile(true, 60, 96L * 1024 * 1024, DEFAULT_TIME_WINDOW_US, 12_000, 2, 1_000), //
			new Profile(true, 72, 100L * 1024 * 1024, DEFAULT_TIME_WINDOW_US, 14_000, 2, 1_000), //
			new Profile(true, 86, 100L * 1024 * 1024, DEFAULT_TIME_WINDOW_US, 16_000, 3, 1_000), //
			new Profile(true, 100, 100L * 1024 * 1024, DEFAULT_TIME_WINDOW_US, 18_000, 4, 1_000), //
			new Profile(false, 0, 0, 0, 0, 0, 0), //
	};

	private GgmlVulkanScheduler() {
	}

	/** Whether the loaded Vulkan backend exposes the scheduler extension. */
	public static boolean isSupported() {
		return GgmlBackend.doIsVulkanSchedulerSupported();
	}

	/**
	 * Set inference priority in eleven levels.
	 *
	 * @param priority 0.0 favors the game, 1.0 favors inference.
	 * @return {@code false} when the loaded Vulkan backend does not support this
	 *         extension.
	 */
	public static boolean setInferencePriority(float priority) {
		return setLevel(toLevel(priority));
	}

	/**
	 * Set an explicit scheduler level.
	 *
	 * @param level 0 favors the game, 10 favors inference.
	 */
	public static boolean setLevel(int level) {
		int clampedLevel = clamp(level, 0, PROFILES.length - 1);
		Profile profile = PROFILES[clampedLevel];
		Params current = getParams();
		return GgmlBackend.doSetVulkanSchedulerParams(profile.enabled, current.paused, current.abortRequested,
				profile.maxNodesPerChunk, profile.maxMatmulBytesPerChunk, profile.timeWindowUs, profile.activeWindowUs,
				profile.maxChunksInFlight, profile.sleepGranularityUs);
	}

	public static int toLevel(float priority) {
		if (!Float.isFinite(priority))
			throw new IllegalArgumentException("Priority must be finite.");
		float clampedPriority = Math.max(0.0f, Math.min(1.0f, priority));
		return Math.round(clampedPriority * 10.0f);
	}

	public static boolean setPaused(boolean paused) {
		Params current = getParams();
		return GgmlBackend.doSetVulkanSchedulerParams(current.enabled, paused, current.abortRequested,
				current.maxNodesPerChunk, current.maxMatmulBytesPerChunk, current.timeWindowUs, current.activeWindowUs,
				current.maxChunksInFlight, current.sleepGranularityUs);
	}

	public static boolean requestAbort() {
		Params current = getParams();
		return GgmlBackend.doSetVulkanSchedulerParams(current.enabled, current.paused, true, current.maxNodesPerChunk,
				current.maxMatmulBytesPerChunk, current.timeWindowUs, current.activeWindowUs,
				current.maxChunksInFlight, current.sleepGranularityUs);
	}

	public static boolean clearAbort() {
		Params current = getParams();
		return GgmlBackend.doSetVulkanSchedulerParams(current.enabled, current.paused, false, current.maxNodesPerChunk,
				current.maxMatmulBytesPerChunk, current.timeWindowUs, current.activeWindowUs,
				current.maxChunksInFlight, current.sleepGranularityUs);
	}

	public static boolean resetStats() {
		return GgmlBackend.doResetVulkanSchedulerStats();
	}

	public static Params getParams() {
		long[] values = GgmlBackend.doGetVulkanSchedulerParams();
		if (values == null)
			return Params.unsupported();
		return new Params(values[0] != 0, values[1] != 0, values[2] != 0, (int) values[3], values[4],
				(int) values[5], (int) values[6], (int) values[7], (int) values[8]);
	}

	public static Stats getStats() {
		long[] values = GgmlBackend.doGetVulkanSchedulerStats();
		if (values == null)
			return Stats.unsupported();
		return new Stats(true, values[0], values[1], values[2], values[3], values[4], values[5], values[6],
				values[7], values[8], values[9]);
	}

	private static int clamp(int value, int min, int max) {
		return Math.min(max, Math.max(min, value));
	}

	private static final class Profile {
		final boolean enabled;
		final int maxNodesPerChunk;
		final long maxMatmulBytesPerChunk;
		final int timeWindowUs;
		final int activeWindowUs;
		final int maxChunksInFlight;
		final int sleepGranularityUs;

		Profile(boolean enabled, int maxNodesPerChunk, long maxMatmulBytesPerChunk, int timeWindowUs,
				int activeWindowUs, int maxChunksInFlight, int sleepGranularityUs) {
			this.enabled = enabled;
			this.maxNodesPerChunk = maxNodesPerChunk;
			this.maxMatmulBytesPerChunk = maxMatmulBytesPerChunk;
			this.timeWindowUs = timeWindowUs;
			this.activeWindowUs = activeWindowUs;
			this.maxChunksInFlight = maxChunksInFlight;
			this.sleepGranularityUs = sleepGranularityUs;
		}
	}

	public static final class Params {
		private final boolean supported;
		private final boolean enabled;
		private final boolean paused;
		private final boolean abortRequested;
		private final int maxNodesPerChunk;
		private final long maxMatmulBytesPerChunk;
		private final int timeWindowUs;
		private final int activeWindowUs;
		private final int maxChunksInFlight;
		private final int sleepGranularityUs;

		private Params(boolean enabled, boolean paused, boolean abortRequested, int maxNodesPerChunk,
				long maxMatmulBytesPerChunk, int timeWindowUs, int activeWindowUs, int maxChunksInFlight,
				int sleepGranularityUs) {
			this(true, enabled, paused, abortRequested, maxNodesPerChunk, maxMatmulBytesPerChunk, timeWindowUs,
					activeWindowUs, maxChunksInFlight, sleepGranularityUs);
		}

		private Params(boolean supported, boolean enabled, boolean paused, boolean abortRequested, int maxNodesPerChunk,
				long maxMatmulBytesPerChunk, int timeWindowUs, int activeWindowUs, int maxChunksInFlight,
				int sleepGranularityUs) {
			this.supported = supported;
			this.enabled = enabled;
			this.paused = paused;
			this.abortRequested = abortRequested;
			this.maxNodesPerChunk = maxNodesPerChunk;
			this.maxMatmulBytesPerChunk = maxMatmulBytesPerChunk;
			this.timeWindowUs = timeWindowUs;
			this.activeWindowUs = activeWindowUs;
			this.maxChunksInFlight = maxChunksInFlight;
			this.sleepGranularityUs = sleepGranularityUs;
		}

		private static Params unsupported() {
			return new Params(false, false, false, false, 0, 0, 0, 0, 0, 0);
		}

		public boolean supported() {
			return supported;
		}

		public boolean enabled() {
			return enabled;
		}

		public boolean paused() {
			return paused;
		}

		public boolean abortRequested() {
			return abortRequested;
		}

		public int maxNodesPerChunk() {
			return maxNodesPerChunk;
		}

		public long maxMatmulBytesPerChunk() {
			return maxMatmulBytesPerChunk;
		}

		public int timeWindowUs() {
			return timeWindowUs;
		}

		public int activeWindowUs() {
			return activeWindowUs;
		}

		public int maxChunksInFlight() {
			return maxChunksInFlight;
		}

		public int sleepGranularityUs() {
			return sleepGranularityUs;
		}
	}

	public static final class Stats {
		private final boolean supported;
		private final long chunkCount;
		private final long lastChunkWallUs;
		private final long lastChunkNodes;
		private final long lastChunkMatmulBytes;
		private final long slotWaitCount;
		private final long totalSlotWaitUs;
		private final long lastSlotWaitUs;
		private final long fenceWaitCount;
		private final long totalFenceWaitUs;
		private final long lastFenceWaitUs;

		private Stats(boolean supported, long chunkCount, long lastChunkWallUs, long lastChunkNodes,
				long lastChunkMatmulBytes, long slotWaitCount, long totalSlotWaitUs, long lastSlotWaitUs,
				long fenceWaitCount, long totalFenceWaitUs, long lastFenceWaitUs) {
			this.supported = supported;
			this.chunkCount = chunkCount;
			this.lastChunkWallUs = lastChunkWallUs;
			this.lastChunkNodes = lastChunkNodes;
			this.lastChunkMatmulBytes = lastChunkMatmulBytes;
			this.slotWaitCount = slotWaitCount;
			this.totalSlotWaitUs = totalSlotWaitUs;
			this.lastSlotWaitUs = lastSlotWaitUs;
			this.fenceWaitCount = fenceWaitCount;
			this.totalFenceWaitUs = totalFenceWaitUs;
			this.lastFenceWaitUs = lastFenceWaitUs;
		}

		private static Stats unsupported() {
			return new Stats(false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		}

		public boolean supported() {
			return supported;
		}

		public long chunkCount() {
			return chunkCount;
		}

		public long lastChunkWallUs() {
			return lastChunkWallUs;
		}

		public long lastChunkNodes() {
			return lastChunkNodes;
		}

		public long lastChunkMatmulBytes() {
			return lastChunkMatmulBytes;
		}

		public long slotWaitCount() {
			return slotWaitCount;
		}

		public long totalSlotWaitUs() {
			return totalSlotWaitUs;
		}

		public long lastSlotWaitUs() {
			return lastSlotWaitUs;
		}

		public long fenceWaitCount() {
			return fenceWaitCount;
		}

		public long totalFenceWaitUs() {
			return totalFenceWaitUs;
		}

		public long lastFenceWaitUs() {
			return lastFenceWaitUs;
		}
	}
}
