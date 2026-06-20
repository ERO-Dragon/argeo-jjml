package org.argeo.jjml.mtmd;

import org.argeo.jjml.llm.LlamaCppNative;

/** Availability of the native bindings to libmtmd. */
public class MtmdNative {
	public final static String SYSTEM_PROPERTY_MTMD_ENABLED = "jjml.mtmd.enabled";
	public final static String SYSTEM_PROPERTY_VISION_ENABLED = "jjml.vision.enabled";

	private final static String JJML_MTMD_LIBRARY_NAME = "Java_org_argeo_jjml_mtmd";

	private static boolean librariesLoaded = false;

	/*
	 * STATIC UTILITIES
	 */
	public static boolean isAvailable() {
		if (!isEnabled())
			return false;
		try {
			ensureLibrariesLoaded();
			return true;
		} catch (RuntimeException | UnsatisfiedLinkError e) {
			return false;
		}
	}

	public synchronized static void ensureLibrariesLoaded() {
		if (librariesLoaded)
			return;
		ensureEnabled();
		LlamaCppNative.ensureLibrariesLoaded();
		loadLibraries();
	}

	public static boolean isEnabled() {
		return Boolean.getBoolean(SYSTEM_PROPERTY_MTMD_ENABLED)
				|| Boolean.getBoolean(SYSTEM_PROPERTY_VISION_ENABLED);
	}

	public static void enable() {
		checkLibrariesNotLoaded();
		System.setProperty(SYSTEM_PROPERTY_MTMD_ENABLED, Boolean.TRUE.toString());
	}

	/**
	 * Explicitly disable multimodal/vision support before the native mtmd binding is
	 * loaded.
	 */
	public static void disable() {
		checkLibrariesNotLoaded();
		System.setProperty(SYSTEM_PROPERTY_MTMD_ENABLED, Boolean.FALSE.toString());
		System.setProperty(SYSTEM_PROPERTY_VISION_ENABLED, Boolean.FALSE.toString());
	}

	static void ensureEnabled() {
		if (!isEnabled())
			throw new IllegalStateException("Multimodal/vision support is disabled. Set "
					+ SYSTEM_PROPERTY_MTMD_ENABLED + "=true before loading mtmd.");
	}

	synchronized static void loadLibraries() {
		checkLibrariesNotLoaded();
		System.loadLibrary(JJML_MTMD_LIBRARY_NAME);
		librariesLoaded = true;
	}

	/** Fails if libraries already loaded. */
	private static void checkLibrariesNotLoaded() {
		if (librariesLoaded)
			throw new IllegalStateException("Shared libraries are already loaded.");
	}

	/** singleton */
	private MtmdNative() {
	}
}
