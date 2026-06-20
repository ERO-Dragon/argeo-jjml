package org.argeo.jjml.ggml;

//import static java.lang.System.Logger.Level.INFO;
//import static java.lang.System.Logger.Level.WARNING;

import java.io.File;
import java.io.IOException;
//import java.lang.System.Logger;
import java.nio.charset.Charset;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/** A registered GGML backend. */
public class GgmlBackend {
//	private final static Logger logger = System.getLogger(GgmlBackend.class.getName());

	private final static String ENV_VK_LOADER_LAYERS_DISABLE = "VK_LOADER_LAYERS_DISABLE";
	private final static String VK_LOADER_LAYERS_DISABLE_IMPLICIT = "~implicit~";

	private final static String GGML_DL_PREFIX = "ggml-";
	// FIXME currently unused
	private final static List<GgmlBackend> loadedBackends = new ArrayList<>();

	/** Pointer to ggml_backend_reg_t. */
	private final long pointer;
	private final String name;
	private final Path path;

	public GgmlBackend(long pointer, String name, Path path) {
		this.pointer = pointer;
		this.name = name;
		this.path = path;
	}

	private static native long doLoadBackend(byte[] backendPath);

	private static native void doLoadAllBackends(byte[] basePath);

	private static native boolean doSetProcessEnvironment(byte[] name, byte[] value, boolean overwrite);

	static native boolean doIsVulkanSchedulerSupported();

	static native boolean doSetVulkanSchedulerParams(boolean enabled, boolean paused, boolean abortRequested,
			int maxNodesPerChunk, long maxMatmulBytesPerChunk, int timeWindowUs, int activeWindowUs,
			int maxChunksInFlight, int sleepGranularityUs);

	static native long[] doGetVulkanSchedulerParams();

	static native long[] doGetVulkanSchedulerStats();

	static native boolean doResetVulkanSchedulerStats();

	public static void loadAllBackends() {
		List<Path> basePaths = new ArrayList<>();

		// First try the "standard" deployment paths, so that they can be overridden
		// TODO make it cleaner and more configurable
		// TODO hardcode some paths on the native side and configure at build?
		// Debian
		String arch = System.getProperty("os.arch");
		String gnuArch;
		if ("arm64".equals(arch) || "aarch64".equals(arch))
			gnuArch = "aarch64";
		else
			gnuArch = "x86_64";
		Path path = Paths.get("/usr/lib/" + gnuArch + "-linux-gnu/ggml/backends0");
		// System.out.println(path);
		if (Files.exists(path))
			basePaths.add(path);
		else // Argeo
			basePaths.add(Paths.get("/usr/libexec/" + gnuArch + "-linux-gnu/ggml"));

		// Try the JNI path
		{
			String javaLibraryPath = System.getProperty("java.library.path");
			if (javaLibraryPath != null && !"".equals(javaLibraryPath.trim())) {
				// System.out.println(javaLibraryPath);
				String[] paths = javaLibraryPath.split(File.pathSeparator);
				for (String p : paths)
					basePaths.add(Paths.get(p));
			}
		}

		// As a last override option, environment path LD_LIBRARY_PATH
		{
			String ldLibraryPath = System.getenv("LD_LIBRARY_PATH");
			if (ldLibraryPath != null && !"".equals(ldLibraryPath.trim())) {
				// System.out.println(ldLibraryPath);
				String[] paths = ldLibraryPath.split(File.pathSeparator);
				for (String p : paths)
					basePaths.add(Paths.get(p));
			}
			
			// MacOS
			ldLibraryPath = System.getenv("DYLD_LIBRARY_PATH");
			if (ldLibraryPath != null && !"".equals(ldLibraryPath.trim())) {
				// System.out.println(ldLibraryPath);
				String[] paths = ldLibraryPath.split(File.pathSeparator);
				for (String p : paths)
					basePaths.add(Paths.get(p));
			}
		}

		// load
		Path lastRelevantPath = null;
		basePaths: for (Path basePath : basePaths) {
			if (Files.exists(basePath)) {
				// loadBackends(basePath);

				// We cannot use System.mapLibraryName("ggml-cpu*"),
				// because backends are *.so and not *.dylib on MacOS
				try (DirectoryStream<Path> ds = Files.newDirectoryStream(basePath, "*ggml-cpu*")) {
					Iterator<Path> it = ds.iterator();
					// scanning some directories causes crashes on Windows,
					// so we skip irrelevant ones
					if (!it.hasNext())
						continue basePaths;
				} catch (IOException e) {
					// silent
					continue basePaths;
				}
				lastRelevantPath = basePath;
			}
		}

		//
		// ACTUAL SEARCH
		//
//		logger.log(INFO, "Searching for ggml backends in: " + basePath);
		// loadBackends(basePath);
		if (lastRelevantPath != null) {
			if (Files.exists(lastRelevantPath.resolve(backendLibraryName(StandardBackend.vulkan))))
				applyVulkanLoaderPolicy();
			doLoadAllBackends(filePathToNative(lastRelevantPath));
		} else {
			System.err.println("Could not find ggml backends in any of " + basePaths);
		}
		//
	}

	private static void applyVulkanLoaderPolicy() {
		setProcessEnvironment(ENV_VK_LOADER_LAYERS_DISABLE, VK_LOADER_LAYERS_DISABLE_IMPLICIT);
	}

	private static void setProcessEnvironment(String name, String value) {
		if (!doSetProcessEnvironment(stringToNative(name), stringToNative(value), true))
			throw new IllegalStateException("Could not set process environment variable " + name + ".");
	}

	public String getName() {
		return name;
	}

	public Path getPath() {
		return path;
	}

	/** Load backends available in this directory. */
	public static void loadBackends(Path basePath) {
		// TODO recurse? could be useful for local builds

		backendNames: for (StandardBackend backendName : StandardBackend.values()) {
			// skip backends whose names are already loaded
			for (GgmlBackend backend : loadedBackends) {
				if (backendName.name().equals(backend.getName())) {
//					logger.log(WARNING, backendName.name() + " already loaded from " + backend.getPath());
					System.err.println(backendName.name() + " already loaded from " + backend.getPath());
					continue backendNames;
				}
			}

			String dllName = backendLibraryName(backendName);
			Path backendPath = basePath.resolve(dllName);
			if (Files.exists(backendPath)) {
				if (StandardBackend.vulkan.equals(backendName))
					applyVulkanLoaderPolicy();
				long pointer = doLoadBackend(filePathToNative(backendPath));
				if (pointer > 0) {
					// TODO log it
					GgmlBackend backend = new GgmlBackend(pointer, backendName.name(), backendPath);
					loadedBackends.add(backend);
//					logger.log(INFO, "Loaded backend " + backendName.name() + " from " + backend.getPath());
					System.out.println("Loaded backend " + backendName.name() + " from " + backend.getPath());
				}
			}
		}
	}

	private static String backendLibraryName(StandardBackend backendName) {
		if (File.separatorChar == '\\')
			return GGML_DL_PREFIX + backendName.name() + ".dll";
		// FIXME deal with MacOS
		return "lib" + GGML_DL_PREFIX + backendName.name() + ".so";
	}

	/** Path as bytes, based on the OS native encoding. */
	private static byte[] filePathToNative(Path path) {
		return path.toString().getBytes(Charset.forName(System.getProperty("sun.jnu.encoding", "UTF-8")));
	}

	/** String as bytes, based on the OS native encoding. */
	private static byte[] stringToNative(String value) {
		return value.getBytes(Charset.forName(System.getProperty("sun.jnu.encoding", "UTF-8")));
	}
}
