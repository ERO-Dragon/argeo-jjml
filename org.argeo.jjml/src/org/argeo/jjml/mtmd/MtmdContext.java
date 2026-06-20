package org.argeo.jjml.mtmd;

import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.function.LongSupplier;

import org.argeo.jjml.llm.LlamaCppModel;

public class MtmdContext implements LongSupplier, AutoCloseable {
	private final long pointer;

	public MtmdContext(LlamaCppModel model, Path mmprojPath, int threads) {
		this(model, mmprojPath, true, threads);
	}

	public MtmdContext(LlamaCppModel model, Path mmprojPath, boolean useGpu, int threads) {
		MtmdNative.ensureLibrariesLoaded();
		Objects.requireNonNull(model);
		Objects.requireNonNull(mmprojPath);
		if (!Files.isRegularFile(mmprojPath))
			throw new IllegalArgumentException("mmproj path does not exist or is not a file: " + mmprojPath);
		this.pointer = doInit(model, filePathToNative(mmprojPath), useGpu, threads);
	}

	private static native long doInit(LlamaCppModel model, byte[] mmprojPath, boolean useGpu, int threads);

	private native void doDestroy();

	@Override
	public long getAsLong() {
		return pointer;
	}

	@Override
	public void close() throws Exception {
		doDestroy();
	}

	/** Path as bytes, based on the OS native encoding. */
	private static byte[] filePathToNative(Path path) {
		return path.toString().getBytes(Charset.forName(System.getProperty("sun.jnu.encoding", "UTF-8")));
	}

}
