package org.argeo.jjml.llm;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/** Capability flags inferred by llama.cpp from a model chat template. */
public final class LlamaCppChatTemplateCapabilities {
	private final boolean supportsTools;
	private final boolean supportsToolCalls;
	private final boolean supportsSystemRole;
	private final boolean supportsParallelToolCalls;
	private final boolean supportsPreserveReasoning;
	private final boolean supportsStringContent;
	private final boolean supportsTypedContent;
	private final boolean supportsObjectArguments;

	LlamaCppChatTemplateCapabilities(boolean supportsTools, boolean supportsToolCalls, boolean supportsSystemRole,
			boolean supportsParallelToolCalls, boolean supportsPreserveReasoning, boolean supportsStringContent,
			boolean supportsTypedContent, boolean supportsObjectArguments) {
		this.supportsTools = supportsTools;
		this.supportsToolCalls = supportsToolCalls;
		this.supportsSystemRole = supportsSystemRole;
		this.supportsParallelToolCalls = supportsParallelToolCalls;
		this.supportsPreserveReasoning = supportsPreserveReasoning;
		this.supportsStringContent = supportsStringContent;
		this.supportsTypedContent = supportsTypedContent;
		this.supportsObjectArguments = supportsObjectArguments;
	}

	public boolean supportsTools() {
		return supportsTools;
	}

	public boolean supportsToolCalls() {
		return supportsToolCalls;
	}

	public boolean supportsSystemRole() {
		return supportsSystemRole;
	}

	public boolean supportsParallelToolCalls() {
		return supportsParallelToolCalls;
	}

	public boolean supportsPreserveReasoning() {
		return supportsPreserveReasoning;
	}

	public boolean supportsStringContent() {
		return supportsStringContent;
	}

	public boolean supportsTypedContent() {
		return supportsTypedContent;
	}

	public boolean supportsObjectArguments() {
		return supportsObjectArguments;
	}

	public Map<String, Boolean> asMap() {
		Map<String, Boolean> map = new LinkedHashMap<>();
		map.put("supports_tools", supportsTools);
		map.put("supports_tool_calls", supportsToolCalls);
		map.put("supports_system_role", supportsSystemRole);
		map.put("supports_parallel_tool_calls", supportsParallelToolCalls);
		map.put("supports_preserve_reasoning", supportsPreserveReasoning);
		map.put("supports_string_content", supportsStringContent);
		map.put("supports_typed_content", supportsTypedContent);
		map.put("supports_object_arguments", supportsObjectArguments);
		return Collections.unmodifiableMap(map);
	}
}
