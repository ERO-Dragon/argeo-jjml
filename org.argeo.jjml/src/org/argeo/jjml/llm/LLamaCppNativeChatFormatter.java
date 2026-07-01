package org.argeo.jjml.llm;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Format chat messages using llama.cpp capabilities. Supports both legacy
 * template mode and Jinja2 template mode (with enable_thinking and
 * chat_template_kwargs).
 */
public class LLamaCppNativeChatFormatter {

	/*
	 * NATIVE METHODS
	 */

	/** Legacy mode - no Jinja2 support. */
	private static native byte[] doFormatChatMessages(byte[][] utf8Roles, byte[][] utf8Contents,
			boolean addAssistantTokens, byte[] ut8ChatTemplate);

	/** Jinja2 mode - supports enable_thinking and chat_template_kwargs. */
	private static native byte[] doFormatChatMessagesJinja(long modelPointer, byte[][] utf8Roles,
			byte[][] utf8Contents, boolean addGenerationPrompt, byte[] utf8ChatTemplate, boolean enableThinking,
			byte[][] kwargsKeys, byte[][] kwargsValues);

	/** Jinja2 mode returning llama.cpp's full chat formatting metadata. */
	private static native LlamaCppChatFormat doFormatChatMessagesJinjaFull(long modelPointer, byte[][] utf8Roles,
			byte[][] utf8Contents, boolean addGenerationPrompt, byte[] utf8ChatTemplate, boolean enableThinking,
			byte[][] kwargsKeys, byte[][] kwargsValues, byte[][] toolNames, byte[][] toolDescriptions,
			byte[][] toolParametersJson, byte[] toolChoice, boolean parallelToolCalls, byte[] jsonSchema);

	/** Check if the model's chat template supports enable_thinking. */
	private static native boolean doSupportsEnableThinking(long modelPointer, byte[] utf8ChatTemplate);

	/** Return template capability flags inferred by llama.cpp. */
	private static native LlamaCppChatTemplateCapabilities doGetTemplateCapabilities(long modelPointer,
			byte[] utf8ChatTemplate);

	/*
	 * USABLE METHODS
	 */

	/**
	 * Format a list of chat messages using legacy mode (no Jinja2 support).
	 * 
	 * @param messages           the list of qualified chat messages
	 * @param addAssistantTokens whether a given message should be considered 'user'
	 *                           (returns <code>true</code>) or 'assistant'
	 * @param chatTemplate       the llama.cpp id for the chat template (e.g.
	 *                           'granite'), not a full template
	 * @return the formatted messages as single string
	 */
	static String formatChatMessages(List<LlamaCppChatMessage> messages,
			Predicate<LlamaCppChatMessage> addAssistantTokens, String chatTemplate) {
		// filter out null values
		List<LlamaCppChatMessage> msgs = messages.stream().filter(Objects::nonNull).collect(Collectors.toList());
		byte[][] roles = new byte[msgs.size()][];
		byte[][] contents = new byte[msgs.size()][];

		boolean currIsUserRole = false;
		messages: for (int i = 0; i < msgs.size(); i++) {
			LlamaCppChatMessage message = msgs.get(i);
			if (message == null)
				continue messages; // ignore
			roles[i] = message.getRole().getBytes(UTF_8);
			currIsUserRole = addAssistantTokens.test(message);
			contents[i] = message.getContent().getBytes(UTF_8);
		}

		byte[] templateBytes = chatTemplate != null ? chatTemplate.getBytes(UTF_8) : null;
		byte[] res = doFormatChatMessages(roles, contents, currIsUserRole, templateBytes);
		return new String(res, UTF_8);
	}

	/**
	 * Format chat messages using Jinja2 template engine, with support for
	 * enable_thinking and chat_template_kwargs.
	 * 
	 * @param modelPointer       the native model pointer
	 * @param messages           the list of qualified chat messages
	 * @param addGenerationPrompt whether to add the generation prompt
	 * @param chatTemplate       the chat template string (can be null to use the
	 *                           model's built-in template)
	 * @param enableThinking     whether to enable thinking/reasoning mode
	 * @param chatTemplateKwargs additional key-value pairs passed to the Jinja2
	 *                           template
	 * @return the formatted messages as single string
	 */
	static String formatChatMessagesJinja(long modelPointer, List<LlamaCppChatMessage> messages,
			boolean addGenerationPrompt, String chatTemplate, boolean enableThinking,
			Map<String, String> chatTemplateKwargs) {
		return formatChatMessagesJinjaFull(modelPointer, messages, addGenerationPrompt, chatTemplate, enableThinking,
				chatTemplateKwargs, Collections.emptyList(), LlamaCppChatToolChoice.AUTO, false, null).prompt();
	}

	static LlamaCppChatFormat formatChatMessagesJinjaFull(long modelPointer, List<LlamaCppChatMessage> messages,
			boolean addGenerationPrompt, String chatTemplate, boolean enableThinking,
			Map<String, String> chatTemplateKwargs, List<LlamaCppChatTool> tools,
			LlamaCppChatToolChoice toolChoice, boolean parallelToolCalls, String jsonSchema) {
		List<LlamaCppChatMessage> msgs = messages.stream().filter(Objects::nonNull).collect(Collectors.toList());
		byte[][] roles = new byte[msgs.size()][];
		byte[][] contents = new byte[msgs.size()][];

		for (int i = 0; i < msgs.size(); i++) {
			LlamaCppChatMessage message = msgs.get(i);
			roles[i] = message.getRole().getBytes(UTF_8);
			contents[i] = message.getContent().getBytes(UTF_8);
		}

		Map<String, String> kwargs = chatTemplateKwargs != null ? chatTemplateKwargs : Collections.emptyMap();
		byte[][] kwargsKeys = new byte[kwargs.size()][];
		byte[][] kwargsValues = new byte[kwargs.size()][];
		int idx = 0;
		for (Map.Entry<String, String> entry : kwargs.entrySet()) {
			kwargsKeys[idx] = entry.getKey().getBytes(UTF_8);
			kwargsValues[idx] = entry.getValue().getBytes(UTF_8);
			idx++;
		}

		List<LlamaCppChatTool> toolList = tools != null ? tools : Collections.emptyList();
		byte[][] toolNames = new byte[toolList.size()][];
		byte[][] toolDescriptions = new byte[toolList.size()][];
		byte[][] toolParametersJson = new byte[toolList.size()][];
		for (int i = 0; i < toolList.size(); i++) {
			LlamaCppChatTool tool = toolList.get(i);
			toolNames[i] = tool.name().getBytes(UTF_8);
			toolDescriptions[i] = tool.description().getBytes(UTF_8);
			toolParametersJson[i] = tool.parametersJson().getBytes(UTF_8);
		}

		byte[] templateBytes = chatTemplate != null ? chatTemplate.getBytes(UTF_8) : null;
		byte[] toolChoiceBytes = (toolChoice != null ? toolChoice : LlamaCppChatToolChoice.AUTO).llamaName()
				.getBytes(UTF_8);
		byte[] jsonSchemaBytes = jsonSchema != null ? jsonSchema.getBytes(UTF_8) : null;
		return doFormatChatMessagesJinjaFull(modelPointer, roles, contents, addGenerationPrompt, templateBytes,
				enableThinking, kwargsKeys, kwargsValues, toolNames, toolDescriptions, toolParametersJson,
				toolChoiceBytes, parallelToolCalls, jsonSchemaBytes);
	}

	/**
	 * Check if the model's chat template supports enable_thinking.
	 * 
	 * @param modelPointer the native model pointer
	 * @param chatTemplate the chat template string (can be null to use the model's
	 *                     built-in template)
	 * @return true if the template supports enable_thinking
	 */
	static boolean supportsEnableThinking(long modelPointer, String chatTemplate) {
		byte[] templateBytes = chatTemplate != null ? chatTemplate.getBytes(UTF_8) : null;
		return doSupportsEnableThinking(modelPointer, templateBytes);
	}

	static LlamaCppChatTemplateCapabilities getTemplateCapabilities(long modelPointer, String chatTemplate) {
		byte[] templateBytes = chatTemplate != null ? chatTemplate.getBytes(UTF_8) : null;
		return doGetTemplateCapabilities(modelPointer, templateBytes);
	}
}
