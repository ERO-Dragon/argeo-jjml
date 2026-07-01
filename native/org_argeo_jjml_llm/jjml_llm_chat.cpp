#include <string>
#include <vector>
#include <cassert>
#include <map>

#include <llama.h>
#include <chat.h>
#include <common.h>

#include <argeo/jni/argeo_jni.h>

#include "org_argeo_jjml_llm_LLamaCppNativeChatFormatter.h" // IWYU pragma: keep
#include "org_argeo_jjml_llm_.h"

namespace {

static jbyteArray jjml_chat_to_byte_array(JNIEnv *env, const std::string &value) {
	jbyteArray res = env->NewByteArray(value.length());
	env->SetByteArrayRegion(res, 0, value.length(),
			reinterpret_cast<const jbyte*>(value.data()));
	return res;
}

static jstring jjml_chat_to_string(JNIEnv *env, const std::string &value) {
	jbyteArray bytes = jjml_chat_to_byte_array(env, value);
	jclass stringClass = argeo::jni::find_jclass(env, "java/lang/String");
	jclass standardCharsetsClass = argeo::jni::find_jclass(env,
			"java/nio/charset/StandardCharsets");
	jfieldID utf8Field = env->GetStaticFieldID(standardCharsetsClass, "UTF_8",
			"Ljava/nio/charset/Charset;");
	jobject utf8 = env->GetStaticObjectField(standardCharsetsClass, utf8Field);
	jmethodID ctor = env->GetMethodID(stringClass, "<init>",
			"([BLjava/nio/charset/Charset;)V");
	return static_cast<jstring>(env->NewObject(stringClass, ctor, bytes, utf8));
}

static std::vector<common_chat_msg> jjml_chat_messages_from_java(JNIEnv *env,
		jobjectArray roles, jobjectArray contents) {
	const jsize messages_size = env->GetArrayLength(roles);
	assert(env->GetArrayLength(contents) == messages_size);

	std::vector<common_chat_msg> chat_messages;
	chat_messages.reserve(messages_size);
	for (int i = 0; i < messages_size; i++) {
		common_chat_msg msg;
		msg.role = argeo::jni::to_string(env, roles, i);
		msg.content = argeo::jni::to_string(env, contents, i);
		chat_messages.push_back(msg);
	}
	return chat_messages;
}

static std::map<std::string, std::string> jjml_chat_kwargs_from_java(
		JNIEnv *env, jobjectArray kwargsKeys, jobjectArray kwargsValues) {
	std::map<std::string, std::string> template_kwargs;
	if (kwargsKeys == nullptr && kwargsValues == nullptr)
		return template_kwargs;
	if (kwargsKeys == nullptr || kwargsValues == nullptr)
		throw std::invalid_argument(
				"Chat template kwargs keys and values must both be provided");

	const jsize kwargs_size = env->GetArrayLength(kwargsKeys);
	if (env->GetArrayLength(kwargsValues) != kwargs_size)
		throw std::invalid_argument(
				"Chat template kwargs keys and values have different sizes");
	for (int i = 0; i < kwargs_size; i++) {
		std::string key = argeo::jni::to_string(env, kwargsKeys, i);
		std::string val = argeo::jni::to_string(env, kwargsValues, i);
		template_kwargs[key] = val;
	}
	return template_kwargs;
}

static std::vector<common_chat_tool> jjml_chat_tools_from_java(JNIEnv *env,
		jobjectArray names, jobjectArray descriptions,
		jobjectArray parametersJson) {
	std::vector<common_chat_tool> tools;
	if (names == nullptr && descriptions == nullptr && parametersJson == nullptr)
		return tools;
	if (names == nullptr || descriptions == nullptr || parametersJson == nullptr)
		throw std::invalid_argument(
				"Tool names, descriptions and parameters must all be provided");

	const jsize tools_size = env->GetArrayLength(names);
	if (env->GetArrayLength(descriptions) != tools_size
			|| env->GetArrayLength(parametersJson) != tools_size)
		throw std::invalid_argument("Tool arrays have different sizes");
	tools.reserve(tools_size);
	for (int i = 0; i < tools_size; i++) {
		common_chat_tool tool;
		tool.name = argeo::jni::to_string(env, names, i);
		tool.description = argeo::jni::to_string(env, descriptions, i);
		tool.parameters = argeo::jni::to_string(env, parametersJson, i);
		tools.push_back(tool);
	}
	return tools;
}

static common_chat_params jjml_chat_apply_jinja(JNIEnv *env, jlong modelPointer,
		jobjectArray roles, jobjectArray contents, jboolean addGenerationPrompt,
		jbyteArray chatTemplateStr, jboolean enableThinking,
		jobjectArray kwargsKeys, jobjectArray kwargsValues,
		jobjectArray toolNames, jobjectArray toolDescriptions,
		jobjectArray toolParametersJson, jbyteArray toolChoice,
		jboolean parallelToolCalls, jbyteArray jsonSchema) {
	auto *model = argeo::jni::as_pointer<llama_model*>(modelPointer);

	std::string u8_chat_template;
	if (chatTemplateStr != nullptr)
		u8_chat_template = argeo::jni::to_string(env, chatTemplateStr);

	auto tmpls = common_chat_templates_init(model, u8_chat_template);

	common_chat_templates_inputs inputs;
	inputs.messages = jjml_chat_messages_from_java(env, roles, contents);
	inputs.add_generation_prompt = addGenerationPrompt;
	inputs.use_jinja = true;
	inputs.enable_thinking = enableThinking;
	inputs.chat_template_kwargs = jjml_chat_kwargs_from_java(env, kwargsKeys,
			kwargsValues);
	inputs.tools = jjml_chat_tools_from_java(env, toolNames, toolDescriptions,
			toolParametersJson);
	if (toolChoice != nullptr)
		inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(
				argeo::jni::to_string(env, toolChoice));
	inputs.parallel_tool_calls = parallelToolCalls;
	if (jsonSchema != nullptr)
		inputs.json_schema = argeo::jni::to_string(env, jsonSchema);

	return common_chat_templates_apply(tmpls.get(), inputs);
}

static bool jjml_chat_cap(const std::map<std::string, bool> &caps,
		const std::string &key) {
	auto it = caps.find(key);
	return it != caps.end() ? it->second : false;
}

static jobjectArray jjml_chat_grammar_triggers_to_java(JNIEnv *env,
		const std::vector<common_grammar_trigger> &triggers) {
	jclass triggerClass = argeo::jni::find_jclass(env,
			JCLASS_GRAMMAR_TRIGGER);
	jobjectArray result = env->NewObjectArray(
			static_cast<jsize>(triggers.size()), triggerClass, nullptr);
	for (jsize i = 0; i < static_cast<jsize>(triggers.size()); ++i) {
		const common_grammar_trigger &trigger = triggers[i];
		jobject obj = env->NewObject(triggerClass, LlamaCppGrammarTrigger__init,
				static_cast<jint>(trigger.type),
				jjml_chat_to_string(env, trigger.value),
				static_cast<jint>(trigger.token));
		env->SetObjectArrayElement(result, i, obj);
		env->DeleteLocalRef(obj);
	}
	return result;
}

} // namespace

/*
 * CHAT (Legacy - uses llama_chat_apply_template, no Jinja2 support)
 */
JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_llm_LLamaCppNativeChatFormatter_doFormatChatMessages(
		JNIEnv *env, jclass, jobjectArray roles, jobjectArray contents,
		jboolean addAssistantTokens, jbyteArray chatTemplateStr) {
	const jsize messages_size = env->GetArrayLength(roles);
	assert(env->GetArrayLength(contents) == messages_size);

	std::vector<llama_chat_message> chat_messages;

	try {
		size_t alloc_size = 0;
		std::vector<std::string> u8_roles;
		std::vector<std::string> u8_contents;
		u8_roles.reserve(messages_size);
		u8_contents.reserve(messages_size);
		chat_messages.reserve(messages_size);

		// since the content can be quite big, we go through the heap
		for (int i = 0; i < messages_size; i++) {
			u8_roles.push_back(argeo::jni::to_string(env, roles, i));
			u8_contents.push_back(argeo::jni::to_string(env, contents, i));

			llama_chat_message message { u8_roles.back().c_str(),
					u8_contents.back().c_str() };
			chat_messages.push_back(message);

			// using the same factor as in common.cpp
			alloc_size += static_cast<size_t>(
					(u8_roles.back().length() + u8_contents.back().length())
							* 1.25);
		}

		std::string u8_chat_template;
		if (chatTemplateStr != nullptr)
			u8_chat_template = argeo::jni::to_string(env, chatTemplateStr);

		std::vector<char> buf(alloc_size > 0 ? alloc_size : 1);
		int32_t resLength = llama_chat_apply_template(
				chatTemplateStr != nullptr ? u8_chat_template.c_str() : nullptr,
				chat_messages.data(), chat_messages.size(), addAssistantTokens,
				buf.data(), buf.size());

		// error: chat template is not supported
		if (resLength < 0) {
			if (chatTemplateStr != nullptr)
				throw std::runtime_error("Custom template is not supported");
			else
				throw std::runtime_error("Built-in template is not supported");
		}

		// if it turns out that our buffer is too small, we resize it
		if ((size_t) resLength > buf.size()) {
			buf.resize(resLength);
			resLength = llama_chat_apply_template(
					chatTemplateStr != nullptr ?
							u8_chat_template.c_str() : nullptr,
					chat_messages.data(), chat_messages.size(),
					addAssistantTokens, buf.data(), buf.size());
		}

		if (resLength < 0) {
			if (chatTemplateStr != nullptr)
				throw std::runtime_error("Custom template is not supported");
			else
				throw std::runtime_error("Built-in template is not supported");
		}

		std::string u8_res(buf.data(), resLength);
		return jjml_chat_to_byte_array(env, u8_res);
	} catch (std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
}

/*
 * CHAT (Jinja2 - uses common_chat_templates_apply, supports enable_thinking and chat_template_kwargs)
 */
JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_llm_LLamaCppNativeChatFormatter_doFormatChatMessagesJinja(
		JNIEnv *env, jclass, jlong modelPointer, jobjectArray roles, jobjectArray contents,
		jboolean addGenerationPrompt, jbyteArray chatTemplateStr,
		jboolean enableThinking, jobjectArray kwargsKeys, jobjectArray kwargsValues) {
	const jsize messages_size = env->GetArrayLength(roles);
	assert(env->GetArrayLength(contents) == messages_size);

	try {
		common_chat_params params = jjml_chat_apply_jinja(env, modelPointer,
				roles, contents, addGenerationPrompt, chatTemplateStr,
				enableThinking, kwargsKeys, kwargsValues, nullptr, nullptr,
				nullptr, nullptr, false, nullptr);
		return jjml_chat_to_byte_array(env, params.prompt);
	} catch (std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
}

JNIEXPORT jobject JNICALL Java_org_argeo_jjml_llm_LLamaCppNativeChatFormatter_doFormatChatMessagesJinjaFull(
		JNIEnv *env, jclass, jlong modelPointer, jobjectArray roles,
		jobjectArray contents, jboolean addGenerationPrompt,
		jbyteArray chatTemplateStr, jboolean enableThinking,
		jobjectArray kwargsKeys, jobjectArray kwargsValues,
		jobjectArray toolNames, jobjectArray toolDescriptions,
		jobjectArray toolParametersJson, jbyteArray toolChoice,
		jboolean parallelToolCalls, jbyteArray jsonSchema) {
	try {
		common_chat_params params = jjml_chat_apply_jinja(env, modelPointer,
				roles, contents, addGenerationPrompt, chatTemplateStr,
				enableThinking, kwargsKeys, kwargsValues, toolNames,
				toolDescriptions, toolParametersJson, toolChoice,
				parallelToolCalls, jsonSchema);
		return env->NewObject(argeo::jni::find_jclass(env, JCLASS_CHAT_FORMAT),
				LlamaCppChatFormat__init, jjml_chat_to_string(env, params.prompt),
				jjml_chat_to_string(env, params.grammar),
				static_cast<jboolean>(params.grammar_lazy),
				jjml_chat_to_string(env, params.generation_prompt),
				static_cast<jboolean>(params.supports_thinking),
				jjml_chat_to_string(env, params.thinking_start_tag),
				jjml_chat_to_string(env, params.thinking_end_tag),
				jjml_chat_grammar_triggers_to_java(env,
						params.grammar_triggers),
				jjml_chat_to_string(env, params.parser));
	} catch (std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
}

/*
 * Check if the model's chat template supports enable_thinking
 */
JNIEXPORT jboolean JNICALL Java_org_argeo_jjml_llm_LLamaCppNativeChatFormatter_doSupportsEnableThinking(
		JNIEnv *env, jclass, jlong modelPointer, jbyteArray chatTemplateStr) {
	try {
		auto *model = argeo::jni::as_pointer<llama_model*>(modelPointer);

		std::string u8_chat_template;
		if (chatTemplateStr != nullptr)
			u8_chat_template = argeo::jni::to_string(env, chatTemplateStr);

		auto tmpls = common_chat_templates_init(model, u8_chat_template);
		return common_chat_templates_support_enable_thinking(tmpls.get()) ? JNI_TRUE : JNI_FALSE;
	} catch (std::exception &ex) {
		return JNI_FALSE;
	}
}

JNIEXPORT jobject JNICALL Java_org_argeo_jjml_llm_LLamaCppNativeChatFormatter_doGetTemplateCapabilities(
		JNIEnv *env, jclass, jlong modelPointer, jbyteArray chatTemplateStr) {
	try {
		auto *model = argeo::jni::as_pointer<llama_model*>(modelPointer);

		std::string u8_chat_template;
		if (chatTemplateStr != nullptr)
			u8_chat_template = argeo::jni::to_string(env, chatTemplateStr);

		auto tmpls = common_chat_templates_init(model, u8_chat_template);
		std::map<std::string, bool> caps = common_chat_templates_get_caps(
				tmpls.get());
		return env->NewObject(
				argeo::jni::find_jclass(env,
						JCLASS_CHAT_TEMPLATE_CAPABILITIES),
				LlamaCppChatTemplateCapabilities__init,
				static_cast<jboolean>(jjml_chat_cap(caps, "supports_tools")),
				static_cast<jboolean>(jjml_chat_cap(caps,
						"supports_tool_calls")),
				static_cast<jboolean>(jjml_chat_cap(caps,
						"supports_system_role")),
				static_cast<jboolean>(jjml_chat_cap(caps,
						"supports_parallel_tool_calls")),
				static_cast<jboolean>(jjml_chat_cap(caps,
						"supports_preserve_reasoning")),
				static_cast<jboolean>(jjml_chat_cap(caps,
						"supports_string_content")),
				static_cast<jboolean>(jjml_chat_cap(caps,
						"supports_typed_content")),
				static_cast<jboolean>(jjml_chat_cap(caps,
						"supports_object_arguments")));
	} catch (std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
}
