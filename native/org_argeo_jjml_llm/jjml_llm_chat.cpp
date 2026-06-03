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
		jbyteArray res = env->NewByteArray(u8_res.length());
		env->SetByteArrayRegion(res, 0, u8_res.length(), (jbyte*) &u8_res[0]);
		return res;
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
		auto *model = argeo::jni::as_pointer<llama_model*>(modelPointer);

		// Build messages
		std::vector<common_chat_msg> chat_messages;
		for (int i = 0; i < messages_size; i++) {
			common_chat_msg msg;
			msg.role = argeo::jni::to_string(env, roles, i);
			msg.content = argeo::jni::to_string(env, contents, i);
			chat_messages.push_back(msg);
		}

		// Chat template override
		std::string u8_chat_template;
		if (chatTemplateStr != nullptr)
			u8_chat_template = argeo::jni::to_string(env, chatTemplateStr);

		// Initialize chat templates from model
		auto tmpls = common_chat_templates_init(model, u8_chat_template);

		// Build chat_template_kwargs from parallel arrays
		std::map<std::string, std::string> template_kwargs;
		if (kwargsKeys != nullptr && kwargsValues != nullptr) {
			const jsize kwargs_size = env->GetArrayLength(kwargsKeys);
			assert(env->GetArrayLength(kwargsValues) == kwargs_size);
			for (int i = 0; i < kwargs_size; i++) {
				std::string key = argeo::jni::to_string(env, kwargsKeys, i);
				std::string val = argeo::jni::to_string(env, kwargsValues, i);
				template_kwargs[key] = val;
			}
		}

		// Build inputs
		common_chat_templates_inputs inputs;
		inputs.messages = chat_messages;
		inputs.add_generation_prompt = addGenerationPrompt;
		inputs.use_jinja = true;
		inputs.enable_thinking = enableThinking;
		inputs.chat_template_kwargs = template_kwargs;

		// Apply template
		common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);

		std::string u8_res = params.prompt;
		jbyteArray res = env->NewByteArray(u8_res.length());
		env->SetByteArrayRegion(res, 0, u8_res.length(), (jbyte*) u8_res.c_str());
		return res;
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
