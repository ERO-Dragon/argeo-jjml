#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include <ggml-backend.h>

#include <argeo/jni/argeo_jni.h>

#include "org_argeo_jjml_ggml_GgmlBackend.h" // IWYU pragma: keep

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {
struct ggml_vk_scheduler_params {
	bool enabled;
	bool paused;
	bool abort_requested;
	int32_t max_nodes_per_chunk;
	uint64_t max_matmul_bytes_per_chunk;
	uint32_t time_window_us;
	uint32_t active_window_us;
	uint32_t max_chunks_in_flight;
	uint32_t sleep_granularity_us;
};

struct ggml_vk_scheduler_stats {
	uint64_t chunk_count;
	uint64_t last_chunk_wall_us;
	uint64_t last_chunk_nodes;
	uint64_t last_chunk_matmul_bytes;
	uint64_t slot_wait_count;
	uint64_t total_slot_wait_us;
	uint64_t last_slot_wait_us;
	uint64_t fence_wait_count;
	uint64_t total_fence_wait_us;
	uint64_t last_fence_wait_us;
};

using ggml_backend_vk_set_scheduler_params_t = void (*)(
		const ggml_vk_scheduler_params *);
using ggml_backend_vk_get_scheduler_params_t = void (*)(
		ggml_vk_scheduler_params *);
using ggml_backend_vk_get_scheduler_stats_t = void (*)(
		ggml_vk_scheduler_stats *);
using ggml_backend_vk_reset_scheduler_stats_t = void (*)();

bool is_backend_debug_enabled() {
	const char *value = std::getenv("JJML_GGML_BACKEND_DEBUG");
	return value && value[0] && value[0] != '0';
}

void *find_vulkan_scheduler_proc(const char *name) {
	ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
	if (!reg)
		return nullptr;
	return ggml_backend_reg_get_proc_address(reg, name);
}

template<typename T>
T find_vulkan_scheduler_proc(const char *name) {
	return reinterpret_cast<T>(find_vulkan_scheduler_proc(name));
}

bool has_vulkan_scheduler() {
	return find_vulkan_scheduler_proc(
			"ggml_backend_vk_set_scheduler_params")
			&& find_vulkan_scheduler_proc(
					"ggml_backend_vk_get_scheduler_params")
			&& find_vulkan_scheduler_proc(
					"ggml_backend_vk_get_scheduler_stats")
			&& find_vulkan_scheduler_proc(
					"ggml_backend_vk_reset_scheduler_stats");
}

#ifdef _WIN32
std::wstring utf8_to_wide(const std::string &value) {
	if (value.empty())
		return std::wstring();

	int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
			value.c_str(), static_cast<int>(value.size()), nullptr, 0);
	UINT code_page = CP_UTF8;
	DWORD flags = MB_ERR_INVALID_CHARS;
	if (size <= 0) {
		code_page = CP_ACP;
		flags = 0;
		size = MultiByteToWideChar(CP_ACP, 0, value.c_str(),
				static_cast<int>(value.size()), nullptr, 0);
	}
	if (size <= 0)
		return std::wstring();

	std::wstring wide(static_cast<size_t>(size), L'\0');
	if (!MultiByteToWideChar(code_page, flags, value.c_str(),
			static_cast<int>(value.size()), wide.data(), size))
		return std::wstring();
	return wide;
}
#endif

bool set_process_environment(const std::string &name, const std::string &value,
		bool overwrite) {
	if (name.empty())
		return false;
	if (!overwrite && std::getenv(name.c_str()) != nullptr)
		return true;
#ifdef _WIN32
	std::wstring wide_name = utf8_to_wide(name);
	std::wstring wide_value = utf8_to_wide(value);
	if (wide_name.empty())
		return false;
	return SetEnvironmentVariableW(wide_name.c_str(), wide_value.c_str()) ?
			true : false;
#else
	return setenv(name.c_str(), value.c_str(), overwrite ? 1 : 0) == 0;
#endif
}
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_ggml_GgmlBackend_doLoadBackend(
		JNIEnv *env, jclass, jbyteArray path) {
	std::string p = argeo::jni::to_string(env, path);
#ifdef GGML_BACKEND_DL
	if (is_backend_debug_enabled())
		std::cerr << "jjml: loading ggml backend " << p << std::endl;
	ggml_backend_reg_t reg = ggml_backend_load(p.c_str());
	if (is_backend_debug_enabled())
		std::cerr << "jjml: loaded ggml backend "
				<< (reg ? ggml_backend_reg_name(reg) : "<null>")
				<< std::endl;
	return reinterpret_cast<jlong>(reg);
#else
	// FIXME throw exception
#endif
	return 0;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_ggml_GgmlBackend_doLoadAllBackends(
		JNIEnv *env, jclass, jbyteArray basePath) {
	std::string search_path = argeo::jni::to_string(env, basePath);
#ifdef GGML_BACKEND_DL
	if (is_backend_debug_enabled())
		std::cerr << "jjml: loading all ggml backends from " << search_path
				<< std::endl;
	ggml_backend_load_all_from_path(search_path.c_str());
	if (is_backend_debug_enabled())
		std::cerr << "jjml: loaded all ggml backends from " << search_path
				<< std::endl;
#else
	// FIXME throw exception
#endif
}

extern "C" JNIEXPORT jboolean JNICALL Java_org_argeo_jjml_ggml_GgmlBackend_doSetProcessEnvironment(
		JNIEnv *env, jclass, jbyteArray name, jbyteArray value,
		jboolean overwrite) {
	std::string n = argeo::jni::to_string(env, name);
	std::string v = argeo::jni::to_string(env, value);
	return set_process_environment(n, v, overwrite == JNI_TRUE) ?
			JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_org_argeo_jjml_ggml_GgmlBackend_doIsVulkanSchedulerSupported(
		JNIEnv *, jclass) {
	return has_vulkan_scheduler();
}

extern "C" JNIEXPORT jboolean JNICALL Java_org_argeo_jjml_ggml_GgmlBackend_doSetVulkanSchedulerParams(
		JNIEnv *, jclass, jboolean enabled, jboolean paused,
		jboolean abortRequested, jint maxNodesPerChunk,
		jlong maxMatmulBytesPerChunk, jint timeWindowUs,
		jint activeWindowUs, jint maxChunksInFlight,
		jint sleepGranularityUs) {
	auto set_params = find_vulkan_scheduler_proc<
			ggml_backend_vk_set_scheduler_params_t>(
			"ggml_backend_vk_set_scheduler_params");
	if (!set_params)
		return false;

	ggml_vk_scheduler_params params { enabled == JNI_TRUE,
			paused == JNI_TRUE, abortRequested == JNI_TRUE,
			maxNodesPerChunk < 0 ? 0 : maxNodesPerChunk,
			maxMatmulBytesPerChunk < 0 ? 0
					: static_cast<uint64_t>(maxMatmulBytesPerChunk),
			timeWindowUs < 0 ? 0 : static_cast<uint32_t>(timeWindowUs),
			activeWindowUs < 0 ? 0 : static_cast<uint32_t>(activeWindowUs),
			maxChunksInFlight < 0 ? 0 : static_cast<uint32_t>(maxChunksInFlight),
			sleepGranularityUs < 0 ? 0 : static_cast<uint32_t>(sleepGranularityUs) };
	set_params(&params);
	return true;
}

extern "C" JNIEXPORT jlongArray JNICALL Java_org_argeo_jjml_ggml_GgmlBackend_doGetVulkanSchedulerParams(
		JNIEnv *env, jclass) {
	auto get_params = find_vulkan_scheduler_proc<
			ggml_backend_vk_get_scheduler_params_t>(
			"ggml_backend_vk_get_scheduler_params");
	if (!get_params)
		return nullptr;

	ggml_vk_scheduler_params params {};
	get_params(&params);

	jlong values[9] = { params.enabled ? 1 : 0, params.paused ? 1 : 0,
			params.abort_requested ? 1 : 0, params.max_nodes_per_chunk,
			static_cast<jlong>(params.max_matmul_bytes_per_chunk),
			params.time_window_us, params.active_window_us,
			params.max_chunks_in_flight, params.sleep_granularity_us };
	jlongArray result = env->NewLongArray(9);
	if (!result)
		return nullptr;
	env->SetLongArrayRegion(result, 0, 9, values);
	return result;
}

extern "C" JNIEXPORT jlongArray JNICALL Java_org_argeo_jjml_ggml_GgmlBackend_doGetVulkanSchedulerStats(
		JNIEnv *env, jclass) {
	auto get_stats = find_vulkan_scheduler_proc<
			ggml_backend_vk_get_scheduler_stats_t>(
			"ggml_backend_vk_get_scheduler_stats");
	if (!get_stats)
		return nullptr;

	ggml_vk_scheduler_stats stats {};
	get_stats(&stats);

	jlong values[10] = { static_cast<jlong>(stats.chunk_count),
			static_cast<jlong>(stats.last_chunk_wall_us),
			static_cast<jlong>(stats.last_chunk_nodes),
			static_cast<jlong>(stats.last_chunk_matmul_bytes),
			static_cast<jlong>(stats.slot_wait_count),
			static_cast<jlong>(stats.total_slot_wait_us),
			static_cast<jlong>(stats.last_slot_wait_us),
			static_cast<jlong>(stats.fence_wait_count),
			static_cast<jlong>(stats.total_fence_wait_us),
			static_cast<jlong>(stats.last_fence_wait_us) };
	jlongArray result = env->NewLongArray(10);
	if (!result)
		return nullptr;
	env->SetLongArrayRegion(result, 0, 10, values);
	return result;
}

extern "C" JNIEXPORT jboolean JNICALL Java_org_argeo_jjml_ggml_GgmlBackend_doResetVulkanSchedulerStats(
		JNIEnv *, jclass) {
	auto reset_stats = find_vulkan_scheduler_proc<
			ggml_backend_vk_reset_scheduler_stats_t>(
			"ggml_backend_vk_reset_scheduler_stats");
	if (!reset_stats)
		return false;

	reset_stats();
	return true;
}
