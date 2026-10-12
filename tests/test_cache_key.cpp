#include "cache_key.h"
#include "test_utils.h"

#include <chrono>
#include <filesystem>
#include <fstream>

using namespace agent_cpp;
namespace fs = std::filesystem;

namespace {

uint64_t
fnv(const std::string& s)
{
    Fnv1a h;
    h.update(s.data(), s.size());
    return h.value();
}

// Temp file removed on scope exit
struct TempFile
{
    fs::path path;
    TempFile(const std::string& name, const std::string& content)
    {
        path = fs::temp_directory_path() /
               (name + "." +
                std::to_string(
                  std::chrono::steady_clock::now().time_since_epoch().count()));
        std::ofstream(path, std::ios::binary) << content;
    }
    ~TempFile()
    {
        std::error_code ec;
        fs::remove(path, ec);
    }
};

const CacheContext kCtx{ 4096, 1, 1, 10 };

} // namespace

TEST(test_fnv1a_known_vectors)
{
    ASSERT_TRUE(fnv("") == 0xcbf29ce484222325ULL);
    ASSERT_TRUE(fnv("a") == 0xaf63dc4c8601ec8cULL);
    ASSERT_TRUE(fnv("foobar") == 0x85944171f73967e8ULL);
}

TEST(test_hash_file_hashes_content_only)
{
    TempFile a("agentcpp_a", "same bytes");
    TempFile b("agentcpp_b", "same bytes");
    fs::last_write_time(b.path, fs::last_write_time(a.path) + std::chrono::hours(5));
    ASSERT_TRUE(*hash_file(a.path) == fnv("same bytes"));
    ASSERT_TRUE(*hash_file(a.path) == *hash_file(b.path));
    ASSERT_FALSE(hash_file("/nonexistent/adapter.gguf").has_value());
}

TEST(test_file_identity_changes_with_size)
{
    TempFile f("agentcpp_id", "12345");
    auto before = file_identity(f.path);
    std::ofstream(f.path, std::ios::binary | std::ios::app) << "6";
    auto after = file_identity(f.path);
    ASSERT_TRUE(before.has_value() && after.has_value());
    ASSERT_FALSE(*before == *after);
    ASSERT_FALSE(file_identity("/nonexistent/model.gguf").has_value());
}

TEST(test_digest_independent_of_adapter_order)
{
    ASSERT_TRUE(cache_digest(1, { { 10, 1.0F }, { 20, 0.5F } }, kCtx) ==
                cache_digest(1, { { 20, 0.5F }, { 10, 1.0F } }, kCtx));
}

TEST(test_digest_differs_by_adapter_scale_and_set)
{
    auto base = cache_digest(1, { { 10, 1.0F } }, kCtx);
    ASSERT_FALSE(base == cache_digest(1, { { 11, 1.0F } }, kCtx));
    ASSERT_FALSE(base == cache_digest(1, { { 10, 0.5F } }, kCtx));
    ASSERT_FALSE(base == cache_digest(1, {}, kCtx));
    ASSERT_FALSE(base == cache_digest(1, { { 10, 1.0F }, { 11, 1.0F } }, kCtx));
    // Same adapter stacked twice is a different model
    ASSERT_FALSE(base == cache_digest(1, { { 10, 1.0F }, { 10, 1.0F } }, kCtx));
}

TEST(test_digest_ignores_zero_scale_adapters)
{
    ASSERT_TRUE(cache_digest(1, { { 10, 1.0F }, { 20, 0.0F } }, kCtx) ==
                cache_digest(1, { { 10, 1.0F } }, kCtx));
    ASSERT_TRUE(cache_digest(1, { { 10, 0.0F } }, kCtx) ==
                cache_digest(1, {}, kCtx));
}

TEST(test_digest_depends_on_model_and_every_context_field)
{
    auto base = cache_digest(1, {}, kCtx);
    ASSERT_TRUE(base == cache_digest(1, {}, kCtx));
    ASSERT_FALSE(base == cache_digest(2, {}, kCtx));

    auto c = kCtx;
    c.n_ctx = 8192;
    ASSERT_FALSE(base == cache_digest(1, {}, c));
    c = kCtx;
    c.cache_type_k = 8;
    ASSERT_FALSE(base == cache_digest(1, {}, c));
    c = kCtx;
    c.cache_type_v = 8;
    ASSERT_FALSE(base == cache_digest(1, {}, c));
    c = kCtx;
    c.session_version = 11;
    ASSERT_FALSE(base == cache_digest(1, {}, c));
}

TEST(test_digest_fields_do_not_run_together)
{
    ASSERT_FALSE(cache_digest(1, {}, { 1, 2, 0, 0 }) ==
                 cache_digest(1, {}, { 2, 1, 0, 0 }));
}

TEST(test_keyed_cache_path_format)
{
    ASSERT_STREQ(keyed_cache_path("dir/prompt.bin", 0xabcdef0123456789ULL).string(),
                 "dir/prompt.abcdef0123456789.bin");
    ASSERT_STREQ(keyed_cache_path("prompt.bin", 0x1ULL).string(),
                 "prompt.0000000000000001.bin");
    ASSERT_STREQ(keyed_cache_path("prompt", 0x1ULL).string(),
                 "prompt.0000000000000001");
}

int
main()
{
    std::cout << "\n=== Running Cache Key Unit Tests ===\n" << std::endl;
    try {
        RUN_TEST(test_fnv1a_known_vectors);
        RUN_TEST(test_hash_file_hashes_content_only);
        RUN_TEST(test_file_identity_changes_with_size);
        RUN_TEST(test_digest_independent_of_adapter_order);
        RUN_TEST(test_digest_differs_by_adapter_scale_and_set);
        RUN_TEST(test_digest_ignores_zero_scale_adapters);
        RUN_TEST(test_digest_depends_on_model_and_every_context_field);
        RUN_TEST(test_digest_fields_do_not_run_together);
        RUN_TEST(test_keyed_cache_path_format);
        std::cout << "\n=== All Cache Key Tests Passed ===\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ FAILED: " << e.what() << std::endl;
        return 1;
    }
}
