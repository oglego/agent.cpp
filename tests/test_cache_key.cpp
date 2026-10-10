#include "cache_key.h"
#include "test_utils.h"
#include <filesystem>
#include <fstream>

using agent_cpp::CacheKey;
using agent_cpp::CacheKeyInputs;

namespace fs = std::filesystem;

namespace {

struct TempDir
{
    fs::path path;
    TempDir()
    {
        path = fs::temp_directory_path() /
               ("agent_cpp_cache_key_" + std::to_string(std::rand()));
        fs::create_directories(path);
    }
    ~TempDir()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string file(const std::string& name, const std::string& content) const
    {
        fs::path p = path / name;
        std::ofstream(p, std::ios::binary) << content;
        return p.string();
    }
};

CacheKeyInputs
base_inputs()
{
    CacheKeyInputs in;
    in.n_ctx = 4096;
    in.cache_type_k = 1;
    in.cache_type_v = 1;
    return in;
}

} // namespace

TEST(test_identical_inputs_match)
{
    TempDir dir;
    auto a = dir.file("a.gguf", "adapter-a");
    auto in = base_inputs();
    in.loras.push_back({ a, 1.0F });

    ASSERT_EQ(CacheKey::from_inputs(in).mismatch(CacheKey::from_inputs(in)),
              std::string());
}

TEST(test_no_adapters_vs_adapter_mismatch)
{
    TempDir dir;
    auto a = dir.file("a.gguf", "adapter-a");
    auto none = base_inputs();
    auto with = base_inputs();
    with.loras.push_back({ a, 1.0F });

    auto reason =
      CacheKey::from_inputs(with).mismatch(CacheKey::from_inputs(none));
    ASSERT_FALSE(reason.empty());
    ASSERT_TRUE(reason.find("adapters") != std::string::npos);
}

TEST(test_different_adapter_file_mismatch)
{
    TempDir dir;
    auto a = dir.file("a.gguf", "adapter-a");
    auto b = dir.file("b.gguf", "adapter-b");
    auto in_a = base_inputs();
    auto in_b = base_inputs();
    in_a.loras.push_back({ a, 1.0F });
    in_b.loras.push_back({ b, 1.0F });

    ASSERT_FALSE(CacheKey::from_inputs(in_a)
                   .mismatch(CacheKey::from_inputs(in_b))
                   .empty());
}

TEST(test_different_base_model_mismatch)
{
    auto first = base_inputs();
    auto second = base_inputs();
    first.model_identity = "model-a";
    second.model_identity = "model-b";

    ASSERT_TRUE(CacheKey::from_inputs(first)
                  .mismatch(CacheKey::from_inputs(second))
                  .find("model") != std::string::npos);
}

TEST(test_scale_change_mismatch)
{
    TempDir dir;
    auto a = dir.file("a.gguf", "adapter-a");
    auto full = base_inputs();
    auto half = base_inputs();
    full.loras.push_back({ a, 1.0F });
    half.loras.push_back({ a, 0.5F });

    ASSERT_FALSE(CacheKey::from_inputs(full)
                   .mismatch(CacheKey::from_inputs(half))
                   .empty());
}

TEST(test_adapter_modified_in_place_mismatch)
{
    TempDir dir;
    auto a = dir.file("a.gguf", "adapter-a");
    auto in = base_inputs();
    in.loras.push_back({ a, 1.0F });
    auto before = CacheKey::from_inputs(in);

    dir.file("a.gguf", "adapter-a-retrained-longer"); // size changes

    ASSERT_FALSE(CacheKey::from_inputs(in).mismatch(before).empty());
}

TEST(test_adapter_identity_is_a_load_time_snapshot)
{
    TempDir dir;
    auto a = dir.file("a.gguf", "adapter-a");
    auto loaded = base_inputs();
    loaded.loras.push_back({ a, 1.0F, CacheKey::file_identity(a) });
    auto before = CacheKey::from_inputs(loaded);

    dir.file("a.gguf", "adapter-replaced-after-load");
    ASSERT_EQ(CacheKey::from_inputs(loaded).mismatch(before), std::string());

    auto reloaded = base_inputs();
    reloaded.loras.push_back({ a, 1.0F, CacheKey::file_identity(a) });
    ASSERT_FALSE(CacheKey::from_inputs(reloaded).mismatch(before).empty());
}

TEST(test_adapter_order_does_not_matter)
{
    TempDir dir;
    auto a = dir.file("a.gguf", "adapter-a");
    auto b = dir.file("b.gguf", "adapter-b");
    auto ab = base_inputs();
    auto ba = base_inputs();
    ab.loras = { { a, 1.0F }, { b, 0.5F } };
    ba.loras = { { b, 0.5F }, { a, 1.0F } };

    ASSERT_EQ(CacheKey::from_inputs(ab).mismatch(CacheKey::from_inputs(ba)),
              std::string());
}

TEST(test_zero_scale_adapter_ignored)
{
    TempDir dir;
    auto a = dir.file("a.gguf", "adapter-a");
    auto none = base_inputs();
    auto zero = base_inputs();
    zero.loras.push_back({ a, 0.0F });

    ASSERT_EQ(CacheKey::from_inputs(zero).mismatch(CacheKey::from_inputs(none)),
              std::string());
}

TEST(test_relative_and_absolute_path_agree)
{
    TempDir dir;
    auto a = dir.file("a.gguf", "adapter-a");
    auto abs_in = base_inputs();
    auto rel_in = base_inputs();
    abs_in.loras.push_back({ a, 1.0F });
    rel_in.loras.push_back({ (dir.path / "." / "a.gguf").string(), 1.0F });

    ASSERT_EQ(
      CacheKey::from_inputs(abs_in).mismatch(CacheKey::from_inputs(rel_in)),
      std::string());
}

TEST(test_non_adapter_settings_mismatch)
{
    auto base = base_inputs();
    auto ctx = base_inputs();
    ctx.n_ctx = 8192;
    auto kv = base_inputs();
    kv.cache_type_k = 8;

    auto ctx_reason =
      CacheKey::from_inputs(ctx).mismatch(CacheKey::from_inputs(base));
    ASSERT_TRUE(ctx_reason.find("n_ctx") != std::string::npos);
    auto kv_reason =
      CacheKey::from_inputs(kv).mismatch(CacheKey::from_inputs(base));
    ASSERT_TRUE(kv_reason.find("cache_type_k") != std::string::npos);
}

TEST(test_serialize_parse_roundtrip)
{
    TempDir dir;
    auto a = dir.file("a.gguf", "adapter-a");
    auto in = base_inputs();
    in.loras.push_back({ a, 0.75F });
    auto key = CacheKey::from_inputs(in);

    CacheKey parsed;
    ASSERT_TRUE(CacheKey::parse(key.serialize(), parsed));
    ASSERT_EQ(key.mismatch(parsed), std::string());
}

TEST(test_parse_rejects_garbage)
{
    CacheKey out;
    ASSERT_FALSE(CacheKey::parse("", out));
    ASSERT_FALSE(CacheKey::parse("not a key value line\n", out));
}

TEST(test_sidecar_write_read_remove)
{
    TempDir dir;
    auto cache = (dir.path / "prompt.cache").string();
    auto key = CacheKey::from_inputs(base_inputs());

    CacheKey out;
    ASSERT_FALSE(CacheKey::read(cache, out)); // missing sidecar
    ASSERT_TRUE(key.write(cache));
    ASSERT_TRUE(fs::exists(cache + ".meta"));
    ASSERT_FALSE(fs::exists(cache + ".meta.tmp"));
    ASSERT_TRUE(CacheKey::read(cache, out));
    ASSERT_EQ(key.mismatch(out), std::string());

    CacheKey::remove(cache);
    ASSERT_FALSE(CacheKey::read(cache, out));
}

TEST(test_truncated_sidecar_is_a_mismatch_not_a_match)
{
    TempDir dir;
    auto cache = (dir.path / "prompt.cache").string();
    auto key = CacheKey::from_inputs(base_inputs());
    ASSERT_TRUE(key.write(cache));

    // Simulate a partially written sidecar: only the first line survives.
    std::ofstream(cache + ".meta", std::ios::trunc) << "cache_type_k=1\n";

    CacheKey stored;
    if (CacheKey::read(cache, stored)) {
        ASSERT_FALSE(key.mismatch(stored).empty());
    }
}

int
main()
{
    std::cout << "\n=== Running Cache Key Unit Tests ===\n" << std::endl;

    try {
        RUN_TEST(test_identical_inputs_match);
        RUN_TEST(test_no_adapters_vs_adapter_mismatch);
        RUN_TEST(test_different_adapter_file_mismatch);
        RUN_TEST(test_different_base_model_mismatch);
        RUN_TEST(test_scale_change_mismatch);
        RUN_TEST(test_adapter_modified_in_place_mismatch);
        RUN_TEST(test_adapter_identity_is_a_load_time_snapshot);
        RUN_TEST(test_adapter_order_does_not_matter);
        RUN_TEST(test_zero_scale_adapter_ignored);
        RUN_TEST(test_relative_and_absolute_path_agree);
        RUN_TEST(test_non_adapter_settings_mismatch);
        RUN_TEST(test_serialize_parse_roundtrip);
        RUN_TEST(test_parse_rejects_garbage);
        RUN_TEST(test_sidecar_write_read_remove);
        RUN_TEST(test_truncated_sidecar_is_a_mismatch_not_a_match);

        std::cout << "\n=== All tests passed! ✓ ===\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
