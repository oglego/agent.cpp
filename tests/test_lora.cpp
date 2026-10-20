#include "error.h"
#include "cache_metadata.h"
#include "model.h"
#include "test_utils.h"

TEST(test_model_config_loras_defaults)
{
    agent_cpp::ModelConfig config;

    ASSERT_TRUE(config.loras.empty());
}

TEST(test_lora_adapter_config_default_scale)
{
    agent_cpp::LoraAdapterConfig lora_config;
    lora_config.path = "adapter.gguf";

    ASSERT_EQ(lora_config.scale, 1.0F);
}

TEST(test_model_config_loras_accepts_multiple_adapters)
{
    agent_cpp::ModelConfig config;
    config.loras.push_back({ "math_lora.gguf", 1.0F });
    config.loras.push_back({ "style_lora.gguf", 0.5F });

    ASSERT_EQ(config.loras.size(), (size_t)2);
    ASSERT_EQ(config.loras[0].path, "math_lora.gguf");
    ASSERT_EQ(config.loras[1].scale, 0.5F);
}

TEST(test_prompt_cache_metadata_includes_base_model_fingerprint)
{
    agent_cpp::ModelConfig config;
    config.loras = { { "adapter.gguf", 0.75F } };
    const std::vector<std::string> adapter_fingerprints{ "adapter-hash" };

    const auto first = agent_cpp::detail::cache_metadata(
      "base-model-one", config, adapter_fingerprints);
    const auto same_model = agent_cpp::detail::cache_metadata(
      "base-model-one", config, adapter_fingerprints);
    const auto different_model = agent_cpp::detail::cache_metadata(
      "base-model-two", config, adapter_fingerprints);
    const auto different_adapter = agent_cpp::detail::cache_metadata(
      "base-model-one", config, { "different-adapter-hash" });
    auto different_context_size_config = config;
    different_context_size_config.n_ctx += 1;
    const auto different_context_size = agent_cpp::detail::cache_metadata(
      "base-model-one", different_context_size_config, adapter_fingerprints);
    auto different_key_cache_config = config;
    different_key_cache_config.cache_type_k = GGML_TYPE_F32;
    const auto different_key_cache_type = agent_cpp::detail::cache_metadata(
      "base-model-one", different_key_cache_config, adapter_fingerprints);
    auto different_value_cache_config = config;
    different_value_cache_config.cache_type_v = GGML_TYPE_F32;
    const auto different_value_cache_type = agent_cpp::detail::cache_metadata(
      "base-model-one", different_value_cache_config, adapter_fingerprints);
    auto different_scale_config = config;
    different_scale_config.loras[0].scale = 1.0F;
    const auto different_scale = agent_cpp::detail::cache_metadata(
      "base-model-one", different_scale_config, adapter_fingerprints);

    ASSERT_FALSE(first.empty());
    ASSERT_EQ(first, same_model);
    ASSERT_FALSE(first == different_model);
    ASSERT_FALSE(first == different_adapter);
    ASSERT_FALSE(first == different_context_size);
    ASSERT_FALSE(first == different_key_cache_type);
    ASSERT_FALSE(first == different_value_cache_type);
    ASSERT_FALSE(first == different_scale);
}

TEST(test_prompt_cache_metadata_rejects_missing_fingerprints)
{
    agent_cpp::ModelConfig config;
    config.loras = { { "adapter.gguf", 1.0F } };

    ASSERT_TRUE(agent_cpp::detail::cache_metadata(
                  "", config, { "adapter-hash" })
                  .empty());
    ASSERT_TRUE(agent_cpp::detail::cache_metadata(
                  "base-model-hash", config, {})
                  .empty());
}

// Full adapter loading requires a loaded GGUF model and context. Verify
// manually in examples/lora.

int
main()
{
    std::cout << "\n=== Running LoRA Unit Tests ===\n" << std::endl;

    try {
        RUN_TEST(test_model_config_loras_defaults);
        RUN_TEST(test_lora_adapter_config_default_scale);
        RUN_TEST(test_model_config_loras_accepts_multiple_adapters);
        RUN_TEST(test_prompt_cache_metadata_includes_base_model_fingerprint);
        RUN_TEST(test_prompt_cache_metadata_rejects_missing_fingerprints);

        std::cout << "\n=== All tests passed! ✓ ===\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
