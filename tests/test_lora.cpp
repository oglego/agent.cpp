#include "model.h"
#include "test_utils.h"
#include <string>
#include <vector>

// ── LoraAdapterConfig struct tests ────────────────────────────────────────────

TEST(test_lora_adapter_config_defaults)
{
    agent_cpp::LoraAdapterConfig config;

    ASSERT_TRUE(config.path.empty());
    ASSERT_EQ(config.scale, 1.0F);
}

TEST(test_lora_adapter_config_explicit_fields)
{
    agent_cpp::LoraAdapterConfig config;
    config.path  = "my-adapter.gguf";
    config.scale = 0.75F;

    ASSERT_STREQ(config.path.c_str(), "my-adapter.gguf");
    ASSERT_EQ(config.scale, 0.75F);
}

// ── filter_active_lora_adapters tests ─────────────────────────────────────────

TEST(test_filter_active_lora_adapters_empty_input)
{
    std::vector<agent_cpp::LoraAdapterConfig> adapters;

    std::vector<agent_cpp::LoraAdapterConfig> active =
      agent_cpp::filter_active_lora_adapters(adapters);

    ASSERT_TRUE(active.empty());
}

TEST(test_filter_active_lora_adapters_all_zero_scale)
{
    std::vector<agent_cpp::LoraAdapterConfig> adapters = {
        { "adapter-a.gguf", 0.0F },
        { "adapter-b.gguf", 0.0F },
    };

    std::vector<agent_cpp::LoraAdapterConfig> active =
      agent_cpp::filter_active_lora_adapters(adapters);

    ASSERT_TRUE(active.empty());
}

TEST(test_filter_active_lora_adapters_all_active)
{
    std::vector<agent_cpp::LoraAdapterConfig> adapters = {
        { "adapter-a.gguf", 1.0F },
        { "adapter-b.gguf", 0.5F },
        { "adapter-c.gguf", 2.0F },
    };

    std::vector<agent_cpp::LoraAdapterConfig> active =
      agent_cpp::filter_active_lora_adapters(adapters);

    ASSERT_EQ(active.size(), 3U);
    ASSERT_STREQ(active[0].path.c_str(), "adapter-a.gguf");
    ASSERT_STREQ(active[1].path.c_str(), "adapter-b.gguf");
    ASSERT_STREQ(active[2].path.c_str(), "adapter-c.gguf");
}

TEST(test_filter_active_lora_adapters_single_active)
{
    std::vector<agent_cpp::LoraAdapterConfig> adapters = {
        { "only.gguf", 0.8F },
    };

    std::vector<agent_cpp::LoraAdapterConfig> active =
      agent_cpp::filter_active_lora_adapters(adapters);

    ASSERT_EQ(active.size(), 1U);
    ASSERT_STREQ(active[0].path.c_str(), "only.gguf");
    ASSERT_EQ(active[0].scale, 0.8F);
}

TEST(test_filter_active_lora_adapters_drops_zero_scale_entries)
{
    std::vector<agent_cpp::LoraAdapterConfig> adapters = {
        { "adapter-a.gguf", 1.0F },
        { "adapter-b.gguf", 0.0F },
        { "adapter-c.gguf", 0.5F },
    };

    std::vector<agent_cpp::LoraAdapterConfig> active =
      agent_cpp::filter_active_lora_adapters(adapters);

    ASSERT_EQ(active.size(), 2U);
    ASSERT_STREQ(active[0].path.c_str(), "adapter-a.gguf");
    ASSERT_EQ(active[0].scale, 1.0F);
    ASSERT_STREQ(active[1].path.c_str(), "adapter-c.gguf");
    ASSERT_EQ(active[1].scale, 0.5F);
}

TEST(test_filter_active_lora_adapters_preserves_order)
{
    std::vector<agent_cpp::LoraAdapterConfig> adapters = {
        { "first.gguf", 0.25F },
        { "second.gguf", 0.0F },
        { "third.gguf", 2.0F },
        { "fourth.gguf", 0.0F },
    };

    std::vector<agent_cpp::LoraAdapterConfig> active =
      agent_cpp::filter_active_lora_adapters(adapters);

    ASSERT_EQ(active.size(), 2U);
    ASSERT_STREQ(active[0].path.c_str(), "first.gguf");
    ASSERT_STREQ(active[1].path.c_str(), "third.gguf");
}

// Negative scale is non-zero, so it counts as active. llama.cpp interprets
// scale linearly, so negative values invert the adapter's effect — they are
// intentionally allowed through, not filtered.
TEST(test_filter_active_lora_adapters_negative_scale_is_active)
{
    std::vector<agent_cpp::LoraAdapterConfig> adapters = {
        { "inverted.gguf", -0.5F },
        { "disabled.gguf", 0.0F },
    };

    std::vector<agent_cpp::LoraAdapterConfig> active =
      agent_cpp::filter_active_lora_adapters(adapters);

    ASSERT_EQ(active.size(), 1U);
    ASSERT_STREQ(active[0].path.c_str(), "inverted.gguf");
    ASSERT_EQ(active[0].scale, -0.5F);
}

// Scale values are preserved exactly as provided — there is no clamping.
TEST(test_filter_active_lora_adapters_scale_boundary_values)
{
    std::vector<agent_cpp::LoraAdapterConfig> adapters = {
        { "full.gguf", 1.0F },
        { "off.gguf", 0.0F },
        { "over.gguf", 2.0F },
        { "tiny.gguf", 0.001F },
    };

    std::vector<agent_cpp::LoraAdapterConfig> active =
      agent_cpp::filter_active_lora_adapters(adapters);

    ASSERT_EQ(active.size(), 3U);
    ASSERT_STREQ(active[0].path.c_str(), "full.gguf");
    ASSERT_STREQ(active[1].path.c_str(), "over.gguf");
    ASSERT_STREQ(active[2].path.c_str(), "tiny.gguf");
    ASSERT_EQ(active[0].scale, 1.0F);
    ASSERT_EQ(active[1].scale, 2.0F);
    ASSERT_EQ(active[2].scale, 0.001F);
}

int
main()
{
    std::cout << "\n=== Running LoRA Unit Tests ===\n" << std::endl;

    try {
        // LoraAdapterConfig struct
        RUN_TEST(test_lora_adapter_config_defaults);
        RUN_TEST(test_lora_adapter_config_explicit_fields);

        // filter_active_lora_adapters
        RUN_TEST(test_filter_active_lora_adapters_empty_input);
        RUN_TEST(test_filter_active_lora_adapters_all_zero_scale);
        RUN_TEST(test_filter_active_lora_adapters_all_active);
        RUN_TEST(test_filter_active_lora_adapters_single_active);
        RUN_TEST(test_filter_active_lora_adapters_drops_zero_scale_entries);
        RUN_TEST(test_filter_active_lora_adapters_preserves_order);
        RUN_TEST(test_filter_active_lora_adapters_negative_scale_is_active);
        RUN_TEST(test_filter_active_lora_adapters_scale_boundary_values);

        std::cout << "\n=== All tests passed! ✓ ===\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
