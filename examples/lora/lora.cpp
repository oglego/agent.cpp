// lora.cpp — Demonstrates loading one or more LoRA adapters on top of a base
// model and running an interactive chat session.
//
// Usage:
//   lora-example -m model.gguf -l adapter.gguf [-s 0.8] [-l second.gguf [-s 1.0]]
//
// Multiple -l/-s pairs are accepted; -s always applies to the preceding -l.
// If -s is omitted the adapter's default scale (1.0) is used.

#include "agent.h"
#include "chat_loop.h"
#include "error.h"
#include "model.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

static void
print_usage(char** argv)
{
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf -l adapter.gguf\n", argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -m <path>   Path to the base GGUF model file (required)\n");
    printf("  -l <path>   Path to a LoRA adapter GGUF file (repeatable)\n");
    printf("  -s <scale>  Scale for the preceding -l adapter "
           "(default: 1.0, 0.0 disables)\n");
    printf("\n");
    printf("examples:\n");
    printf("  # Single adapter at default scale\n");
    printf("  %s -m base.gguf -l my-lora.gguf\n", argv[0]);
    printf("\n");
    printf("  # Single adapter at half strength\n");
    printf("  %s -m base.gguf -l my-lora.gguf -s 0.5\n", argv[0]);
    printf("\n");
    printf("  # Two adapters with individual scales\n");
    printf("  %s -m base.gguf -l style.gguf -s 0.8 -l task.gguf -s 1.0\n",
           argv[0]);
    printf("\n");
}

int
main(int argc, char** argv)
{
    std::string model_path;
    std::vector<agent_cpp::LoraAdapterConfig> lora_adapters;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) {
                model_path = argv[++i];
            } else {
                print_usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-l") == 0) {
            if (i + 1 < argc) {
                agent_cpp::LoraAdapterConfig cfg;
                cfg.path = argv[++i];
                lora_adapters.push_back(cfg);
            } else {
                print_usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0) {
            if (i + 1 < argc) {
                if (lora_adapters.empty()) {
                    fprintf(stderr,
                            "error: -s must follow a -l flag\n");
                    print_usage(argv);
                    return 1;
                }
                lora_adapters.back().scale =
                  std::stof(std::string(argv[++i]));
            } else {
                print_usage(argv);
                return 1;
            }
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            print_usage(argv);
            return 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "error: -m <model> is required\n");
        print_usage(argv);
        return 1;
    }

    // Report the adapters that will be loaded
    if (lora_adapters.empty()) {
        printf("No LoRA adapters specified — running base model only.\n");
    } else {
        printf("LoRA adapters to load:\n");
        for (const auto& lora : lora_adapters) {
            if (lora.scale == 0.0F) {
                printf("  [disabled] %s  (scale=0.0)\n",
                       lora.path.c_str());
            } else {
                printf("  [active]   %s  (scale=%.3f)\n",
                       lora.path.c_str(),
                       lora.scale);
            }
        }
    }

    agent_cpp::ModelConfig model_config;
    model_config.lora_adapters = lora_adapters;
    model_config.n_ctx         = 4096;
    model_config.temp          = 0.0F;

    printf("Loading model from '%s'...\n", model_path.c_str());
    std::shared_ptr<agent_cpp::Model> model;
    try {
        model = agent_cpp::Model::create(model_path, model_config);
    } catch (const agent_cpp::ModelError& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    printf("Model loaded successfully.\n");

    // Show which adapters are actually active after loading
    const auto& active = model->get_lora_adapters();
    if (!active.empty()) {
        printf("Active LoRA adapters applied to context:\n");
        for (const auto& lora : active) {
            printf("  %s  (scale=%.3f)\n",
                   lora.path.c_str(),
                   lora.scale);
        }
    }

    std::vector<std::unique_ptr<agent_cpp::Tool>> tools;

    const std::string instructions =
      "You are a helpful assistant. Answer the user's questions concisely "
      "and accurately.";

    agent_cpp::Agent agent(
      std::move(model), std::move(tools), {}, instructions);

    printf("\nLoRA Chat ready!\n");
    if (lora_adapters.empty()) {
        printf("   Running base model (no adapters loaded).\n");
    } else {
        printf("   Running with %zu LoRA adapter(s).\n", active.size());
    }
    printf("   Type an empty line to quit.\n\n");

    run_chat_loop(agent);
    return 0;
}
