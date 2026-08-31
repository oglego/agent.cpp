// Minimal HTTP server for agent.cpp
//
// Serves a single agent over HTTP. Requests are handled one at a time
// (guarded by a mutex) - there is no concurrency/slot pooling and no
// streaming. This is intentionally the smallest useful version; see the
// README in this directory for what a production version would add.
//
//   POST /chat
//     { "messages": [{"role": "user", "content": "Hello!"}] }
//   ->
//     { "messages": [...full conversation with assistant reply...] }

#include "agent.h"
#include "calculator_tool.h"
#include "chat.h"
#include "error.h"
#include "model.h"
#include "tool.h"

#include "httplib.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using agent_cpp::json;

namespace {

// Convert an incoming JSON messages array into agent.cpp's chat message type.
// Only role/content are supported - this is the minimal version, so tool
// call round-tripping through the API is intentionally left out (the agent
// still executes any tools it's configured with internally).
std::vector<common_chat_msg>
messages_from_json(const json& messages_json)
{
    std::vector<common_chat_msg> messages;
    for (const auto& m : messages_json) {
        common_chat_msg msg;
        msg.role = m.at("role").get<std::string>();
        msg.content = m.at("content").get<std::string>();
        messages.push_back(msg);
    }
    return messages;
}

json
messages_to_json(const std::vector<common_chat_msg>& messages)
{
    json out = json::array();
    for (const auto& m : messages) {
        out.push_back({ { "role", m.role }, { "content", m.content } });
    }
    return out;
}

void
print_usage(char** argv)
{
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-p port] [-s system_instructions]\n",
           argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -m <path>   Path to the GGUF model file (required)\n");
    printf("  -p <port>   Port to listen on (default: 8080)\n");
    printf("  -s <text>   System instructions for the agent (optional)\n");
    printf("\n");
}

} // namespace

int
main(int argc, char** argv)
{
    std::string model_path;
    std::string instructions;
    int port = 8080;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            instructions = argv[++i];
        } else {
            print_usage(argv);
            return 1;
        }
    }

    if (model_path.empty()) {
        print_usage(argv);
        return 1;
    }

    printf("Loading model...\n");
    std::shared_ptr<agent_cpp::Model> model;
    try {
        model = agent_cpp::Model::create(model_path, agent_cpp::ModelConfig{});
    } catch (const agent_cpp::ModelError& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    printf("Model loaded successfully\n");

    // Registering the shared CalculatorTool here is just a demo of
    // tool-calling over the HTTP API - swap in your own Tool subclasses
    // the same way the other examples (shell, memory, etc.) do.
    std::vector<std::unique_ptr<agent_cpp::Tool>> tools;
    tools.push_back(std::make_unique<CalculatorTool>());
    std::vector<std::unique_ptr<agent_cpp::Callback>> callbacks;

    agent_cpp::Agent agent(
      std::move(model), std::move(tools), std::move(callbacks), instructions);

    // Agent/Model are not thread-safe (single KV cache) - this mutex
    // serializes requests so the server handles one at a time. A
    // concurrency-supporting version would replace this with a pool of
    // Model instances created via Model::create_with_weights, sharing one
    // set of weights across multiple contexts.
    std::mutex agent_mutex;

    httplib::Server svr;

    svr.Get("/health",
            [](const httplib::Request&, httplib::Response& res) {
                res.set_content(R"({"status":"ok"})", "application/json");
            });

    svr.Post(
      "/chat", [&](const httplib::Request& req, httplib::Response& res) {
          try {
              json body = json::parse(req.body);
              std::vector<common_chat_msg> messages =
                messages_from_json(body.at("messages"));

              {
                  std::lock_guard<std::mutex> lock(agent_mutex);
                  agent.run_loop(messages);
              }

              json response_body = { { "messages",
                                        messages_to_json(messages) } };
              res.set_content(response_body.dump(), "application/json");
          } catch (const agent_cpp::Error& e) {
              res.status = 500;
              res.set_content(json{ { "error", e.what() } }.dump(),
                               "application/json");
          } catch (const std::exception& e) {
              res.status = 400;
              res.set_content(json{ { "error", e.what() } }.dump(),
                               "application/json");
          }
      });

    printf("Agent server listening on http://localhost:%d\n", port);
    printf("Try: curl -X POST http://localhost:%d/chat "
           "-H 'Content-Type: application/json' "
           "-d '{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'\n\n",
           port);

    svr.listen("0.0.0.0", port);

    return 0;
}
