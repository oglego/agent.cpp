# Minimal Agent Server Example

Exposes a single agent.cpp `Agent` over a plain HTTP API so it can be
used from any language, not just C++.

This is deliberately the **minimal** version:

- One model, loaded once at startup.
- One request handled at a time - a `std::mutex` around the agent loop
  serializes concurrent requests. There is no slot pool / multi-session
  concurrency.
- No streaming - each request blocks until the full response is ready,
  then returns one JSON object.
- The shared `CalculatorTool` (from `examples/shared/calculator_tool.h`)
  is registered as a demo of tool-calling over the HTTP API - swap in
  your own `Tool` subclasses the same way the `shell` or `memory`
  examples do.
- No auth, no rate limiting, no TLS.

A production version would add a `Model::create_with_weights`-based slot
pool for concurrency, wire `Agent::run_loop`'s `ResponseCallback` into a
Server-Sent-Events response for streaming, and accept tool definitions
per-request. Those are intentionally left out here to keep the example
small and easy to read end to end.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target server-example -j
```

## Run

```bash
./build/server-example -m /path/to/model.gguf -p 8080
```

Optional flags:

- `-s "<text>"` - system instructions for the agent.

## Use

```bash
curl -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

Response:

```json
{
  "messages": [
    { "role": "user", "content": "Hello!" },
    { "role": "assistant", "content": "Hi there! How can I help?" }
  ]
}
```

Send the full `messages` array back on the next request (including the
assistant's reply) to continue the conversation - the server itself is
stateless between requests, same as the OpenAI chat completions API.

Health check:

```bash
curl http://localhost:8080/health
```

## Seeing the tool in action

Ask something that forces a calculation, and the model will call the
registered `calculator` tool internally before replying:

```bash
curl -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 482 times 17?"}]}'
```

The `messages` array in the response only includes user/assistant turns
(the minimal JSON mapping doesn't round-trip tool_call/tool-result
messages) - but the final assistant reply will reflect the tool's
result, and you'll see the tool execute in the server's own stdout/
stderr if you add a logging `Callback` (see `logging_callback.h` in
`examples/shared`).
