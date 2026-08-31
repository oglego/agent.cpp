# Examples

This directory contains example applications demonstrating agent.cpp capabilities.

## Grammar

The [grammar](./grammar) example demonstrates constraining model output to a GBNF grammar, so every response is guaranteed to match a fixed structure. You can also point it at a custom grammar file and root rule.

## LoRA

The [lora](./lora) example demonstrates loading one or more LoRA adapters on top of a base model and running an interactive chat session.

```
# Single adapter at full strength
lora-example -m base.gguf -l my-lora.gguf

# Single adapter at half strength
lora-example -m base.gguf -l my-lora.gguf -s 0.5

# Two adapters with individual scales
lora-example -m base.gguf -l style.gguf -s 0.8 -l task.gguf -s 1.0
```

Adapters must be in GGUF format (use llama.cpp's `convert_lora_to_gguf.py` to convert from the standard HuggingFace safetensors format). Each `-l` flag can optionally be followed by a `-s <scale>` flag to control how strongly that adapter is applied — `1.0` is full strength, `0.0` disables the adapter without unloading it.

## Shared Utilities

The [shared](./shared) directory contains reusable helper components used across multiple examples. These are **not part of the public API** but can be useful as reference implementations.

| File | Description |
|------|-------------|
| `calculator_tool.h` | A simple calculator tool for basic math operations (add, subtract, multiply, divide). Demonstrates how to implement a `Tool` with JSON Schema parameters. |
| `chat_loop.h` | Interactive chat loop that reads user input from stdin and prints agent responses. Handles colored output for TTY terminals. |
| `error_recovery_callback.h` | Callback that converts tool errors into JSON results, allowing the agent to see errors and retry gracefully instead of crashing. |
| `logging_callback.h` | Callback that logs tool calls and their results to stderr. Useful for debugging and understanding agent behavior. |
