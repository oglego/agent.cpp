#include "model.h"
#include "chat.h"
#include "error.h"
#include "tool.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace agent_cpp {

namespace {

// Fallback parser for models whose actual tool-call output doesn't match
// the grammar built from their chat template (e.g. observed with
// granite-4.0-micro: the model emits Hermes-style
// <tool_call>{...}</tool_call> tags, but the template-derived PEG grammar
// doesn't recognize them - a mismatch we've traced to llama.cpp's
// chat-template autoparser, not something fixable from here). Extracts
// every <tool_call>...</tool_call> block and parses each JSON body
// directly. A malformed individual block is skipped rather than aborting
// the whole extraction; returns true if at least one call was recovered.
bool
parse_tool_call_xml_fallback(const std::string& raw, common_chat_msg& parsed_msg)
{
    parsed_msg = {};
    parsed_msg.role = "assistant";

    const std::string open_tag = "<tool_call";
    const std::string close_tag = "</tool_call>";

    const auto trim = [](std::string s) {
        const auto is_ws = [](unsigned char ch) {
            return ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t';
        };
        while (!s.empty() && is_ws(static_cast<unsigned char>(s.front()))) {
            s.erase(s.begin());
        }
        while (!s.empty() && is_ws(static_cast<unsigned char>(s.back()))) {
            s.pop_back();
        }
        return s;
    };

    std::string::size_type search_pos = 0;
    while (true) {
        const std::string::size_type start = raw.find(open_tag, search_pos);
        if (start == std::string::npos) {
            break;
        }

        const std::string::size_type end = raw.find(close_tag, start);
        if (end == std::string::npos) {
            // Unterminated tag - nothing more to recover after this point.
            break;
        }

        std::string body = raw.substr(start, end - start + close_tag.size());
        search_pos = end + close_tag.size();

        const std::string::size_type gt = body.find('>');
        if (gt == std::string::npos) {
            continue;
        }
        body = trim(body.substr(gt + 1));

        const std::string::size_type first_json = body.find('{');
        const std::string::size_type last_json = body.rfind('}');
        if (first_json == std::string::npos ||
            last_json == std::string::npos || last_json < first_json) {
            continue;
        }

        const std::string candidate =
          body.substr(first_json, last_json - first_json + 1);
        try {
            const json j = json::parse(candidate);
            if (!j.is_object()) {
                continue;
            }

            const std::string name = j.value("name", "");
            if (name.empty()) {
                continue;
            }

            common_chat_tool_call call;
            call.name = name;
            call.arguments =
              j.contains("arguments") ? j.at("arguments").dump() : "{}";
            parsed_msg.tool_calls.push_back(call);
        } catch (const std::exception&) {
            continue;
        }
    }

    return !parsed_msg.tool_calls.empty();
}

} // namespace

std::string
load_grammar_file(const std::string& grammar_path)
{
    std::ifstream file(grammar_path);
    if (!file) {
        throw ModelError("failed to open grammar file '" + grammar_path + "'");
    }

    std::ostringstream contents;
    contents << file.rdbuf();

    std::string grammar = contents.str();
    if (grammar.empty()) {
        throw ModelError("grammar file '" + grammar_path + "' is empty");
    }
    return grammar;
}

std::shared_ptr<ModelWeights>
ModelWeights::create(const std::string& model_path,
                     const std::string& chat_template_override)
{
    std::shared_ptr<ModelWeights> weights(new ModelWeights());

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    weights->model_ =
      llama_model_load_from_file(model_path.c_str(), model_params);
    if (weights->model_ == nullptr) {
        throw ModelError("unable to load model from '" + model_path + "'");
    }

    auto tmpls = common_chat_templates_init(weights->model_,
                                            chat_template_override);
    if (!tmpls) {
        throw ModelError("failed to initialize chat templates");
    }
    weights->templates_ =
      std::shared_ptr<common_chat_templates>(std::move(tmpls));

    return weights;
}

ModelWeights::~ModelWeights()
{
    if (model_ != nullptr) {
        llama_model_free(model_);
    }
}

std::shared_ptr<Model>
Model::create(const std::string& model_path, const ModelConfig& model_config)
{
    auto weights =
      ModelWeights::create(model_path, model_config.chat_template_override);
    return create_with_weights(std::move(weights), model_config);
}

std::shared_ptr<Model>
Model::create_with_weights(std::shared_ptr<ModelWeights> weights,
                           const ModelConfig& model_config)
{
    std::shared_ptr<Model> model(new Model());
    model->weights_ = std::move(weights);
    model->initialize_context(model_config);
    return model;
}

Model::~Model()
{
    if (sampler_ != nullptr) {
        llama_sampler_free(sampler_);
    }
    if (ctx_ != nullptr) {
        llama_free(ctx_);
    }
    // weights_ is automatically released when ref count drops to zero
}

Model::Model(Model&& other) noexcept
  : weights_(std::move(other.weights_))
  , ctx_(other.ctx_)
  , sampler_(other.sampler_)
  , grammar_sampler_(other.grammar_sampler_)
  , processed_tokens_(std::move(other.processed_tokens_))
  , n_past_(other.n_past_)
  , config_(other.config_)
{
    other.ctx_ = nullptr;
    other.sampler_ = nullptr;
    other.grammar_sampler_ = nullptr;
    other.n_past_ = 0;
}

Model&
Model::operator=(Model&& other) noexcept
{
    if (this != &other) {
        if (sampler_ != nullptr) {
            llama_sampler_free(sampler_);
        }
        if (ctx_ != nullptr) {
            llama_free(ctx_);
        }

        weights_ = std::move(other.weights_);
        ctx_ = other.ctx_;
        sampler_ = other.sampler_;
        grammar_sampler_ = other.grammar_sampler_;
        processed_tokens_ = std::move(other.processed_tokens_);
        n_past_ = other.n_past_;
        config_ = other.config_;

        other.ctx_ = nullptr;
        other.sampler_ = nullptr;
        other.grammar_sampler_ = nullptr;
        other.n_past_ = 0;
    }
    return *this;
}

void
Model::initialize_context(const ModelConfig& model_config)
{
    config_ = model_config;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = model_config.n_ctx;
    ctx_params.n_batch = model_config.n_batch;
    ctx_params.n_threads = model_config.n_threads;
    ctx_params.n_threads_batch = model_config.n_threads_batch;
    ctx_params.type_k = model_config.cache_type_k;
    ctx_params.type_v = model_config.cache_type_v;

    ctx_ = llama_init_from_model(weights_->get_model(), ctx_params);
    if (ctx_ == nullptr) {
        throw ModelError("failed to create llama context");
    }

    sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());

    if (!model_config.grammar.empty()) {
        llama_sampler* grammar_sampler =
          llama_sampler_init_grammar(weights_->get_vocab(),
                                     model_config.grammar.c_str(),
                                     model_config.grammar_root.c_str());
        if (grammar_sampler == nullptr) {
            llama_sampler_free(sampler_);
            sampler_ = nullptr;
            throw ModelError("failed to parse GBNF grammar (root rule '" +
                             model_config.grammar_root + "')");
        }
        // Add grammar before the rest of the sampler chain
        llama_sampler_chain_add(sampler_, grammar_sampler);
        // Keep a non-owning reference so we can reset this sampler between
        // turns
        grammar_sampler_ = grammar_sampler;
    }

    llama_sampler_chain_add(sampler_,
                            llama_sampler_init_top_k(model_config.top_k));
    llama_sampler_chain_add(sampler_,
                            llama_sampler_init_top_p(model_config.top_p, 1));
    llama_sampler_chain_add(sampler_,
                            llama_sampler_init_min_p(model_config.min_p, 1));
    llama_sampler_chain_add(sampler_,
                            llama_sampler_init_temp(model_config.temp));
    llama_sampler_chain_add(sampler_,
                            llama_sampler_init_dist(model_config.seed));
}

std::vector<llama_token>
Model::tokenize(const std::string& prompt) const
{
    const llama_vocab* vocab = weights_->get_vocab();
    // Use processed_tokens to determine if this is the first tokenization
    // This is important for cache loading: even if KV cache memory is
    // populated, we need IS_FIRST=true if we're tokenizing a full prompt from
    // scratch to ensure consistent BOS token handling for prefix matching
    const bool IS_FIRST = processed_tokens_.empty();

    const int N_PROMPT_TOKENS = -llama_tokenize(
      vocab, prompt.c_str(), prompt.size(), nullptr, 0, IS_FIRST, true);
    std::vector<llama_token> prompt_tokens(N_PROMPT_TOKENS);
    if (llama_tokenize(vocab,
                       prompt.c_str(),
                       prompt.size(),
                       prompt_tokens.data(),
                       prompt_tokens.size(),
                       IS_FIRST,
                       true) < 0) {
        return {}; // Return empty vector on failure
    }
    return prompt_tokens;
}

common_chat_msg
Model::generate(const std::vector<common_chat_msg>& messages,
                const std::vector<common_chat_tool>& tools,
                const ResponseCallback& callback)
{
    common_chat_templates_inputs inputs;
    inputs.messages = messages;
    inputs.tools = tools;
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.add_generation_prompt = true;
    inputs.enable_thinking = false;

    auto params =
      common_chat_templates_apply(weights_->get_templates(), inputs);

    // Tokenize the prompt
    std::vector<llama_token> prompt_tokens = tokenize(params.prompt);
    if (prompt_tokens.empty()) {
        throw ModelError("failed to tokenize prompt");
    }

    std::string response = generate_from_tokens(prompt_tokens, callback);

    common_chat_parser_params syntax;
    // Use explicitly configured format, or fall back to auto-detected format
    syntax.format = config_.chat_format.value_or(params.format);
    syntax.parse_tool_calls = true;
    // Load the template-specific PEG grammar; without this, PEG-based
    // formats silently fall back to parsing everything as plain content.
    syntax.parser.load(params.parser);

    common_chat_msg parsed_msg;
    try {
        parsed_msg = common_chat_parse(response, false, syntax);
    } catch (const std::exception& e) {
        if (parse_tool_call_xml_fallback(response, parsed_msg)) {
            return parsed_msg;
        }
        if (response.find("<tool_call") == std::string::npos) {
            // No tool call present at all - the template-derived grammar
            // rejected this as unparseable even though it's just plain
            // content (observed with granite-4.0-micro's broken native
            // grammar). Treat the raw response as the final answer rather
            // than failing on ordinary conversational replies.
            parsed_msg = {};
            parsed_msg.role = "assistant";
            parsed_msg.content = response;
            return parsed_msg;
        }
        throw ModelError(std::string("failed to parse model output: ") +
                          e.what());
    }

    // Some models' output is accepted by the template-derived parser as
    // plain content (no exception) even though it contains a real tool
    // call the parser failed to recognize. Recover it if so, without
    // discarding the original result if recovery fails.
    if (parsed_msg.tool_calls.empty() &&
        response.find("<tool_call") != std::string::npos) {
        common_chat_msg recovered;
        if (parse_tool_call_xml_fallback(response, recovered)) {
            parsed_msg = recovered;
        }
    }
    parsed_msg.role = "assistant";

    return parsed_msg;
}

std::string
Model::generate_from_tokens(const std::vector<llama_token>& all_tokens,
                            const ResponseCallback& callback)
{
    const llama_vocab* vocab = weights_->get_vocab();
    std::string response{};
    const int n_ctx = llama_n_ctx(ctx_);
    const int n_batch = llama_n_batch(ctx_);

    // Find common prefix length between processed tokens and new tokens
    size_t common_prefix = 0;
    while (common_prefix < processed_tokens_.size() &&
           common_prefix < all_tokens.size() &&
           processed_tokens_[common_prefix] == all_tokens[common_prefix]) {
        common_prefix++;
    }

    // If tokens diverged, clear KV cache from divergence point onwards
    if (common_prefix < processed_tokens_.size()) {
        llama_memory_t mem = llama_get_memory(ctx_);
        llama_memory_seq_rm(mem, 0, common_prefix, -1);
        processed_tokens_.resize(common_prefix);
        n_past_ = common_prefix;
    }

    size_t i = common_prefix;
    while (i < all_tokens.size()) {
        size_t batch_size = std::min(all_tokens.size() - i, (size_t)n_batch);

        if (n_past_ + (int)batch_size > n_ctx) {
            throw ModelError("context size exceeded");
        }

        std::vector<llama_token> batch_tokens(
          all_tokens.begin() + i, all_tokens.begin() + i + batch_size);

        llama_batch batch =
          llama_batch_get_one(batch_tokens.data(), batch_tokens.size());

        if (llama_decode(ctx_, batch) != 0) {
            throw ModelError("failed to decode batch");
        }

        n_past_ += batch_tokens.size();
        processed_tokens_.insert(
          processed_tokens_.end(), batch_tokens.begin(), batch_tokens.end());
        i += batch_size;
    }

    // Reset the grammar sampler before each turn so a finished grammar does
    // not force EOS on the next call.
    if (grammar_sampler_ != nullptr) {
        llama_sampler_reset(grammar_sampler_);
    }

    llama_token new_token_id{};
    while (true) {
        new_token_id = llama_sampler_sample(sampler_, ctx_, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[256];
        int n =
          llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            throw ModelError("failed to convert token to piece");
        }
        std::string piece(buf, n);

        if (callback) {
            callback(piece);
        }
        response += piece;

        if (n_past_ + 1 > n_ctx) {
            throw ModelError("context size exceeded during generation");
        }

        llama_batch batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(ctx_, batch) != 0) {
            throw ModelError("failed to decode token");
        }

        n_past_++;
        processed_tokens_.push_back(new_token_id);
    }

    return response;
}

bool
Model::save_cache(const std::string& cache_path)
{
    return llama_state_save_file(ctx_,
                                 cache_path.c_str(),
                                 processed_tokens_.data(),
                                 processed_tokens_.size());
}

std::vector<llama_token>
Model::load_cache(const std::string& cache_path)
{
    // Start with a reasonable capacity, will be resized based on actual count
    std::vector<llama_token> tokens(llama_n_ctx(ctx_));
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx_,
                               cache_path.c_str(),
                               tokens.data(),
                               tokens.capacity(),
                               &n_token_count_out)) {
        return {};
    }

    tokens.resize(n_token_count_out);
    set_cache_state(tokens);
    return tokens;
}

} // namespace agent_cpp
