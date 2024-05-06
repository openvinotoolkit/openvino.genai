#include "text_streamer.hpp"

TextCallbackStreamer::TextCallbackStreamer(const Tokenizer& tokenizer) {
    this->tokenizer = tokenizer;
}

TextCallbackStreamer::TextCallbackStreamer(const Tokenizer& tokenizer, std::function<void (std::string)> callback) {
    this->tokenizer = tokenizer;
    this->m_callback = callback;
}

void TextCallbackStreamer::put(int64_t token) {
    // do not print anything and flush cache if EOS token is met
    if (token == tokenizer.m_eos_token) {
        end();
        return ;
    }

    token_cache.push_back(token);
    std::string text = tokenizer.detokenize(token_cache);
    if (!text.empty() && '\n' == text.back()) {
        // Flush the cache after the new line symbol
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len};
        token_cache.clear();
        print_len = 0;
        return;
    }
    if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
        // Don't print incomplete text
        return;
    }
    std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
    print_len = text.size();
}

void TextCallbackStreamer::end() {
    std::string text = tokenizer.detokenize(token_cache);
    std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
    token_cache.clear();
    print_len = 0;
}

void TextCallbackStreamer::set_tokenizer(Tokenizer tokenizer) {
    this->tokenizer = tokenizer;
}
