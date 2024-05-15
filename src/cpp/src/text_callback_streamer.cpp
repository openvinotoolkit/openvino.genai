#include "text_callback_streamer.hpp"

namespace ov {

TextCallbackStreamer::TextCallbackStreamer(const Tokenizer& tokenizer, std::function<void (std::string)> callback, bool print_eos_token) {
    m_tokenizer = tokenizer;
    m_print_eos_token = print_eos_token;
    on_decoded_text_callback = callback;
    m_enabled = true;
}

TextCallbackStreamer::TextCallbackStreamer(const Tokenizer& tokenizer, bool print_eos_token) {
    m_tokenizer = tokenizer;
    m_print_eos_token = print_eos_token;
}

void TextCallbackStreamer::put(int64_t token) {
    std::stringstream res;
    // do nothing if <eos> token is met and if print_eos_token=false
    if (!m_print_eos_token && token == m_tokenizer.get_eos_token_id())
        return;

    m_tokens_cache.push_back(token);
    std::string text = m_tokenizer.decode(m_tokens_cache);
    if (!text.empty() && '\n' == text.back()) {
        // Flush the cache after the new line symbol
        res << std::string_view{text.data() + print_len, text.size() - print_len};
        m_tokens_cache.clear();
        print_len = 0;
        on_finalized_text(res.str());
        return;
    }
    if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
        // Don't print incomplete text
        on_finalized_text(res.str());
        return;
    }
    res << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
    print_len = text.size();
    on_finalized_text(res.str());
    return;
}

void TextCallbackStreamer::end() {
    std::stringstream res;
    std::string text = m_tokenizer.decode(m_tokens_cache);
    res << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
    m_tokens_cache.clear();
    print_len = 0;
    on_finalized_text(res.str());
}

void TextCallbackStreamer::set_tokenizer(Tokenizer tokenizer) {
    this->m_tokenizer = tokenizer;
}

void TextCallbackStreamer::set_callback(std::function<void (std::string)> callback) {
    on_decoded_text_callback = callback;
    m_enabled = true;
}

void TextCallbackStreamer::set_callback() {
    on_decoded_text_callback = [](std::string words){};
    m_enabled = false;
}

void TextCallbackStreamer::on_finalized_text(const std::string& subword) {
    if (m_enabled) {
        on_decoded_text_callback(subword);
    }
}

} // namespace ov
