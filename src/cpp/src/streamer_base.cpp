// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

// class LambdaStreamer::LambdaStreamerImpl {
// public:
//     LambdaStreamerImpl(Tokenizer tokenizer, std::function<bool(std::string)> func): m_tokenizer(tokenizer), m_func(func) {}
//     LambdaStreamerImpl(std::function<bool(std::string)> func): m_func(func) {}
    
//     Tokenizer m_tokenizer;
//     std::function<bool(std::string)> m_func;
//     bool m_print_eos_token = false;
//     std::vector<int64_t> m_tokens_cache;
//     size_t print_len = 0;
    
//     bool put(int64_t token) {
//         std::stringstream res;
//         // do nothing if <eos> token is met and if print_eos_token=false
//         if (!m_print_eos_token && token == m_tokenizer.get_eos_token_id())
//             return m_func(res.str());

//         m_tokens_cache.push_back(token);
//         std::string text = m_tokenizer.decode(m_tokens_cache);
//         if (!text.empty() && '\n' == text.back()) {
//             // Flush the cache after the new line symbol
//             res << std::string_view{text.data() + print_len, text.size() - print_len};
//             m_tokens_cache.clear();
//             print_len = 0;
//             return m_func(res.str());
//         }
//         if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
//             // Don't print incomplete text
//             return m_func(res.str());
//         }
//         res << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
//         print_len = text.size();
//     return m_func(res.str());
// }

// bool end() {
//     std::stringstream res;
//     std::string text = m_tokenizer.decode(m_tokens_cache);
//     res << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
//     m_tokens_cache.clear();
//     print_len = 0;
//     return m_func(res.str());
// }

// };

// LambdaStreamer::LambdaStreamer(Tokenizer tokenizer, std::function<bool(std::string)> func) {}

// LambdaStreamer::LambdaStreamer(std::function<bool(std::string)> func) {
//     m_pimpl = std::make_shared<LambdaStreamer::LambdaStreamerImpl>(func);
// }

// void LambdaStreamer::put(int64_t token) { m_pimpl -> put(token);}

// void LambdaStreamer::end() { m_pimpl->end();}

}  // namespace genai
}  // namespace ov



// class LambdaStreamer: public StreamerBase {
// public:
//     // LambdaStreamer(Tokenizer tokenizer, std::function<bool(std::string)> func);
//     LambdaStreamer(std::function<bool(std::string)> func);

//     void put(int64_t token) override;
//     void end() override;
    
//     bool operator==(const LambdaStreamer& other) const {
//         // For simplicity, we assume lambdas are not comparable.
//         // If you need to compare actual logic, you may need to use type erasure or another method.
//         return false; // This can be changed based on your specific needs.
//     }
// private:
    
//     class LambdaStreamerImpl;
//     std::shared_ptr<LambdaStreamerImpl> m_pimpl;
// };