#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

struct TokenIdScore {
    int id;
    float score;

    TokenIdScore() = default;
    TokenIdScore(int id, float score) : id(id), score(score) {}

    bool operator<(const TokenIdScore& other) const { return score < other.score; }
    bool operator>(const TokenIdScore& other) const { return score > other.score; }

    friend std::ostream& operator<<(std::ostream& os, const TokenIdScore& self) {
        return os << "TokenIdScore(id=" << self.id << ", score=" << self.score << ")";
    }
};

void sampling_softmax_inplace(TokenIdScore* first, TokenIdScore* last) {
    float max_score = std::max_element(first, last)->score;
    float sum = 0.f;
    for (TokenIdScore* p = first; p != last; p++) {
        float s = std::exp(p->score - max_score);
        p->score = s;
        sum += s;
    }
    float inv_sum = 1.f / sum;
    for (TokenIdScore* p = first; p != last; p++) {
        p->score *= inv_sum;
    }
}

void sampling_top_k(TokenIdScore* first, TokenIdScore* kth, TokenIdScore* last) {
    std::nth_element(first, kth, last, std::greater<TokenIdScore>());
}

TokenIdScore* sampling_top_p(TokenIdScore* first, TokenIdScore* last, float top_p) {
    // fast top_p in expected O(n) time complexity
    sampling_softmax_inplace(first, last);

    while (first + 1 < last) {
        const float pivot_score = (last - 1)->score; // use mid score?
        TokenIdScore* mid =
            std::partition(first, last - 1, [pivot_score](const TokenIdScore& x) { return x.score > pivot_score; });
        std::swap(*mid, *(last - 1));

        const float prefix_sum =
            std::accumulate(first, mid, 0.f, [](float sum, const TokenIdScore& x) { return sum + x.score; });
        if (prefix_sum >= top_p) {
            last = mid;
        }
        else if (prefix_sum + mid->score < top_p) {
            first = mid + 1;
            top_p -= prefix_sum + mid->score;
        }
        else {
            return mid + 1;
        }
    }
    return last;
}

void sampling_repetition_penalty(float* first, float* last, const std::vector<int>& input_ids,
    float penalty) {
    if (penalty < 0) {
        std::cout << "penalty must be a positive float, but got " << penalty;
        return;
    }
    const float inv_penalty = 1.f / penalty;
    const int vocab_size = last - first;
    std::vector<bool> occurrence(vocab_size, false);
    for (const int id : input_ids) {
        if (!occurrence[id]) {
            first[id] *= (first[id] > 0) ? inv_penalty : penalty;
        }
        occurrence[id] = true;
    }
}

void sampling_temperature(float* first, float* last, float temp) {
    const float inv_temp = 1.f / temp;
    for (float* it = first; it != last; it++) {
        *it *= inv_temp;
    }
}



