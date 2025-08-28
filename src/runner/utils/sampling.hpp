// sampling.h
#ifndef SAMPLING_H
#define SAMPLING_H

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <iostream> // For debugging, can be removed

namespace sampling {

// Helper function to apply softmax to a vector of scores
// Modifies the input vector in-place to hold probabilities
void softmax(std::vector<float>& scores) {
    if (scores.empty()) return;

    // Find the maximum score for numerical stability
    float max_score = *std::max_element(scores.begin(), scores.end());

    // Compute exp(score - max_score) and sum
    float sum = 0.0;
    for (float& s : scores) {
        s = std::exp(s - max_score);
        sum += s;
    }

    // Normalize by the sum
    if (sum > 0.0) { // Avoid division by zero
        for (float& s : scores) {
            s /= sum;
        }
    }
}

// Helper function for multinomial sampling based on probabilities
// Returns the sampled index
int multinomial_sample(const std::vector<float>& probabilities, std::mt19937& gen) {
    if (probabilities.empty()) {
        // Handle error or return a default value
        // Returning -1 to indicate error
        return -1;
    }

    std::uniform_real_distribution<float> dis(0.0, 1.0);
    float rand_val = dis(gen);
    float cumulative_prob = 0.0;

    for (size_t i = 0; i < probabilities.size(); ++i) {
        cumulative_prob += probabilities[i];
        if (rand_val <= cumulative_prob) {
            return static_cast<int>(i);
        }
    }
    // In case of rounding errors, return the last index
    return static_cast<int>(probabilities.size() - 1);
}


// Nucleus (Top-p) + Top-k Sampling
// weighted_scores: vector of logits/scores for each token
// top_p: cumulative probability threshold (0.0 to 1.0)
// top_k: maximum number of top tokens to consider
// gen: random number generator reference
// Returns the sampled token ID (index)
int nucleus_sampling(std::vector<float> weighted_scores, float top_p = 0.8, int top_k = 25, std::mt19937& gen) {
    if (weighted_scores.empty()) {
        return -1; // Or handle error appropriately
    }

    size_t vocab_size = weighted_scores.size();
    // Apply softmax to get probabilities
    softmax(weighted_scores); // weighted_scores now holds probabilities

    // Create vector of (probability, index) pairs
    std::vector<std::pair<float, int>> prob_index_pairs(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        prob_index_pairs[i] = {weighted_scores[i], static_cast<int>(i)};
    }

    // Sort by probability descending
    std::sort(prob_index_pairs.begin(), prob_index_pairs.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first; // Descending order
              });

    // Select top-p and top-k candidates
    std::vector<float> selected_probs;
    std::vector<int> selected_indices;
    float cum_prob = 0.0;

    size_t limit = std::min(static_cast<size_t>(top_k), vocab_size);

    for (size_t i = 0; i < limit; ++i) {
        float prob = prob_index_pairs[i].first;
        int idx = prob_index_pairs[i].second;

        if (cum_prob < top_p) { // Check top-p condition
            cum_prob += prob;
            selected_probs.push_back(prob);
            selected_indices.push_back(idx);
        } else {
            break; // Stop if cumulative probability is reached
        }
    }

    // Handle case where no tokens were selected (e.g., all probs are 0 or top_p=0)
    if (selected_probs.empty()) {
         // Fallback: use the single highest probability token
         // Or could fall back to random sampling on the full distribution
         // Here, we'll pick the top token
         return prob_index_pairs.empty() ? -1 : prob_index_pairs[0].second;
    }

    // Renormalize selected probabilities
    float sum_selected = std::accumulate(selected_probs.begin(), selected_probs.end(), 0.0);
    if (sum_selected > 0.0) {
        for (float& p : selected_probs) {
            p /= sum_selected;
        }
    } else {
        // If sum is zero, assign uniform probability
        for (float& p : selected_probs) {
             p = 1.0 / selected_probs.size();
        }
    }


    // Sample from the selected subset
    int sub_index = multinomial_sample(selected_probs, gen);
    if (sub_index >= 0 && static_cast<size_t>(sub_index) < selected_indices.size()) {
        return selected_indices[sub_index];
    }
    // Fallback
    return selected_indices.empty() ? -1 : selected_indices[0];
}


// Random Sampling
// weighted_scores: vector of logits/scores for each token
// gen: random number generator reference
// Returns the sampled token ID (index)
int random_sampling(std::vector<float> weighted_scores, std::mt19937& gen) {
    if (weighted_scores.empty()) {
        return -1; // Or handle error appropriately
    }
    // Apply softmax to get probabilities
    softmax(weighted_scores); // weighted_scores now holds probabilities

    // Sample directly from the full distribution
    return multinomial_sample(weighted_scores, gen);
}

// Repetition-Aware Sampling (RAS)
// weighted_scores: vector of logits/scores for each token
// decoded_tokens: vector of previously sampled token IDs
// top_p, top_k, win_size, tau_r: RAS parameters
// gen: random number generator reference
// Returns the sampled token ID (index)
int ras_sampling(std::vector<float> weighted_scores,
                 const std::vector<int>& decoded_tokens,
                 float top_p = 0.8, int top_k = 25,
                 int win_size = 10, float tau_r = 0.1,
                 std::mt19937& gen = std::mt19937{std::random_device{}()}) { // Default gen for convenience

    // Step 1: Get candidate from nucleus sampling
    int top_ids = nucleus_sampling(weighted_scores, top_p, top_k, gen);

    if (top_ids < 0 || decoded_tokens.empty() || win_size <= 0 || tau_r <= 0.0) {
        // If nucleus failed or no history or invalid params, just return nucleus result or handle error
        return top_ids;
    }

    // Step 2: Check for repetition in the recent window
    int rep_num = 0;
    size_t start_check_idx = (decoded_tokens.size() > static_cast<size_t>(win_size)) ?
                             (decoded_tokens.size() - win_size) : 0;
    size_t end_check_idx = decoded_tokens.size();

    for (size_t i = start_check_idx; i < end_check_idx; ++i) {
        if (decoded_tokens[i] == top_ids) {
            rep_num++;
        }
    }

    float repetition_ratio = static_cast<float>(rep_num) / static_cast<float>(win_size);

    // Step 3: If repetition is too high, fallback to random sampling
    if (repetition_ratio >= tau_r) {
        // std::cout << "RAS triggered for token " << top_ids << " (rep ratio: " << repetition_ratio << ")\n"; // Debug
        return random_sampling(weighted_scores, gen);
    }

    // Otherwise, return the nucleus sampling result
    return top_ids;
}

} // namespace sampling

#endif // SAMPLING_H