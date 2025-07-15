#include "include/tokenise.hpp"
#include <vector>
#include <future>


// Helper to merge two unordered_maps. This is efficient as it moves map2's contents.
std::unordered_map<std::string, int> mergeTwoMaps(std::unordered_map<std::string, int> map1,
    std::unordered_map<std::string, int> map2) 
{
    // Prefer inserting smaller map into larger one for efficiency
    if (map1.size() < map2.size()) {
        std::swap(map1, map2);
    }
    for (auto& pair : map2) { // Use auto& to allow moving value out
        map1[pair.first] += pair.second;
    }
    return map1;
}

// Recursive function for parallel map merging (merge tree)
std::future<std::unordered_map<std::string, int>> merge_maps(
    std::vector<std::future<std::unordered_map<std::string, int>>>& futures,
    size_t start_idx, size_t end_idx) {

    if (start_idx == end_idx) {
        // Base case: a single map, just return its future
        return std::move(futures[start_idx]);
    }
    if (start_idx + 1 == end_idx) {
        // Base case: two maps, merge them directly
        return std::async(std::launch::async, [
            m1 = futures[start_idx].get(), // Get and move map from first future
            m2 = futures[end_idx].get()    // Get and move map from second future
        ]() mutable {
            return mergeTwoMaps(std::move(m1), std::move(m2));
        });
    }

    size_t mid_idx = start_idx + (end_idx - start_idx) / 2;

    // Recursively merge left and right halves
    auto left_future = merge_maps(futures, start_idx, mid_idx);
    auto right_future = merge_maps(futures, mid_idx + 1, end_idx);

    // Merge the results of the two parallel merges
    return std::async(std::launch::async, [
        m1 = left_future.get(), // Get and move map from left sub-tree
        m2 = right_future.get() // Get and move map from right sub-tree
    ]() mutable {
        return mergeTwoMaps(std::move(m1), std::move(m2));
    });
}


// Dummy implementation for parallel_merge_pair
// This is critical for performance and needs an efficient BPE update logic.
// A naive approach: re-split words containing the best_pair.
void merge_pair(const std::pair<std::string, std::string>& best_pair,
                         std::unordered_map<std::string, std::vector<std::string>>& splits,
                         int num_threads)
{
    // Step 1: Identify words to update.
    // Instead of collecting strings, collect iterators or keys.
    // For efficiency, a real BPE implementation often uses an inverted index
    // (token -> list of words containing token) to quickly find words.
    // Since we don't have that, we iterate, but we can optimize the update.
    
    // We'll collect the keys of the words that need updating.
    std::vector<std::string> keys_to_update;
    keys_to_update.reserve(splits.size()); // Reserve to avoid reallocations

    for (const auto& entry : splits) {
        // Only add words that actually contain the pair.
        // This check can be optimized: search for first token, then check next.
        for (size_t i = 0; i + 1 < entry.second.size(); ++i) {
            if (entry.second[i] == best_pair.first && entry.second[i+1] == best_pair.second) {
                keys_to_update.push_back(entry.first);
                break; // Found the pair, no need to check further in this word's split
            }
        }
    }

    // If no words contain the pair, nothing to do
    if (keys_to_update.empty()) {
        return;
    }

    // Step 2: Parallelize the update, returning local changes.
    std::vector<std::future<std::unordered_map<std::string, std::vector<std::string>>>> futures;
    futures.reserve(num_threads);

    // Calculate effective number of threads if there are fewer words than threads
    int effective_num_threads = std::min<int>(num_threads, (int)keys_to_update.size());
    if (effective_num_threads == 0) return; // Should not happen if keys_to_update is not empty

    size_t chunk_size = keys_to_update.size() / effective_num_threads;
    
    for (int t = 0; t < effective_num_threads; ++t) {
        size_t start_idx = chunk_size * t;
        size_t end_idx = (t == effective_num_threads - 1) ? keys_to_update.size() : start_idx + chunk_size;

        futures.push_back(std::async(std::launch::async, [&, start_idx, end_idx]() {
            // Each thread works on a *copy* of the relevant parts of `splits`
            // and builds its own local updates.
            std::unordered_map<std::string, std::vector<std::string>> local_updated_splits;

            for (size_t i = start_idx; i < end_idx; ++i) {
                const std::string& word_key = keys_to_update[i];
                // Get the current subwords for this word. This read *must* be safe.
                // Assuming `splits` is thread-safe for reads while threads are spawned.
                // If `splits` can change during this phase, it needs read-write lock or similar.
                // For BPE, `splits` is generally stable until the next merge.
                
                // OPTIMIZATION: Get existing subwords without locking global `splits` map.
                // This implies `splits` itself is read-only at this stage, or protected by R/W lock.
                // For simplicity, we'll assume `splits` is thread-safe for concurrent reads (or copied).
                // A better design would be to pass a const reference to splits into the lambda
                // or pre-copy the relevant parts for each thread.
                // Given the original code's locking pattern, we'll fetch the *original* split.
                const std::vector<std::string>& current_subwords = splits.at(word_key); // Use .at() for bounds checking

                std::vector<std::string> new_subwords;
                new_subwords.reserve(current_subwords.size()); // Optimize allocation

                for (size_t j = 0; j < current_subwords.size(); ) {
                    if (j + 1 < current_subwords.size() &&
                        current_subwords[j] == best_pair.first &&
                        current_subwords[j+1] == best_pair.second) {
                        new_subwords.push_back(best_pair.first + best_pair.second);
                        j += 2; // Skip the two merged tokens
                    } else {
                        new_subwords.push_back(current_subwords[j]);
                        j += 1;
                    }
                }
                // Store the updated subwords in this thread's local map
                local_updated_splits[word_key] = std::move(new_subwords);
            }
            return local_updated_splits;
        }));
    }

    // Step 3: Aggregate results from all threads.
    // This phase should be sequential or use a merge tree (if many threads)
    // to apply the collected local changes to the global `splits` map.
    // Since `splits` modification is an `std::unordered_map` assignment,
    // it's best done sequentially from the main thread.
    for (auto& f : futures) {
        std::unordered_map<std::string, std::vector<std::string>> local_map = f.get(); // Get the thread's results
        for (auto& entry : local_map) {
            // Update the global `splits` map. This is safe as it's sequential.
            splits[entry.first] = std::move(entry.second);
        }
    }
}
