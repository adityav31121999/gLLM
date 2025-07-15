// corpus.cpp
#include "include/tokenise.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <regex>
#include <algorithm>
#include <future>
#include <string_view>
#include <set>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <set>
#include <unordered_set> 

// Create a global mutex and progress map for debugging.
std::mutex progress_mutex;
std::map<std::string, int> progress_map;


/**
 * @brief The main entry point for learning a BPE vocabulary.
 * This function now implements the standard BPE training pipeline.
 * It bypasses the complex pre-tokenization stage and directly feeds the
 * word counts from the corpus into the BPE refinement algorithm.
 * @param corpus_word_counts A map of unique words and their frequencies from the entire corpus.
 * @param num_merges The number of merge operations to perform.
 * @param final_vocab Output vector to store the learned vocabulary tokens (for external use/saving).
 */
void tokeniser::learn_vocabulary_from_word_counts(const std::unordered_map<std::string, int>& corpus_word_counts, int num_merges,
    std::vector<std::string>& final_vocab)
{
    std::cout << "[INFO] Starting BPE training directly from raw corpus word counts." << std::endl;
    std::cout << "[INFO] Total unique words for training: " << corpus_word_counts.size() << std::endl;
    groupCommonTokens(corpus_word_counts, num_merges, final_vocab);
    this->tokens = final_vocab;
    this->vocSize = this->tokens.size();
}


/**
 * @brief Builds word counts with progress reporting tied to file completion.
 * This version uses the producer-consumer model and an event-driven logging
 * system. The main thread waits on a condition variable and prints a progress
 * update only when the producer thread signals that it has finished processing a file.
 * This function is now fully optimized for std::unordered_map for corpus_word_counts.
 * The final aggregation step is now parallelized using a merge tree.
 * @param file_paths A vector of paths to the text files.
 * @param corpus_word_counts Output map to be filled with unique tokens and their total counts.
 */
void tokeniser::buildCorpusWordCounts(const std::vector<std::string>& file_paths, 
    std::unordered_map<std::string, int>& corpus_word_counts)
{
    corpus_word_counts.clear();
    const size_t CHUNK_SIZE = 10000; // Number of lines per chunk
    ThreadSafeQueue<std::vector<std::string>> work_queue;

    // Determine number of producers and consumers
    int num_producers = (this->num_threads <= 4) ? 1 : 2; // Original logic for producers
    int num_consumers = this->num_threads - num_producers;
    // Ensure at least one consumer if possible, otherwise prioritize producer for small thread counts
    if (num_consumers < 1 && this->num_threads >= 1) { // If num_threads is 1, producer=1, consumer=0, adjust
        num_consumers = 1;
        num_producers = this->num_threads - num_consumers; // Adjust producer count if consumers need a thread
        if (num_producers < 1) num_producers = 1; // Ensure at least one producer too
    } else if (num_consumers < 1) { // Fallback if num_threads is weird, ensure minimum 1 producer, 1 consumer
        num_producers = 1;
        num_consumers = 1;
    }

    ProgressData progress;

    // 1. PRE-COMPUTATION
    for (const auto& path : file_paths) {
        if (std::filesystem::exists(path)) {
            progress.total_bytes += std::filesystem::file_size(path);
        }
    }

    // 2. DEFINE THE CONSUMER'S JOB (Optimized with std::string_view)
    auto consumer_task = [&work_queue]() -> std::unordered_map<std::string, int> {
        std::unordered_map<std::string, int> local_counts;
        std::vector<std::string> chunk;
        while (work_queue.wait_and_pop(chunk)) {
            for (const auto& line_str : chunk) {
                std::string_view line(line_str);
                for (size_t i = 0; i < line.length(); ) {
                    unsigned char current_char = line[i];
                    if (std::isalpha(current_char)) {
                        size_t start = i;
                        while (i < line.length() && std::isalpha(static_cast<unsigned char>(line[i]))) i++;
                        
                        std::string_view original_word_view = line.substr(start, i - start);
                        std::vector<std::string_view> sub_words_view = pre_split_word(original_word_view);
                        
                        for (auto& sub_word_view : sub_words_view) {
                            std::string lowercased_sub_word;
                            lowercased_sub_word.reserve(sub_word_view.length());
                            for (char c : sub_word_view) {
                                lowercased_sub_word += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                            }
                            local_counts[lowercased_sub_word]++;
                        }
                    } 
                    else if (!std::isspace(current_char)) {
                        local_counts[std::string(1, static_cast<char>(current_char))]++;
                        i++;
                    } 
                    else { i++; }
                }
            }
        }
        return local_counts;
    };

    // 3. LAUNCH THREADS
    std::cout << "-> Launching " << num_producers << " Producer(s) and " << num_consumers << " Consumer threads..." << std::endl;

    std::vector<std::future<std::unordered_map<std::string, int>>> consumer_futures;
    consumer_futures.reserve(num_consumers);
    for (int i = 0; i < num_consumers; ++i) {
        consumer_futures.push_back(std::async(std::launch::async, consumer_task));
    }

    // --- PRODUCERS ---
    std::vector<std::future<void>> producer_futures;
    producer_futures.reserve(num_producers);

    // Partition file_paths among producers
    size_t total_files = file_paths.size();
    
    // Calculate base files per producer, and any remainder for the last producer
    size_t files_per_producer_base = total_files / num_producers;
    size_t remainder_files = total_files % num_producers;
    size_t current_file_idx = 0;

    for (int p_idx = 0; p_idx < num_producers; ++p_idx) {
        size_t start_idx = current_file_idx;
        size_t num_files_for_this_producer = files_per_producer_base;
        if (p_idx < remainder_files) { // Distribute remainder files one by one to early producers
            num_files_for_this_producer++;
        }
        size_t end_idx = start_idx + num_files_for_this_producer;
        
        // Create a sub-vector of paths for this producer, moved into the lambda
        std::vector<std::string> producer_file_subset;
        producer_file_subset.reserve(num_files_for_this_producer);
        for (size_t i = start_idx; i < end_idx; ++i) {
            producer_file_subset.push_back(file_paths[i]);
        }
        current_file_idx = end_idx; // Update for next producer

        producer_futures.push_back(std::async(std::launch::async, [&, producer_file_subset = std::move(producer_file_subset)]() mutable {
            for (const auto& path : producer_file_subset) {
                std::string filename = std::filesystem::path(path).filename().string();
                std::ifstream file(path);
                if (!file.is_open()) {
                    std::cerr << "Warning: Producer thread could not open file: " << path << std::endl;
                    {
                        // Protect progress updates with mutex
                        std::unique_lock<std::mutex> lock(progress.mtx);
                        progress.files_completed_count++;
                        progress.last_file_completed = filename + " (Error)";
                        progress.cv.notify_one();
                    }
                    continue;
                }
                
                std::streampos last_pos = file.tellg();
                std::string line;
                std::vector<std::string> chunk_buffer;
                chunk_buffer.reserve(CHUNK_SIZE);

                while (std::getline(file, line)) {
                    chunk_buffer.push_back(std::move(line));
                    if (chunk_buffer.size() >= CHUNK_SIZE) {
                        std::streampos current_pos = file.tellg();
                        long long bytes_in_chunk = current_pos - last_pos;
                        last_pos = current_pos;

                        {
                            std::lock_guard<std::mutex> lock(progress.mtx);
                            progress.bytes_read += bytes_in_chunk;
                        }
                        
                        work_queue.push(std::move(chunk_buffer));
                        chunk_buffer.clear();
                        chunk_buffer.reserve(CHUNK_SIZE);
                    }
                }
                
                // After the loop, account for the final partial chunk and remaining bytes
                file.clear();
                file.seekg(0, std::ios::end);
                std::streampos end_pos = file.tellg();
                long long final_bytes = end_pos - last_pos;

                if (!chunk_buffer.empty()) {
                    work_queue.push(std::move(chunk_buffer));
                }

                // ATOMIC UPDATE AND SIGNAL for progress
                {
                    std::unique_lock<std::mutex> lock(progress.mtx);
                    progress.bytes_read += final_bytes;
                    progress.files_completed_count++;
                    progress.last_file_completed = filename; // This will be overwritten by concurrent producers.
                    progress.cv.notify_one();
                }
                file.close();
            }
        }));
    }

    // 4. MAIN THREAD EVENT-DRIVEN LOGGING LOOP
    {
        std::unique_lock<std::mutex> lock(progress.mtx);
        // size_t total_files = file_paths.size(); // Already calculated above
        size_t last_reported_count = 0;

        while (last_reported_count < total_files) {
            progress.cv.wait(lock, [&] {
                return progress.files_completed_count > last_reported_count;
            });

            double percentage = 0.0;
            if (progress.total_bytes > 0) {
                percentage = static_cast<double>(progress.bytes_read) / progress.total_bytes * 100.0;
            }

            std::cout << "  -> Progress: [" << std::fixed << std::setprecision(4) << percentage << "%] "
                      << "\t| Completed " << progress.files_completed_count << "/" << total_files << " files. "
                      << "\t(Finished '" << progress.last_file_completed << "')" << std::endl;

            last_reported_count = progress.files_completed_count;
        }
    }
    std::cout << "-> Producer(s) have finished reading all files. Waiting for consumers...\n";

    // Wait for all producer threads to complete their work.
    for (auto& pf : producer_futures) {
        pf.get(); // Blocks until this specific producer thread is done.
    }
    // Now that ALL producers are done, it's safe to close the work queue.
    work_queue.close();
    std::cout << "-> Work queue closed. Consumers will now finish processing remaining chunks and exit.\n";


    // 5. AGGREGATE FINAL RESULTS (Parallelized using a merge tree)
    std::cout << "-> Aggregating results using a parallel merge tree...\n";
    std::unordered_map<std::string, int> final_counts;

    if (!consumer_futures.empty()) {
        auto final_future = merge_maps(consumer_futures, 0, consumer_futures.size() - 1);
        final_counts = final_future.get();
    }

    corpus_word_counts.clear();
    corpus_word_counts.insert(final_counts.begin(), final_counts.end());
    std::cout << "-> Aggregation complete. Total unique tokens: " << corpus_word_counts.size() << std::endl;
}
