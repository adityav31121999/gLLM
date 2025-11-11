#ifdef USE_CPU
#include <vector>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <mutex>
#include "include/attention.hpp"

/**
 * @brief calculate key/query matrix using token embedding matrix and K/Q weight matrices
 * @details This function calculates the key (K) and query (Q) matrices by multiplying the input
 *          token embeddings or EV matrix with the corresponding K/Q weight matrices.
 * @param tokenOrEV The input token embedding matrix or EV matrix from the previous block (CONTEXT_WIN x EMBEDDING).
 * @param KQweights K/Q weight matrices (context window x embedding dimension)
 * @param KQtype 1 for K, 0 for Q
 */
void attention::getKeyQuery(const mat& tokenOrEV, const mat& KQweights, bool KQtype)
{
    // tokenOrEV: CONTEXT_WIN x EMBEDDING
    // KQweights: CONTEXT_WIN x EMBEDDING
    // K/Q: CONTEXT_WIN x CONTEXT_WIN
    // K/Q[i][j] = dot(i-th row of tokenOrEV, j-th row of KQweights)
    unsigned int num_of_threads = std::thread::hardware_concurrency();
    if (num_of_threads == 0) {
        num_of_threads = 1;
    }

    if (num_of_threads > 0 && num_of_threads <= 4) {
        for(int i = 0; i < CONTEXT_WIN; i++) {
            for(int j = 0; j < CONTEXT_WIN; j++) {
                float scalar = 0.0f;
                for(int k = 0; k < EMBEDDING; k++) {
                    scalar += tokenOrEV(i, k) * KQweights(j, k);
                }
                if (KQtype == 1) {
                    K(i, j) = scalar;
                }
                else {
                    Q(i, j) = scalar;
                }
            }
        }
    }
    else if (num_of_threads > 4) {
        std::vector<std::thread> threads;

        // Divide the work (rows of K/Q) among threads
        int rows_per_thread = CONTEXT_WIN / num_of_threads;
        int remaining_rows = CONTEXT_WIN % num_of_threads;

        for (unsigned int thread_idx = 0; thread_idx < num_of_threads; ++thread_idx) {
            int start_row = thread_idx * rows_per_thread;
            int end_row = start_row + rows_per_thread;
            if (thread_idx == num_of_threads - 1) {
                end_row += remaining_rows; // Last thread takes any remaining rows
            }

            // Launch a thread to process its assigned range of rows
            threads.emplace_back([this, &tokenOrEV, &KQweights, KQtype, start_row, end_row]() {
                for (int i = start_row; i < end_row; ++i) {
                    for (int j = 0; j < CONTEXT_WIN; ++j) {
                        float scalar = 0.0f;
                        for (int k = 0; k < EMBEDDING; ++k) {
                            scalar += tokenOrEV(i, k) * KQweights(j, k);
                        }
                        // Each thread writes to distinct (i, j) locations, so no explicit mutex is needed for K/Q access.
                        if (KQtype == 1) {
                            K(i, j) = scalar;
                        }
                        else {
                            Q(i, j) = scalar;
                        }
                    }
                }
            });
        }

        // Wait for all threads to complete
        for (auto& t : threads) {
            t.join();
        }
    }
    else {
        // This branch should ideally not be reached if num_of_threads is always >= 1
        // but kept for completeness based on original structure.
        throw std::runtime_error("Invalid thread count: " + std::to_string(num_of_threads));
    }
}

/**
 * @brief Calculates the dot product of the key and query matrices.
 * @details This function computes KdotQ = K * Q, where KdotQ[i][j] is the dot product of the i-th row of K and the j-th column of Q.
 *          It uses multithreading to parallelize the computation over the rows of the resulting matrix.
 */
void attention::getKdotQ()
{
    // K: CONTEXT_WIN x CONTEXT_WIN
    // Q: CONTEXT_WIN x CONTEXT_WIN
    // KdotQ: CONTEXT_WIN x CONTEXT_WIN
    // KdotQ[i][j] = dot(i-th row of K, j-th col of Q)
    int num_of_threads = std::thread::hardware_concurrency();
    if (num_of_threads == 0) {
        num_of_threads = 1;
    }

    if (num_of_threads > 0 && num_of_threads <= 4) {
        for(int i = 0; i < CONTEXT_WIN; i++) {
            for(int j = 0; j < CONTEXT_WIN; j++) {
                float scalar = 0.0f;
                for(int k = 0; k < CONTEXT_WIN; k++) {
                    scalar += K(i, k) * Q(k, j);
                }
                KdotQ(i, j) = scalar;
            }
        }
    }
    else if (num_of_threads > 4) {
        std::vector<std::thread> threads;

        // Divide the work (rows of KdotQ) among threads
        int rows_per_thread = CONTEXT_WIN / num_of_threads;
        int remaining_rows = CONTEXT_WIN % num_of_threads;

        for (unsigned int thread_idx = 0; thread_idx < num_of_threads; ++thread_idx) {
            int start_row = thread_idx * rows_per_thread;
            int end_row = start_row + rows_per_thread;
            if (thread_idx == num_of_threads - 1) {
                end_row += remaining_rows; // Last thread takes any remaining rows
            }

            // Launch a thread to process its assigned range of rows
            threads.emplace_back([this, start_row, end_row]() {
                for (int i = start_row; i < end_row; ++i) {
                    for (int j = 0; j < CONTEXT_WIN; ++j) {
                        // The calculation is K * Q, but the access pattern for Q is column-wise.
                        // The provided loop calculates K * Q.transpose(). Let's stick to the user-provided logic.
                        // KdotQ[i][j] = dot(i-th row of K, j-th row of Q) which is K * Q.transpose()
                        // The provided single-threaded code is KdotQ = K * Q. Let's follow that.
                        float scalar = 0.0f;
                        for (int k = 0; k < CONTEXT_WIN; k++) {
                            scalar += K(i, k) * Q(k, j);
                        }
                        KdotQ(i, j) = scalar;
                    }
                }
            });
        }

        // Wait for all threads to complete
        for (auto& t : threads) {
            t.join();
        }
    }
    else {
        throw std::runtime_error("Invalid thread count: " + std::to_string(num_of_threads));
    }
}

void attention::getKdotQ(const mat &tokens)
{
    /**
     * @brief Calculates KdotQ using tokens and qkCache.
     * @details This function computes KdotQ using the given tokens and the cached qkCache.
     * @param tokens The input tokens matrix (CONTEXT_WIN x EMBEDDING).
     * @note This function uses multithreading to accelerate the computation.
     */
    unsigned int num_of_threads = std::thread::hardware_concurrency();
    if (num_of_threads == 0) {
        num_of_threads = 1;
    }

    // This intermediate matrix is local to the function.
    std::vector<std::vector<float>> tokenQKed(CONTEXT_WIN, std::vector<float>(EMBEDDING, 0.0f));

    if (num_of_threads > 0 && num_of_threads <= 4) {
        // tokenQKed[i][j] = dot(ith row of tokens, jth row of qkCache)
        for (int i = 0; i < CONTEXT_WIN; i++) {
            for (int j = 0; j < EMBEDDING; j++) {
                float scalar = 0.0f;
                for (int k = 0; k < EMBEDDING; k++) {
                    scalar += tokens(i, k) * qkCache(j, k);
                }
                tokenQKed[i][j] = scalar;
            }
        }
        // KdotQ[i][j] = dot(ith row of tokenQKed, jth row of tokens)
        for (int i = 0; i < CONTEXT_WIN; i++) {
            for (int j = 0; j < CONTEXT_WIN; j++) {
                float scalar = 0.0f;
                for (int k = 0; k < EMBEDDING; k++) {
                    scalar += tokenQKed[i][k] * tokens(j, k);
                }
                KdotQ(i, j) = scalar;
            }
        }
    }
    else if (num_of_threads > 4) {
        std::vector<std::thread> threads;
        int rows_per_thread = CONTEXT_WIN / num_of_threads;
        int remaining_rows = CONTEXT_WIN % num_of_threads;

        // --- Part 1: Calculate tokenQKed in parallel ---
        for (unsigned int thread_idx = 0; thread_idx < num_of_threads; ++thread_idx) {
            int start_row = thread_idx * rows_per_thread;
            int end_row = start_row + rows_per_thread + (thread_idx == num_of_threads - 1 ? remaining_rows : 0);

            threads.emplace_back([this, &tokens, &tokenQKed, start_row, end_row]() {
                for (int i = start_row; i < end_row; ++i) {
                    for (int j = 0; j < EMBEDDING; ++j) {
                        float scalar = 0.0f;
                        for (int k = 0; k < EMBEDDING; ++k) {
                            scalar += tokens(i, k) * qkCache(j, k);
                        }
                        tokenQKed[i][j] = scalar; // Safe: each thread writes to distinct rows 'i'
                    }
                }
            });
        }
        for (auto& t : threads) { t.join(); }
        threads.clear();

        // --- Part 2: Calculate KdotQ in parallel ---
        for (unsigned int thread_idx = 0; thread_idx < num_of_threads; ++thread_idx) {
            int start_row = thread_idx * rows_per_thread;
            int end_row = start_row + rows_per_thread + (thread_idx == num_of_threads - 1 ? remaining_rows : 0);

            threads.emplace_back([this, &tokens, &tokenQKed, start_row, end_row]() {
                for (int i = start_row; i < end_row; ++i) {
                    for (int j = 0; j < CONTEXT_WIN; ++j) {
                        float scalar = 0.0f;
                        for (int k = 0; k < EMBEDDING; ++k) {
                            scalar += tokenQKed[i][k] * tokens(j, k);
                        }
                        KdotQ(i, j) = scalar; // Safe: each thread writes to distinct rows 'i'
                    }
                }
            });
        }
        for (auto& t : threads) { t.join(); }
    }
    else {
        throw std::runtime_error("Invalid thread count: " + std::to_string(num_of_threads));
    }
}

void attention::getKdotQ(const mat &tokens, const mat &EVfromPrevBlock)
{
    /**
     * @brief Calculates KdotQ using tokens and EV from the previous block.
     * @details This function computes KdotQ using the given tokens and the EV matrix from the previous block.
     * @param tokens The input tokens matrix (CONTEXT_WIN x EMBEDDING).
     * @param EVfromPrevBlock The EV matrix from the previous block (CONTEXT_WIN x EMBEDDING).
     */
    unsigned int num_of_threads = std::thread::hardware_concurrency();
    if (num_of_threads == 0) {
        num_of_threads = 1;
    }

    // This intermediate matrix is local to the function.
    std::vector<std::vector<float>> tokenQKed(CONTEXT_WIN, std::vector<float>(EMBEDDING, 0.0f));

    if (num_of_threads > 0 && num_of_threads <= 4) {
        // tokenQKed[i][j] = dot(ith row of tokens, jth row of qkCache)
        for (int i = 0; i < CONTEXT_WIN; i++) {
            for (int j = 0; j < EMBEDDING; j++) {
                float scalar = 0.0f;
                for (int k = 0; k < EMBEDDING; k++) {
                    scalar += tokens(i, k) * qkCache(j, k);
                }
                tokenQKed[i][j] = scalar;
            }
        }
        // KdotQ[i][j] = dot(ith row of tokenQKed, jth row of tokens)
        for (int i = 0; i < CONTEXT_WIN; i++) {
            for (int j = 0; j < CONTEXT_WIN; j++) {
                float scalar = 0.0f;
                for (int k = 0; k < EMBEDDING; k++) {
                    scalar += tokenQKed[i][k] * EVfromPrevBlock(j, k);
                }
                KdotQ(i, j) = scalar;
            }
        }
    }
    else if (num_of_threads > 4) {
        std::vector<std::thread> threads;
        int rows_per_thread = CONTEXT_WIN / num_of_threads;
        int remaining_rows = CONTEXT_WIN % num_of_threads;

        // --- Part 1: Calculate tokenQKed in parallel ---
        for (unsigned int thread_idx = 0; thread_idx < num_of_threads; ++thread_idx) {
            int start_row = thread_idx * rows_per_thread;
            int end_row = start_row + rows_per_thread + (thread_idx == num_of_threads - 1 ? remaining_rows : 0);

            threads.emplace_back([this, &tokens, &tokenQKed, start_row, end_row]() {
                for (int i = start_row; i < end_row; ++i) {
                    for (int j = 0; j < EMBEDDING; ++j) {
                        float scalar = 0.0f;
                        for (int k = 0; k < EMBEDDING; ++k) {
                            scalar += tokens(i, k) * qkCache(j, k);
                        }
                        tokenQKed[i][j] = scalar; // Safe: each thread writes to distinct rows 'i'
                    }
                }
            });
        }
        for (auto& t : threads) { t.join(); }
        threads.clear();

        // --- Part 2: Calculate KdotQ in parallel ---
        for (unsigned int thread_idx = 0; thread_idx < num_of_threads; ++thread_idx) {
            int start_row = thread_idx * rows_per_thread;
            int end_row = start_row + rows_per_thread + (thread_idx == num_of_threads - 1 ? remaining_rows : 0);

            threads.emplace_back([this, &tokens, &tokenQKed, &EVfromPrevBlock, start_row, end_row]() {
                for (int i = start_row; i < end_row; ++i) {
                    for (int j = 0; j < CONTEXT_WIN; ++j) {
                        float scalar = 0.0f;
                        for (int k = 0; k < EMBEDDING; ++k) {
                            scalar += tokenQKed[i][k] * EVfromPrevBlock(j, k);
                        }
                        KdotQ(i, j) = scalar; // Safe: each thread writes to distinct rows 'i'
                    }
                }
            });
        }
        for (auto& t : threads) { t.join(); }
    }
    else {
        throw std::runtime_error("Invalid thread count: " + std::to_string(num_of_threads));
    }
}

#endif