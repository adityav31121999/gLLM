
#include "include/basic.hpp"
#include <thread>
#include <chrono>
#include <mutex>

/**
 * @brief Computes the product of numbers in a specified range.
 * @param[in] start The starting number of the range.
 * @param[in] end The ending number of the range.
 * @return The product of all integers from start to end, inclusive.
 */
long long int partial_factorial(int start, int end) {
    long long int result = 1; // Initialize result to 1
    for (int i = start; i <= end; ++i) {
        result *= i; // Multiply result by each number in the range
    }
    return result; // Return the computed product
}

/**
 * @brief Calculates the factorial of a number using multithreading
 * @param[in] n an integer
 * @return the factorial of n
 * @throws std::runtime_error if the factorial of n is larger than a long long int
 */
long long int factorial(int n) {
    // factorial of a number by multithreading
    if (n == 0 || n == 1)
        return 1;

    // Find the number of threads on the system and compare it with the input
    int num_threads = std::thread::hardware_concurrency();
    num_threads = std::max(2, std::min(num_threads, n)); // Ensure at least 2 threads for multithreading

    // Create a vector to store the partial factorial results of each thread
    std::vector<long long int> results(num_threads, 1);
    int chunk_size = n / num_threads; // Calculate the chunk size
    int remainder = n % num_threads;  // Calculate the remainder

    // Lambda function to compute the partial factorial of each chunk
    auto compute_chunk = [&](int thread_id) {
        int start = thread_id * chunk_size + 1;
        int end = (thread_id + 1) * chunk_size;
        if (thread_id == num_threads - 1) { 
            // Last thread may take the remainder
            end += remainder;
        }
        results[thread_id] = partial_factorial(start, end);
    };

    // Launch threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.push_back(std::thread(compute_chunk, i));
    }

    // Join all threads
    for (auto& t : threads) {
        t.join();
    }

    // Combine results from all threads using accumulate with multiplication
    return std::accumulate(results.begin(), results.end(), 1LL, std::multiplies<long long int>());
}

/**
 * @brief Calculate the number of permutations of n items taken r at a time.
 * @param n The total number of items.
 * @param r The number of items to select.
 * @return The number of permutations of n items taken r at a time.
 */
long long int nPr(int n, int r) {
    // Calculate nPr using the formula: nPr = n! / (n-r)!
    return factorial(n) / factorial(n-r);
}

/**
 * @brief Calculate the number of combinations of n items taken r at a time.
 * @param n The total number of items.
 * @param r The number of items to select.
 * @return The number of combinations of n items taken r at a time.
 */
long long int nCr(int n, int r) {
    // nCr = n! / (r! * (n-r)!)
    return factorial(n) / (factorial(r) * factorial(n-r));
}
