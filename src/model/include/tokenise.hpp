#ifndef TOKENISE_HPP
#define TOKENISE_HPP 1

#include <maths.hpp>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <cmath>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <regex>
#include <algorithm>
#include <future>
#include <thread>
#include <memory>
#include <utility>

// for fast multithreading operations
struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const {
        std::size_t h1 = std::hash<std::string>{}(p.first);
        std::size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};


/**
 * @brief A struct to hold shared progress data for logging.
 * Protected by a mutex.
 * Explicitly delete copy/move operations because of std::mutex.
 */
struct ProgressData {
    std::mutex mtx; // This makes ProgressData non-copyable and non-movable
    std::condition_variable cv;
    unsigned long long total_bytes = 0;
    unsigned long long bytes_read = 0;
    size_t files_completed_count = 0;
    std::string last_file_completed;
    int merges_completed = 0;
    int total_merges = 0;
    std::chrono::steady_clock::time_point start_time;

    // Explicitly delete copy constructor and assignment operator
    ProgressData(const ProgressData&) = delete;
    ProgressData& operator=(const ProgressData&) = delete;

    // Explicitly delete move constructor and assignment operator (due to mutex)
    ProgressData(ProgressData&&) = delete;
    ProgressData& operator=(ProgressData&&) = delete;

    // Default constructor is fine for creating a new instance
    ProgressData() = default;
};


/**
 * Class to tokenise dataset into subwords and embeddings. The embeddings are of
 * d dimension with all the values of type float.
 */
class tokeniser {
private:
    int d;          // embedding dimension
    int vocSize;    // vocabulary size (number of merges performed while BPE algorithm)
    int d_val;      // divisor (makes it look like repeatation)

    std::vector<std::string> tokens;                // all possible tokens
    std::vector<std::string> vocab_tokens;          // lexicographically sorted tokens for lookups
    std::vector<float> seeds;                       // seeds for all tokens (seeds.csv)
    std::vector<std::vector<float>> embeddings;     // vector for each token of dimension d
    std::vector<std::vector<float>> deEmbeddings;   // inverse of each token of dimension d
    std::unordered_map<std::string, int> token_to_idx;      // maps token string to index in embeddings
    std::unordered_map<std::string, int> corpusWordCount;   // NEW (or similar if it's not a member)
    std::unordered_map<std::string, int> statOfTokens;      // hold tokens and their stats (unique_tokens.csv)

public:

    std::string path2data;                          // path to dataset
    std::string path2token;                         // path to tokens and embeddings
    int num_threads;                        // number of threads
    int totalCorpusWordCount;               // corpus word count
    std::unique_ptr<ProgressData> bpe_progress;

#ifdef USE_OPENCL
    OpenCLContext ocl;
#endif

// constructors
    // default constructors
#ifdef USE_OPENCL
    tokeniser(int d, OpenCLContext& context); // You'd need to initialize bpe_progress here too
    tokeniser(int d, int d_val, OpenCLContext& context); // And here
    explicit tokeniser(const std::string& path2data, OpenCLContext& context) noexcept; // And here
#elif USE_CUDA || USE_CPU
    // Initialize bpe_progress in all constructors
    tokeniser() : bpe_progress(std::make_unique<ProgressData>()) {}
    tokeniser(int d);
    tokeniser(int d, int d_val);
    explicit tokeniser(const std::string& path2data) noexcept;
#endif

    // Copy Constructor
    tokeniser(const tokeniser& other)
        : d(other.d),
          vocSize(other.vocSize),
          d_val(other.d_val),
          path2data(other.path2data),
          tokens(other.tokens),
          vocab_tokens(other.vocab_tokens),
          seeds(other.seeds),
          embeddings(other.embeddings),
          deEmbeddings(other.deEmbeddings),
          token_to_idx(other.token_to_idx),
          corpusWordCount(other.corpusWordCount),
          statOfTokens(other.statOfTokens),
          num_threads(other.num_threads),
          totalCorpusWordCount(other.totalCorpusWordCount),
          bpe_progress(std::make_unique<ProgressData>()) // Create a NEW, independent ProgressData object
#ifdef USE_OPENCL
          , ocl(other.ocl)
#endif
    {}

    // Move Constructor
    tokeniser(tokeniser&& other) noexcept
        : d(other.d),
          vocSize(other.vocSize),
          d_val(other.d_val),
          path2data(std::move(other.path2data)),
          tokens(std::move(other.tokens)),
          vocab_tokens(std::move(other.vocab_tokens)),
          seeds(std::move(other.seeds)),
          embeddings(std::move(other.embeddings)),
          deEmbeddings(std::move(other.deEmbeddings)),
          token_to_idx(std::move(other.token_to_idx)),
          corpusWordCount(std::move(other.corpusWordCount)),
          statOfTokens(std::move(other.statOfTokens)),
          num_threads(other.num_threads),
          totalCorpusWordCount(other.totalCorpusWordCount),
          bpe_progress(std::move(other.bpe_progress)) // std::unique_ptr handles the move
#ifdef USE_OPENCL
          , ocl(std::move(other.ocl))
#endif
    {}

    // Copy Assignment Operator
    tokeniser& operator=(const tokeniser& other) {
        if (this == &other) { // Handle self-assignment
            return *this;
        }

        d = other.d;
        vocSize = other.vocSize;
        d_val = other.d_val;
        path2data = other.path2data;
        tokens = other.tokens;
        vocab_tokens = other.vocab_tokens;
        seeds = other.seeds;
        embeddings = other.embeddings;
        deEmbeddings = other.deEmbeddings;
        token_to_idx = other.token_to_idx;
        corpusWordCount = other.corpusWordCount;
        statOfTokens = other.statOfTokens;
        num_threads = other.num_threads;
        totalCorpusWordCount = other.totalCorpusWordCount;
        bpe_progress = std::make_unique<ProgressData>(); // Create a new ProgressData object

        #ifdef USE_OPENCL
            ocl = other.ocl;
        #endif
        return *this;
    }

    // Move Assignment Operator
    tokeniser& operator=(tokeniser&& other) noexcept {
        if (this == &other) { // Handle self-assignment
            return *this;
        }

        d = other.d;
        vocSize = other.vocSize;
        d_val = other.d_val;
        path2data = std::move(other.path2data);
        tokens = std::move(other.tokens);
        vocab_tokens = std::move(other.vocab_tokens);
        seeds = std::move(other.seeds);
        embeddings = std::move(other.embeddings);
        deEmbeddings = std::move(other.deEmbeddings);
        token_to_idx = std::move(other.token_to_idx);
        corpusWordCount = std::move(other.corpusWordCount);
        statOfTokens = std::move(other.statOfTokens);
        num_threads = other.num_threads;
        totalCorpusWordCount = other.totalCorpusWordCount;

        // Move the unique_ptr
        bpe_progress = std::move(other.bpe_progress);

        #ifdef USE_OPENCL
            ocl = std::move(other.ocl);
        #endif

        other.d = 0;
        other.vocSize = 0;
        other.d_val = 0;
        other.num_threads = 0;
        other.totalCorpusWordCount = 0;
        return *this;
    }

    // programs for setting values
    void setEmbeddingDimension(int d);
    void setDval(int d_val);
    void setVocabularySize(int vocSize);
    void setNumThreads();
    void setEmbedding(const std::string& token, std::vector<float> embedding);
    void readFromFiles(const std::string& path2ClassDataFolder);
    void setStats(const std::unordered_map<std::string, int>& stats) { statOfTokens = stats; }

    // Getters for read-only access to internal state
    int getEmbeddingDimension() const { return d; }
    int getDval() const { return d_val; }
    int getVocabularySize() const { return vocSize; }
    const std::unordered_map<std::string, int> getTokenStats() const { return statOfTokens; }
    std::vector<float> getEmbeddingForToken(int index) const { return embeddings[index]; };
    std::vector<float> getEmbeddingForToken(const std::string& token) const;
    const std::vector<std::string>& getTokens() const { return tokens; }
    const std::vector<std::string>& getVocab() const { return vocab_tokens; }
    const int getIndexOfToken(const std::string& token) const;
    const std::vector<std::vector<float>>& getEmbeddings() const { return embeddings; }
    const std::vector<std::vector<float>>& getDeEmbeddings() const { return deEmbeddings; }
    float getEmbedding(int i, int j) { return embeddings[i][j]; }
    float getDeEmbedding(int i, int j) { return deEmbeddings[i][j]; }

    void splitWord(const std::string& word, std::vector<std::string>& subwords) const;
    void splitSentence(const std::string& sentence, std::vector<std::string>& all_subwords) const;
    void splitSentence(const std::string& sentence, std::vector<std::string>& all_subwords, std::vector<int>& indices) const;
    void buildCorpusWordCounts(const std::vector<std::string>& file_paths, std::unordered_map<std::string, int>& corpus_word_counts);
    void groupCommonTokens(const std::unordered_map<std::string, int>& corpus_word_counts, int num_merges, std::vector<std::string>& final_vocab);
    void learn_vocabulary_from_word_counts(const std::unordered_map<std::string, int>& corpus_word_counts, int num_merges, std::vector<std::string>& final_vocab);
    void saveUniqueTokensToCSV(const std::unordered_map<std::string, int>& corpus_word_counts, const std::string& outputPath);
    void calculateTokenStatsFromCounts(const std::unordered_map<std::string, int>& corpus_word_counts, const std::string& outputPath);
    void generateAndSaveEmbeddings(const std::string& outputPath, float r1, float r2);
    void generateAndSavedeEmbeddings(const std::string& outputPath, float r1, float r2);
    void saveEmbeddings(const std::string& outputPath, const std::vector<std::vector<float>>& Emebdding);
    void saveEmbeddings(const std::string& outputPath, const float* Emebdding);
    void savedeEmbeddings(const std::string& outputPath, const std::vector<std::vector<float>>& deEmebdding);
    void savedeEmbeddings(const std::string& outputPath, const float* deEmebdding);

    #ifdef USE_CUDA
        void cuEmbeddingFormula(std::vector<std::vector<float>>& embedding, const std::vector<float>& seeds, int& d, int& vocSize, float r1, float r2);
        void cudeEmbeddingFormula(std::vector<std::vector<float>>& embedding, const std::vector<float>& seeds, int& d, int& vocSize, float r1, float r2);
        void cuVectorInverse(std::vector<std::vector<float>>& deEmbedding, const std::vector<std::vector<float>>& embedding, int& d, int& vocSize);
    #elif USE_OPENCL
        void clEmbeddingFormula(OpenCLContext& ocl, std::vector<std::vector<float>>& embedding, const std::vector<float>& seeds, int& d, int& vocSize, float r1, float r2);
        void cldeEmbeddingFormula(OpenCLContext& ocl, std::vector<std::vector<float>>& embedding, const std::vector<float>& seeds, int& d, int& vocSize, float r1, float r2);
        void clVectorInverse(OpenCLContext& ocl, std::vector<std::vector<float>>& deEmbedding, const std::vector<std::vector<float>>& embedding, int& d, int& vocSize);
    #endif

    // training function
    void train(const std::string& path2trainData, int numOfmerges, const std::string& path2tokenData);

    ~tokeniser() = default;
};


/**
 * @brief A thread-safe queue designed for producer-consumer patterns.
 * @tparam T The type of elements to store in the queue.
 */
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool done_ = false;

public:
    // Explicitly delete copy/move for ThreadSafeQueue due to std::mutex and std::condition_variable
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue(ThreadSafeQueue&&) = delete;
    ThreadSafeQueue& operator=(ThreadSafeQueue&&) = delete;

    ThreadSafeQueue() = default; // Default constructor is fine

    /**
     * @brief Pushes a new item onto the queue and notifies a waiting consumer.
     */
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        cv_.notify_one();
    }

    /**
     * @brief Waits for an item and pops it from the queue.
     * @param item The reference to store the popped item in.
     * @return `false` if the queue is closed and empty, `true` otherwise.
     */
    bool wait_and_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || done_; });

        if (queue_.empty() && done_) {
            return false;
        }

        item = std::move(queue_.front());
            queue_.pop();
        return true;
    }

    /**
     * @brief Signals to all consumers that production is complete.
     */
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
        cv_.notify_all(); // Wake up all waiting threads so they can exit.
    }
};


std::vector<std::string_view> pre_split_word(std::string_view word);
std::unordered_map<std::pair<std::string, std::string>, int, PairHash> get_pair_stats(const std::unordered_map<std::string, int>& word_counts, const std::unordered_map<std::string, std::vector<std::string>>& splits);
void merge_pair(const std::pair<std::string, std::string>& best_pair, std::unordered_map<std::string, std::vector<std::string>>& splits, int num_threads);
std::unordered_map<std::string, int> mergeTwoMaps(std::unordered_map<std::string, int> map1, std::unordered_map<std::string, int> map2);
std::future<std::unordered_map<std::string, int>> merge_maps(std::vector<std::future<std::unordered_map<std::string, int>>>& futures,size_t start_idx, size_t end_idx);
std::vector<float> vectorInverse(const std::vector<float>& vec);
std::vector<std::string> pre_tokenize_word_by_corpus_freq(const std::string& word, const std::unordered_map<std::string, int>& corpus_word_counts);

unsigned long long count_lines(const std::string& filename);
std::string trim(const std::string& str);
std::string removeQuotes(const std::string& str);
std::string escapeAndQuoteCsvField(const std::string& field);
bool isHeaderLine(const std::string& line);
std::vector<std::string> readSingleColumnCsv(const std::string& filename);
std::vector<std::string> readSpecificColumnFromCsv(const std::string& filename, int targetColumnIndex);
std::vector<std::vector<float>> readCsvTo2DVector(const std::string& filename);
std::unordered_map<std::string, int> readUnorderedMap(const std::string& filename);
std::unordered_map<std::string, std::vector<float>> readMappedEmbeddings(const std::string& filename);

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// kernel for embedding calculation
__global__ void embeddingFormulaBatchKernel(float* all_embeddings, const float* all_seeds,
            const int N, const int d);
// kernel for vector inverse calculation
__global__ void batchedVectorInverseKernel(float* output, const float* input, int N, int d);
#endif

#endif // TOKENISE_HPP