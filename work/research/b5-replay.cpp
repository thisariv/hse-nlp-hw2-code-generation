// Exact bitset vote from analyze-b3-results.py. No model or answer-code execution.
// Compile without fast-math: Python's left-to-right double sum is preserved.
#include <cstdint>
#include <limits>

extern "C" void replay_many(int rows, int candidates, int configs,
    const uint64_t* adjacency, const uint64_t* valid, const double* lp,
    const uint8_t* correct, const uint64_t* masks, uint64_t first,
    uint8_t* output, int* stopped_out) {
    int stopped = 0;
    for (int i = 0; i < rows; ++i)
        stopped += ((first & valid[i]) == first &&
                    (first & adjacency[i*candidates]) == first);
    *stopped_out = stopped;
    for (int k = 0; k < configs; ++k) {
        for (int i = 0; i < rows; ++i) {
            const uint64_t* adj = adjacency + i*candidates;
            const double* logp = lp + i*candidates;
            bool stop = ((first & valid[i]) == first && (first & adj[0]) == first);
            uint64_t remaining = (stop ? first : masks[k]) & valid[i];
            uint64_t best = 0;
            int best_count = -1;
            bool best_greedy = false;
            double best_mean = -std::numeric_limits<double>::infinity();
            while (remaining) {
                uint64_t seed = remaining & -remaining;
                uint64_t component = seed, frontier = seed;
                while (frontier) {
                    uint64_t bit = frontier & -frontier;
                    frontier ^= bit;
                    uint64_t added = adj[__builtin_ctzll(bit)] & remaining & ~component;
                    component |= added;
                    frontier |= added;
                }
                remaining &= ~component;
                int count = __builtin_popcountll(component);
                bool greedy = bool(component & adj[0]);
                double sum = 0;
                for (uint64_t bits = component; bits; bits &= bits-1)
                    sum += logp[__builtin_ctzll(bits)];
                double mean = sum/count;
                if (count > best_count ||
                    (count == best_count && greedy > best_greedy) ||
                    (count == best_count && greedy == best_greedy && mean > best_mean)) {
                    best = component; best_count = count;
                    best_greedy = greedy; best_mean = mean;
                }
            }
            int winner = 0;
            if (best && !(best & 1)) {
                double highest = -std::numeric_limits<double>::infinity();
                for (uint64_t bits = best; bits; bits &= bits-1) {
                    int j = __builtin_ctzll(bits);
                    if (logp[j] > highest) { highest = logp[j]; winner = j; }
                }
            }
            output[k*rows+i] = correct[i*candidates+winner];
        }
    }
}
