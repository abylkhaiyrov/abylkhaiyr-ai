from __future__ import annotations

from dataclasses import replace

from llm.data.codespecs import Spec

VECTOR = "std::vector<long long>"
CPP_SOLUTIONS = {
    "sum_of_squares": "long long sum_of_squares(long long n) {\n    long long total = 0;\n    for (long long i = 1; i <= n; i++) total += i * i;\n    return total;\n}",
    "is_prime": "bool is_prime(long long n) {\n    if (n < 2) return false;\n    for (long long d = 2; d * d <= n; d++) if (n % d == 0) return false;\n    return true;\n}",
    "gcd": "long long gcd(long long a, long long b) {\n    while (b != 0) {\n        long long t = a % b;\n        a = b;\n        b = t;\n    }\n    return a;\n}",
    "fibonacci": "long long fibonacci(long long n) {\n    long long a = 0, b = 1;\n    for (long long i = 0; i < n; i++) {\n        long long t = a + b;\n        a = b;\n        b = t;\n    }\n    return a;\n}",
    "count_primes_below": "long long count_primes_below(long long n) {\n    if (n < 3) return 0;\n    std::vector<bool> sieve(n, true);\n    sieve[0] = sieve[1] = false;\n    for (long long i = 2; i * i < n; i++)\n        if (sieve[i]) for (long long j = i * i; j < n; j += i) sieve[j] = false;\n    long long count = 0;\n    for (long long i = 0; i < n; i++) if (sieve[i]) count++;\n    return count;\n}",
    "collatz_steps": "long long collatz_steps(long long n) {\n    long long steps = 0;\n    while (n != 1) {\n        n = n % 2 == 0 ? n / 2 : 3 * n + 1;\n        steps++;\n    }\n    return steps;\n}",
    "product_of_list": f"long long product_of_list(const {VECTOR}& xs) {{\n    long long result = 1;\n    for (long long x : xs) result *= x;\n    return result;\n}}",
    "min_max_difference": f"long long min_max_difference(const {VECTOR}& xs) {{\n    return *std::max_element(xs.begin(), xs.end()) - *std::min_element(xs.begin(), xs.end());\n}}",
    "is_perfect_square": "bool is_perfect_square(long long n) {\n    if (n < 0) return false;\n    long long root = (long long)std::sqrt((double)n);\n    return root * root == n;\n}",
    "hamming_distance": "long long hamming_distance(long long a, long long b) {\n    return __builtin_popcountll((unsigned long long)(a ^ b));\n}",
    "triangular": "long long triangular(long long n) {\n    return n * (n + 1) / 2;\n}",
    "max_subarray_sum": f"long long max_subarray_sum(const {VECTOR}& xs) {{\n    long long best = xs[0], current = xs[0];\n    for (size_t i = 1; i < xs.size(); i++) {{\n        current = std::max(xs[i], current + xs[i]);\n        best = std::max(best, current);\n    }}\n    return best;\n}}",
    "two_sum_exists": f"bool two_sum_exists(const {VECTOR}& xs, long long target) {{\n    std::unordered_set<long long> seen;\n    for (long long x : xs) {{\n        if (seen.count(target - x)) return true;\n        seen.insert(x);\n    }}\n    return false;\n}}",
    "bit_count": "long long bit_count(long long n) {\n    long long count = 0;\n    while (n > 0) {\n        count += n & 1;\n        n >>= 1;\n    }\n    return count;\n}",
    "is_leap_year": "bool is_leap_year(long long year) {\n    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);\n}",
    "digital_root": "long long digital_root(long long n) {\n    while (n >= 10) {\n        long long total = 0;\n        while (n > 0) {\n            total += n % 10;\n            n /= 10;\n        }\n        n = total;\n    }\n    return n;\n}",
    "sum_of_multiples": "long long sum_of_multiples(long long n) {\n    long long total = 0;\n    for (long long i = 1; i < n; i++) if (i % 3 == 0 || i % 5 == 0) total += i;\n    return total;\n}",
    "staircase_ways": "long long staircase_ways(long long n) {\n    long long a = 1, b = 1;\n    for (long long i = 0; i < n; i++) {\n        long long t = a + b;\n        a = b;\n        b = t;\n    }\n    return a;\n}",
    "kth_smallest": f"long long kth_smallest({VECTOR} xs, long long k) {{\n    std::sort(xs.begin(), xs.end());\n    return xs[k - 1];\n}}",
    "majority_element": f"long long majority_element(const {VECTOR}& xs) {{\n    std::map<long long, long long> counts;\n    for (long long x : xs) {{\n        counts[x]++;\n        if (counts[x] > (long long)xs.size() / 2) return x;\n    }}\n    return -1;\n}}",
}


def with_cpp(specs: list[Spec]) -> list[Spec]:
    return [replace(spec, cpp=CPP_SOLUTIONS.get(spec.name, "")) for spec in specs]
