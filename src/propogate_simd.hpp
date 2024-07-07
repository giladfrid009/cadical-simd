#include <immintrin.h>
#include <cstdint>

struct prop_result
{
    int* k;
    signed char v;

    prop_result(int* k, signed char v) : k(k), v(v) { }
};

inline uint8_t extract_epu8var(__m256i val, int index)
{
    union
    {
        __m256i m256;
        int8_t array[32];
    } tmp;
    tmp.m256 = val;
    return tmp.array[index];
}

inline prop_result prop_vanilla(int* k, const int* end, const signed char* vals)
{
    signed char v = -1;

    while (k != end && (v = vals[*k]) < 0)
        k++;

    return prop_result(k, v);
}

inline prop_result prop_simd(int* k, const int* end, const signed char* vals)
{
    // WORKS WEW

    const __m256i neg_one = _mm256_set1_epi8(-1);
    const int JUNK_VALS_MASK = 0x11111111;

    while (k <= end - 8)
    {
        // Load 8 indices
        __m256i indices = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k));

        // Prefetch next indices
        _mm_prefetch(reinterpret_cast<const char*>(k + 8), _MM_HINT_T0);

        // Gather 8 (32-bit) values
        // Note that the gathered data is in the form [r---] [r---] [r---] ... where r is actual byte and the rest is junk
        __m256i values = _mm256_i32gather_epi32(reinterpret_cast<const int*>(vals), indices, 1);

        __m256i cmp_res = _mm256_cmpgt_epi8(values, neg_one);
        int cmp_mask = _mm256_movemask_epi8(cmp_res);

        // zero comparison entries that are not valid
        cmp_mask = cmp_mask & JUNK_VALS_MASK;

        if (cmp_mask != 0)
        {
            int i = __builtin_ctz(cmp_mask);
            k = k + i / 4;
            signed char v = extract_epu8var(values, i);
            return prop_result(k, v);
        }

        k += 8;
    }

    // Handle remaining elements
    signed char v = -1;

    while (k != end && (v = vals[*k]) < 0)
        k++;

    return prop_result(k, v);
}