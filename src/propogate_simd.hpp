#include <immintrin.h>

// Uncomment the following line to use AVX-512 instructions
//#define USE_AVX512
//#define USE_AVX256
#define USE_AVX128

struct prop_result
{
    int* k;
    signed char v;

    prop_result(int* k, signed char v) : k(k), v(v) { }
};

# if not defined(USE_AVX256) && not defined(USE_AVX512) && not defined(USE_AVX128)

// TODO: COUNT STATISTIC OF THE LOCATION OF THE NNF LITERAL
// TODO: TRY CORRELATE THE LOCATION OF THE NNF LIT WITH THE FORMULA STRUCTURE SOMEHOW.
inline prop_result prop_simd(int* k, const int* end, const signed char* vals)
{
    signed char v = -1;

    while (k != end && (v = vals[*k]) < 0)
        k++;

    return prop_result(k, v);
}

#elif defined(USE_AVX128)

// TODO: FINISH IMPLEMENTING
inline prop_result prop_simd(int* k, const int* end, const signed char* vals)
{
    constexpr int SIMD_SIZE = 4;
    constexpr __mmask16 SIMD_VALS_MASK = static_cast<unsigned short>(0x1111);
    static const __m128i NEG_ONE = _mm128_set1_epi8(-1);

    while (k <= end - SIMD_SIZE)
    {
        // Load indices
        __m256i indices = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k));

        // Gather (32-bit) values
        // Note that the gathered data is in the form [r---] [r---] [r---] ... where r is actual byte and the rest is junk
        __m256i values = _mm256_i32gather_epi32(reinterpret_cast<const int*>(vals), indices, 1);

        // check for which values hold v > -1 (v >= 0 since they are int)
        __mmask32 cmp_mask = _mm256_movemask_epi8(_mm256_cmpgt_epi8(values, NEG_ONE));

        // zero entries that are not valid (remember that we loaded also junk data)
        cmp_mask = cmp_mask & SIMD_VALS_MASK;

        if (cmp_mask != 0)
        {
            int i = __builtin_ctz(cmp_mask);
            int* k_new = k + i / 4;
            return prop_result(k_new, vals[*k_new]);
        }

        k += SIMD_SIZE;
    }

    // Handle remaining elements
    signed char v = -1;

    while (k != end && (v = vals[*k]) < 0)
        k++;

    return prop_result(k, v);
}

# elif defined(USE_AVX256)

// TODO: COUNT STATISTICS HOW MANY OF THE CALLS TO PROP_SIMD END UP IN THE SIMD LOOP
// TODO: COUNT STATISTICS OF HOW MANY ITERS THE SIMD LOOP PERFORMS 
inline prop_result prop_simd(int* k, const int* end, const signed char* vals)
{
    constexpr int SIMD_SIZE = 8;
    constexpr __mmask32 SIMD_VALS_MASK = 0x11111111;
    static const __m256i NEG_ONE = _mm256_set1_epi8(-1);

    while (k <= end - SIMD_SIZE)
    {
        // Load indices
        __m256i indices = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k));

        // Gather (32-bit) values
        // Note that the gathered data is in the form [r---] [r---] [r---] ... where r is actual byte and the rest is junk
        __m256i values = _mm256_i32gather_epi32(reinterpret_cast<const int*>(vals), indices, 1);

        // check for which values hold v > -1 (v >= 0 since they are int)
        __mmask32 cmp_mask = _mm256_movemask_epi8(_mm256_cmpgt_epi8(values, NEG_ONE));

        // zero entries that are not valid (remember that we loaded also junk data)
        cmp_mask = cmp_mask & SIMD_VALS_MASK;

        if (cmp_mask != 0)
        {
            int i = __builtin_ctz(cmp_mask);
            int* k_new = k + i / 4;
            return prop_result(k_new, vals[*k_new]);
        }

        k += SIMD_SIZE;
    }

    // Handle remaining elements
    signed char v = -1;

    while (k != end && (v = vals[*k]) < 0)
        k++;

    return prop_result(k, v);
}

#elif defined(USE_AVX512)

inline prop_result prop_simd(int* k, const int* end, const signed char* vals)
{
    constexpr int SIMD_SIZE = 16;
    constexpr __mmask64 SIMD_VALS_MASK = 0x1111111111111111ULL;
    static const __m512i NEG_ONE = _mm512_set1_epi8(-1);

    while (k <= end - SIMD_SIZE)
    {
        // Load indices
        __m512i indices = _mm512_loadu_si512(k);

        // Gather (32-bit) values
        // Note that the gathered data is in the form [r---] [r---] [r---] ... where r is actual byte and the rest is junk
        __m512i values = _mm512_i32gather_epi32(indices, vals, 1);

        // check for which values hold v > -1 (v >= 0 since they are int)
        __mmask64 cmp_mask = _mm512_cmpgt_epi8_mask(values, NEG_ONE);

        // zero entries that are not valid (remember that we loaded also junk data)
        cmp_mask = cmp_mask & SIMD_VALS_MASK;

        if (cmp_mask != 0)
        {
            int i = __builtin_ctzll(cmp_mask);
            int* k_new = k + i / 4;
            return prop_result(k_new, vals[*k_new]);
        }

        k += SIMD_SIZE;
    }

    // Handle remaining elements
    signed char v = -1;

    while (k != end && (v = vals[*k]) < 0)
        k++;

    return prop_result(k, v);
}

#endif