#include <immintrin.h>
#include <cstdint>

struct prop_result
{
    int* k;
    signed char v;

    prop_result(const int* k, signed char v) : k(const_cast<int*>(k)), v(v) { }
};

inline prop_result prop_vanilla(const int* __restrict k, const int* __restrict end, const signed char* __restrict vals)
{
    signed char v = -1;

    while (k != end && (v = vals[*k]) < 0)
        k++;

    return prop_result(k, v);
}

inline prop_result prop_simd_ver1(const int* __restrict k, const int* __restrict end, const signed char* __restrict vals)
{
    const __m256i neg_one = _mm256_set1_epi8(-1);

    while (k <= end - 32)
    {
        // Load 32 indices (1024 bits) using four 256-bit loads
        __m256i idx1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k));
        __m256i idx2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k + 8));
        __m256i idx3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k + 16));
        __m256i idx4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k + 24));

        // Prefetch next indices
        //_mm_prefetch(reinterpret_cast<const char*>(k + 32), _MM_HINT_T0);

        // Gather 32 (32-bit) values
        __m256i v1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(vals), idx1, 1);
        __m256i v2 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(vals), idx2, 1);
        __m256i v3 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(vals), idx3, 1);
        __m256i v4 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(vals), idx4, 1);

        // Pack 32-bit to 16-bit, then to 8-bit
        __m256i v12 = _mm256_packs_epi32(v1, v2);
        __m256i v34 = _mm256_packs_epi32(v3, v4);
        __m256i values = _mm256_packs_epi16(v12, v34);

        __m256i cmp_res = _mm256_cmpgt_epi8(values, neg_one);
        uint32_t cmp_mask = _mm256_movemask_epi8(cmp_res);

        if (cmp_mask != 0)
        {
            int i = __builtin_ctz(cmp_mask);
            signed char v = reinterpret_cast<signed char*>(&values)[i];
            return prop_result(k + i, v);
        }

        k += 32;
    }

    // Handle remaining elements
    while (k < end)
    {
        signed char v = vals[*k];
        if (v >= 0) prop_result(k, v);
        ++k;
    }

    return prop_result(end, -1);
}

inline prop_result simd_simd_ver2(const int* __restrict k, const int* __restrict end, const signed char* __restrict vals)
{
    const __m256i neg_one = _mm256_set1_epi8(-1);

    while (k <= end - 16)
    {
        // Load 16 indices (512 bits) using two 256-bit loads
        __m256i idx1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k));
        __m256i idx2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k + 8));

        // Prefetch next indices
        //_mm_prefetch(reinterpret_cast<const char*>(k + 16), _MM_HINT_T0);

        // Gather 16 (32-bit) values
        __m256i v1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(vals), idx1, 1);
        __m256i v2 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(vals), idx2, 1);

        // Pack 32-bit to 16-bit, then to 8-bit
        __m256i v12 = _mm256_packs_epi32(v1, v2);
        __m256i values = _mm256_packs_epi16(v12, v12);

        __m256i cmp_mask = _mm256_cmpgt_epi8(values, neg_one);
        uint32_t mask = _mm256_movemask_epi8(cmp_mask);

        if ((mask & 0xFFFF) != 0)
        {
            int i = __builtin_ctz(mask);
            signed char v = reinterpret_cast<signed char*>(&values)[i];
            return prop_result(k + i, v);
        }

        k += 16;
    }

    // Handle remaining elements
    while (k < end)
    {
        signed char v = vals[*k];
        if (v >= 0) prop_result(k, v);
        ++k;
    }

    return prop_result(end, -1);
}

inline prop_result prop_simd_ver3(const int* __restrict k, const int* __restrict end, const signed char* __restrict vals)
{
    const __m256i neg_one = _mm256_set1_epi8(-1);
    const int JUNK_VALS_MASK = 0x11111111;

    while (k <= end - 8)
    {
        // Load 8 indices
        __m256i indices = _mm256_loadu_si256((__m256i*)k);

        // Gather 8 (32-bit) values
        // Note that the gathered data is in the form [r---] [r---] [r---] ... where r is actual byte and - is junk data
        __m256i values = _mm256_i32gather_epi32((const int*)vals, indices, 1);

        __m256i cmp_res = _mm256_cmpgt_epi8(values, neg_one);
        int cmp_mask = _mm256_movemask_epi8(cmp_res);

        // mask comparison entries that are not valid
        cmp_mask = cmp_mask & JUNK_VALS_MASK;

        if (cmp_mask != 0)
        {
            int i = __builtin_ctz(cmp_mask);
            signed char v = reinterpret_cast<signed char*>(&values)[i];
            return prop_result(k + i, v);
        }

        k += 8;
    }

    // Handle remaining elements
    while (k < end)
    {
        signed char v = vals[*k];
        if (v >= 0) prop_result(k, v);
        ++k;
    }

    return prop_result(end, -1);
}