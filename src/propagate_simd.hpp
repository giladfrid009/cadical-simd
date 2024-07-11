#ifndef PROPAGATE_SIMD_HPP
#define PROPAGATE_SIMD_HPP

#include "internal.hpp"

namespace CaDiCaL
{
    struct prop_result
    {
        int* k;
        signed char v;

        prop_result(int* k, signed char v) : k(k), v(v) { }
    };
}

#endif // PROPAGATE_SIMD_HPP