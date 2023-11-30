#pragma once
#include <immintrin.h>
#include <stdint.h>

static inline uint32_t pdep(uint32_t source, uint32_t mask) {
    return _pdep_u32(source, mask);
}
static inline uint32_t pext(uint32_t source, uint32_t mask) {
    return _pext_u32(source, mask);
}

#define BMI_2D_X_MASK 0x55555555
#define BMI_2D_Y_MASK 0xAAAAAAAA

static inline uint32_t m2D_e_BMI(const uint16_t x, const uint16_t y) {
    return pdep(x, BMI_2D_X_MASK) | pdep(y, BMI_2D_Y_MASK);
}

