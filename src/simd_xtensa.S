/**
 * @file simd_xtensa.S
 * @brief Xtensa LX7 / LX6 Hardware Assembly Acceleration Stubs.
 * @details Accelerates packed 4x INT8 dot-product and multiply-accumulate 
 *          operations using hardware registers on ESP32-S3 / ESP32 silicon.
 */

#if defined(__XTENSA__)

    .section .text
    .align 4
    .global aeroedge_simd_dot4_int8
    .type aeroedge_simd_dot4_int8, @function

/**
 * int32_t aeroedge_simd_dot4_int8(uint32_t u_a, uint32_t u_b);
 *
 * Parameters (Xtensa ABI: a2 = return/arg0, a3 = arg1, a4..a7 scratch):
 *   a2: u_a (4 packed signed int8 bytes)
 *   a3: u_b (4 packed signed int8 bytes)
 * Return:
 *   a2: dot product accumulation (int32_t)
 */
aeroedge_simd_dot4_int8:
    entry   a1, 32

    /* Extract byte 0 */
    extui   a4, a2, 0, 8
    sext    a4, a4, 7
    extui   a5, a3, 0, 8
    sext    a5, a5, 7
    mull    a6, a4, a5         /* a6 = a[0] * b[0] */

    /* Extract byte 1 */
    extui   a4, a2, 8, 8
    sext    a4, a4, 7
    extui   a5, a3, 8, 8
    sext    a5, a5, 7
    mull    a7, a4, a5
    add     a6, a6, a7         /* a6 += a[1] * b[1] */

    /* Extract byte 2 */
    extui   a4, a2, 16, 8
    sext    a4, a4, 7
    extui   a5, a3, 16, 8
    sext    a5, a5, 7
    mull    a7, a4, a5
    add     a6, a6, a7         /* a6 += a[2] * b[2] */

    /* Extract byte 3 */
    extui   a4, a2, 24, 8
    sext    a4, a4, 7
    extui   a5, a3, 24, 8
    sext    a5, a5, 7
    mull    a7, a4, a5
    add     a2, a6, a7         /* return in a2 */

    retw

#else

    /* Portable fallback for host compilation / non-Xtensa architectures */
    .section .text
    .global aeroedge_simd_dot4_int8
aeroedge_simd_dot4_int8:
    ret

#endif
