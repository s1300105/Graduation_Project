static inline void* memcpy_P(void* dest, PGM_VOID_P src, size_t count) {
    const uint8_t* read = (const uint8_t*)(src);
    uint8_t* write = (uint8_t*)(dest);

    while (count)
    {
        *write++ = pgm_read_byte(read++);
        count--;
    }

    return dest;
}