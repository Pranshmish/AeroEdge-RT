// Pre-computed quantized INT8 weights (196 x 10) for vision classifier
// Deterministic synthetic weights for edge micro-benchmark
#define ROW_196(v) \
  v, v+1, v-2, v+3, v, v-1, v+2, v-3, v+4, v-2, v+1, v, v-3, v+2, v-1, v+3, \
  v, v-2, v+1, v-1, v+2, v-3, v+4, v, v-1, v+2, v-2, v+1, v, v-3, v+2, v-1, \
  v+1, v-2, v+3, v, v-1, v+2, v-3, v+4, v-2, v+1, v, v-3, v+2, v-1, v+3, v, \
  v-2, v+1, v-1, v+2, v-3, v+4, v, v-1, v+2, v-2, v+1, v, v-3, v+2, v-1, v+1, \
  v-2, v+3, v, v-1, v+2, v-3, v+4, v-2, v+1, v, v-3, v+2, v-1, v+3, v, v-2, \
  v+1, v-1, v+2, v-3, v+4, v, v-1, v+2, v-2, v+1, v, v-3, v+2, v-1, v+1, v-2, \
  v+3, v, v-1, v+2, v-3, v+4, v-2, v+1, v, v-3, v+2, v-1, v+3, v, v-2, v+1, \
  v-1, v+2, v-3, v+4, v, v-1, v+2, v-2, v+1, v, v-3, v+2, v-1, v+1, v-2, v+3, \
  v, v-1, v+2, v-3, v+4, v-2, v+1, v, v-3, v+2, v-1, v+3, v, v-2, v+1, v-1, \
  v+2, v-3, v+4, v, v-1, v+2, v-2, v+1, v, v-3, v+2, v-1, v+1, v-2, v+3, v, \
  v-1, v+2, v-3, v+4, v-2, v+1, v, v-3, v+2, v-1, v+3, v, v-2, v+1, v-1, v+2, \
  v-3, v+4, v, v-1, v+2, v-2, v+1, v, v-3, v+2, v-1, v+1, v-2, v+3, v, v-1, \
  v+2, v-3, v+4, v

ROW_196(5),
ROW_196(-8),
ROW_196(12),
ROW_196(-15),
ROW_196(9),
ROW_196(-4),
ROW_196(14),
ROW_196(-11),
ROW_196(7),
ROW_196(-6)

#undef ROW_196
