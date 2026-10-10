idf_component_register(
    SRCS "main.cpp"
         "../src/kernel_core.cpp"
         "../src/ops_dispatch.cpp"
         "../src/simd_xtensa.S"
    INCLUDE_DIRS "."
                 "../include"
                 "../models"
)
