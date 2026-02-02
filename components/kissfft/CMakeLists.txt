idf_component_register(
    SRCS
        "src/mfcc_kissfft.cpp"
        "src/kissfft/kiss_fft.c"
        "src/kissfft/kiss_fftr.c"
    INCLUDE_DIRS
        "include"
        "src/kissfft"
)

