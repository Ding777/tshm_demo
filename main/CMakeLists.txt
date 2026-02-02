idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "." "assets"
    PRIV_REQUIRES
        nvs_flash
        esp_timer
        driver
        kissfft
        tshm
        esp_psram
)


