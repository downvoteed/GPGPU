if (APPLE)
    SET(TARGETS
        "cpu"
        "stats"
    )
else()
    SET(TARGETS
        "cpu"
        "stats"
        "gpu"
    )
endif()

subdirs(${TARGETS})