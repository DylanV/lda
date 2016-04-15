set(src ${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCE
        ${SOURCE}
        ${src}/main.cpp
        ${src}/lda.cpp
        ${src}/alpha.cpp
        ${src}/util.cpp
        ${src}/data.cpp
        PARENT_SCOPE
        )
set(HEADERS
        ${HEADERS}
        ${src}/lda.h
        ${src}/alpha.h
        ${src}/util.h
        ${src}/data.h
        PARENT_SCOPE
        )