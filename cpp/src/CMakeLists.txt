#project(lda_lib)
set(src ${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCE
        lda.cpp
        util.cpp
        data.cpp
        dirichlet.cpp
        )
set(HEADERS
        lda.h
        util.h
        data.h
        dirichlet.h
        )

add_library(lib ${SOURCE} ${HEADERS} )

target_include_directories(lib PUBLIC include)