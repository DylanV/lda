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
# Adding the headers is not necessary when each has an associated cpp source file, consider just adding sources
#add_library(lib ${SOURCE} )

target_include_directories(lib PUBLIC include)