#project(lda_lib)
set(src ${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCE
        util.cpp
        data.cpp
        dirichlet.cpp
        gibbs.cpp var_bayes.cpp
        var_bayes.h)

set(HEADERS
        lda_model.H
        util.h
        data.h
        dirichlet.h
        gibbs.h
        var_bayes.h)

add_library(lib ${SOURCE} ${HEADERS} )
# Adding the headers is not necessary when each has an associated cpp source file, consider just adding sources
#add_library(lib ${SOURCE} )

target_include_directories(lib PUBLIC include)