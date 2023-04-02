cmake_minimum_required(VERSION 3.10)
project(CopCor)

add_library(copcor SHARED
    src/utils.c
    src/copula/entropy.c
    src/copula/wolff.c)


target_compile_features(copcor PUBLIC c_std_17)
target_include_directories(copcor PUBLIC src)

set(COPCOR_HEADERS
    src/utils.h
    src/copula/entropy.h
    src/copula/wolff.h)

set_target_properties(copcor PROPERTIES PUBLIC_HEADER "${COPCOR_HEADERS}")

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(copcor PUBLIC OpenMP::OpenMP_C)
endif()


install(TARGETS copcor LIBRARY)

include(GNUInstallDirs)
install(TARGETS copcor PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/copcor)

