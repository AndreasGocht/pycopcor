
option(ENABLE_MARCH "Checks if -march=<CPU_TYPE> is available, and applies it" ON)
option(ENABLE_MTUNE "Checks if -mtune=<CPU_TYPE> is available, and applies it" ON)
set(CPU_TYPE "native" CACHE STRING "CPU-Type to use with -march")

include(CheckCompilerFlag)

function(target_compile_optimisation target compiler)
    if(ENABLE_MARCH)
        unset(COMPILER_SUPPORTS_MARCH CACHE)
        check_compiler_flag(${compiler} "-march=${CPU_TYPE}" COMPILER_SUPPORTS_MARCH)
        if(COMPILER_SUPPORTS_MARCH)
            target_compile_options(${target} PUBLIC "-march=${CPU_TYPE}")    
        else()
            message(FATAL_ERROR "${compiler} compiler does not support -march=${CPU_TYPE}")
        endif()
    endif()

    if(ENABLE_MTUNE)
        unset(COMPILER_SUPPORTS_MTUNE CACHE)
        check_compiler_flag(${compiler} "-mtune=${CPU_TYPE}" COMPILER_SUPPORTS_MTUNE)
        if(COMPILER_SUPPORTS_MTUNE)
            target_compile_options(${target} PUBLIC "-mtune=${CPU_TYPE}")
        else()
            message(FATAL_ERROR "${compiler} compiler does not support -mtune=${CPU_TYPE}")
        endif()  
    endif()  
endfunction()

