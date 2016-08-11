cmake_minimum_required(VERSION 2.8)
project(SOT)

# Build release if no build type has been specified
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "NOTE: No build type specified")
    set(CMAKE_BUILD_TYPE Release)
endif()

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Debug build.")
ELSEIF(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Release build.")
ELSE()
    message(STATUS "Some other build type.")
ENDIF()

set(CMAKE_MACOSX_RPATH 1)

add_subdirectory (doc)
add_subdirectory (test)

# The version number.
set (SOT_VERSION_MAJOR 0)
set (SOT_VERSION_MINOR 9)

include_directories(include)

# Move header files
file(GLOB MAIN_HEADER "include/sot.h")
install(FILES ${MAIN_HEADER} DESTINATION include)
file(GLOB BITS_HEADERS "include/sot_bits/*.h")
install(FILES ${BITS_HEADERS} DESTINATION include/sot_bits)

# Test
enable_testing()
add_test (NAME test_kNN COMMAND test_kNN)
add_test (NAME test_rbf COMMAND test_rbf)
add_test (NAME test_shepard COMMAND test_shepard)
add_test (NAME test_dds COMMAND test_dds)
add_test (NAME test_dycors COMMAND test_dycors)
add_test (NAME test_ga COMMAND test_ga)
add_test (NAME test_srbf COMMAND test_srbf)
add_test (NAME test_sync COMMAND test_sync)
add_test (NAME test_ga_sampling COMMAND test_ga_sampling)
add_test (NAME test_utils COMMAND test_utils)
add_test (NAME test_adaptive COMMAND test_adaptive)
