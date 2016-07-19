cmake_minimum_required(VERSION 2.8)

project(SOT)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MACOSX_RPATH 1)

add_subdirectory (test) 

# The version number.
set (SOT_VERSION_MAJOR 0)
set (SOT_VERSION_MINOR 1)

include_directories(include)

# Move header files
file(GLOB MAIN_HEADER "include/sot.h")
install(FILES ${MAIN_HEADER} DESTINATION include/SOT)
file(GLOB BITS_HEADERS "include/sot_bits/*.h")
install(FILES ${BITS_HEADERS} DESTINATION include/SOT/sot_bits)

# Test
enable_testing()
add_test (NAME test_dycors COMMAND test_dycors)
add_test (NAME test_rbf COMMAND test_rbf)