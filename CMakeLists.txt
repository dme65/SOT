cmake_minimum_required(VERSION 3.0)

# The version number.
set (SOT_VERSION_MAJOR 0)
set (SOT_VERSION_MINOR 1)

project(SOT)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MACOSX_RPATH 1)

include_directories(include)

find_package(BLAS REQUIRED)
find_package(Armadillo REQUIRED)

set(LIBRARIES
    ${ARMADILLO_LIBRARIES}
    ${BLAS_LIBRARIES}
)

include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${BLAS_INCLUDE_DIRS})
message(STATUS BLAS found: ${BLAS_LIBRARIES})

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/experimental_design.cpp")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -msse4")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Ofast -msse4")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -larmadillo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")

#Generate the shared library from the sources
add_library(SOT SHARED ${SOURCES})
target_link_libraries (SOT Armadillo)

#Set the location for library installation -- i.e., /usr/lib in this case
# not really necessary in this example. Use "sudo make install" to apply
#install(TARGETS SOT DESTINATION /usr/local/lib)

install(TARGETS SOT DESTINATION lib)
file(GLOB MAIN_HEADER "include/sot.h")
install(FILES ${MAIN_HEADER} DESTINATION include/SOT)
file(GLOB BITS_HEADERS "include/sot_bits/*.h")
install(FILES ${BITS_HEADERS} DESTINATION include/SOT/sot_bits)
