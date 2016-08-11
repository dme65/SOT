# Find dependencies
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
message(STATUS "Found BLAS: " ${BLAS_LIBRARIES})
find_package(LAPACK REQUIRED)
message(STATUS "Found LAPACK: " ${LAPACK_LIBRARIES})

# Set the library and include directories
SET(libdirs ${ARMADILLO_SOURCE_DIR} ${BLAS_SOURCE_DIR} ${LAPACK_SOURCE_DIR})
SET(includes ${ARMADILLO_INCLUDE_DIRS} ${BLAS_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS})
SET(libs ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
LINK_DIRECTORIES(${libdirs})
include_directories(${includes} "../include")

# Enable C++11 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no \
        C++11 support. Please use a different C++ compiler.")
endif()


if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -lpthread ")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -lpthread ")
endif()

if(CMAKE_USE_LCOV)
    message(STATUS "Adding support for lcov")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O0 -g -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fprofile-arcs -ftest-coverage")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -msse4")
endif()

#  -DARMA_NO_DEBUG -DNDEBUG -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF

# Build tests
add_executable (test_kNN "test_kNN.cpp")
target_link_libraries(test_kNN ${libs})
add_executable (test_rbf "test_rbf.cpp")
target_link_libraries(test_rbf ${libs})
add_executable (test_shepard "test_shepard.cpp")
target_link_libraries(test_shepard ${libs})
add_executable (test_dds "test_dds.cpp")
target_link_libraries(test_dds ${libs})
add_executable (test_dycors "test_dycors.cpp")
target_link_libraries(test_dycors ${libs})
add_executable (test_ga "test_ga.cpp")
target_link_libraries(test_ga ${libs})
add_executable (test_srbf "test_srbf.cpp")
target_link_libraries(test_srbf ${libs})
add_executable (test_sync "test_sync.cpp")
target_link_libraries(test_sync ${libs})
add_executable (test_ga_sampling "test_ga_sampling.cpp")
target_link_libraries(test_ga_sampling ${libs})
add_executable (test_utils "test_utils.cpp")
target_link_libraries(test_utils ${libs})