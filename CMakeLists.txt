cmake_minimum_required(VERSION 3.16.3)

project(gromacs-dssp-ng CXX)

set(CMAKE_CXX_STANDARD 17)  # new
set(CMAKE_CXX_STANDARD_REQUIRED ON) # new
set(CMAKE_CXX_EXTENSIONS OFF) # new

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# CMake modules are in a subdirectory to keep this file cleaner
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # disabled
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # disabled
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # disabled

# In principle, this could be deduced from GROMACS_IS_DOUBLE returned by
# find_package(GROMACS) based on the suffix alone, but it is clearer that the
# user explicitly sets what they want to get, and then need to provide a suffix
# to match.
option(GMX_DOUBLE "Use double precision" OFF)
set(GMX_SUFFIX "" CACHE STRING "Suffix for the GROMACS installation to use (empty for default)")

# This does not allow for a non-suffixed double-precision libgromacs, but
# that should be rare enough for demonstration purposes.
if (GMX_DOUBLE AND NOT GMX_SUFFIX)
    set(GROMACS_SUFFIX "_d")
else()
    set(GROMACS_SUFFIX ${GMX_SUFFIX})
endif()

find_package(GROMACS 2023 REQUIRED)
gromacs_check_double(GMX_DOUBLE)
gromacs_check_compiler(CXX)
include_directories(${GROMACS_INCLUDE_DIRS}) # new
add_definitions(${GROMACS_DEFINITIONS})

# Use static linking on MSVC
if (CMAKE_GENERATOR MATCHES "Visual Studio")
    string(REPLACE /MD /MT CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
    set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE} CACHE STRING "" FORCE)
    string(REPLACE /MD /MT CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
    set(CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG} CACHE STRING "" FORCE)
endif()

add_subdirectory(src)
