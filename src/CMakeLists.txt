add_executable(dssp dssp.cpp)
include_directories(
        ${GROMACS_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}
        )
set_target_properties(dssp PROPERTIES
                      COMPILE_FLAGS "${GROMACS_CXX_FLAGS}")
target_link_libraries(dssp ${GROMACS_LIBRARIES})
