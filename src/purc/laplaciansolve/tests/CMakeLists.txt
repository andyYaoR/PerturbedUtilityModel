# Standalone C++ test suite (ctest).  Built only when LAPLACIANSOLVE_BUILD_TESTS
# is ON; the headers under the parent directory are the units under test.
add_executable(test_core test_core.cpp)
target_include_directories(test_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME core_solve COMMAND test_core)

# Thread-pool partition tests (parallel_for_ranges) - needs the threading runtime.
find_package(Threads REQUIRED)
add_executable(test_parallel test_parallel.cpp)
target_include_directories(test_parallel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_parallel PRIVATE Threads::Threads)
add_test(NAME parallel_ranges COMMAND test_parallel)

# Forest LDL: single-matrix residual + batched-factorization parity.
add_executable(test_forest test_forest.cpp)
target_include_directories(test_forest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(test_forest PRIVATE Threads::Threads)
add_test(NAME forest_ldl COMMAND test_forest)
