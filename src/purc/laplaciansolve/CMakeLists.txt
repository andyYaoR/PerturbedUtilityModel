cmake_minimum_required(VERSION 3.18)
project(laplaciansolve_native LANGUAGES CXX)

# ── Language standard / options ─────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LAPLACIANSOLVE_BUILD_TESTS "Build the standalone C++ ctest suite" OFF)

# ── Python + nanobind discovery ─────────────────────────────────────
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Locate nanobind's CMake config via the installed Python module so this works
# both under scikit-build-core and a bare `cmake` invocation (for ctest).
if(NOT nanobind_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE _NB_CMAKE_DIR
    RESULT_VARIABLE _NB_RESULT)
  if(_NB_RESULT EQUAL 0 AND _NB_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${_NB_CMAKE_DIR}")
  endif()
endif()
find_package(nanobind CONFIG REQUIRED)

# ── Core CPU module ─────────────────────────────────────────────────
# The algorithm lives in header-only templated form under this directory; the
# binding TU (nb_core_module.cpp) is the only compiled source for now.
nanobind_add_module(_laplaciansolve_core nb_core_module.cpp)
target_include_directories(_laplaciansolve_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Aggressive optimization for the hot solve kernel.  -march=native targets the
# build machine (fine for local research use; disable for portable wheels).
option(LAPLACIANSOLVE_NATIVE_ARCH "Optimize for the build machine's CPU" ON)
if(NOT MSVC)
  target_compile_options(_laplaciansolve_core PRIVATE -O3 -funroll-loops)
  if(LAPLACIANSOLVE_NATIVE_ARCH)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" _LS_HAS_MARCH_NATIVE)
    if(_LS_HAS_MARCH_NATIVE)
      target_compile_options(_laplaciansolve_core PRIVATE -march=native)
    endif()
  endif()
endif()

# ── Threads for the std::thread batch pool (parallel.hpp) ───────────
# The native core parallelizes batch solves with a small persistent std::thread
# pool (parallel.hpp), NOT OpenMP, so it links no OpenMP runtime of its own.
# That removes the dual-runtime crash that otherwise occurs when our libomp
# coexists with CHOLMOD's (Homebrew) libomp and torch's libomp at large n, and it
# is cross-platform (Linux/macOS/Windows) with no libomp discovery, no @rpath
# fix-up, and no KMP_DUPLICATE_LIB_OK.  Threads::Threads adds -pthread as needed.
find_package(Threads REQUIRED)
target_link_libraries(_laplaciansolve_core PRIVATE Threads::Threads)

install(TARGETS _laplaciansolve_core LIBRARY DESTINATION purc/laplaciansolve)

# ── Optional CUDA module (built only when a CUDA compiler is present) ─
# CPU-only hosts (e.g. CI, laptops) skip this entirely and build the CPU core
# only.  On a CUDA machine the kernel bodies under cuda/*.cu get filled in;
# until then they raise "not implemented" (see docs/CUDA.md).
option(LAPLACIANSOLVE_ENABLE_CUDA "Build the CUDA module when a compiler is found" ON)

if(LAPLACIANSOLVE_ENABLE_CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    find_program(_LS_NVCC nvcc
      HINTS ENV CUDA_HOME ENV CUDA_PATH ENV CUDA_ROOT /usr/local/cuda
      PATH_SUFFIXES bin)
    if(_LS_NVCC)
      set(CMAKE_CUDA_COMPILER "${_LS_NVCC}")
    endif()
  endif()

  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "LaplacianSolve: CUDA enabled (${CMAKE_CUDA_COMPILER}).")
    nanobind_add_module(
      _laplaciansolve_cuda
      nb_cuda_module.cpp
      cuda/ldl_solve.cu
      cuda/spmv_batched.cu
      cuda/pcg_batched.cu
      cuda/forest_ldl_batched.cu)
    target_include_directories(_laplaciansolve_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(_laplaciansolve_cuda PROPERTIES CUDA_STANDARD 17
                                                          CUDA_STANDARD_REQUIRED ON)
    install(TARGETS _laplaciansolve_cuda LIBRARY DESTINATION purc/laplaciansolve)
  else()
    message(STATUS "LaplacianSolve: no CUDA compiler found; building CPU core only.")
  endif()
endif()

# ── CHOLMOD direct SPD solver (REQUIRED) ────────────────────────────
# SuiteSparse/CHOLMOD is a hard dependency: it is the direct factorization behind
# the batched Newton solve over OD-pairs (per-system distinct matrices), for which
# there is no iterative batched route.  The build fails if it is not found, so every
# install has the same (direct) batched performance.
option(LAPLACIANSOLVE_CHOLMOD "Build the CHOLMOD direct SPD solver (required)" ON)
if(LAPLACIANSOLVE_CHOLMOD)
  find_path(_LS_CHOLMOD_INCLUDE cholmod.h
    HINTS /opt/homebrew/opt/suite-sparse/include/suitesparse
          /opt/homebrew/include/suitesparse /usr/local/opt/suite-sparse/include/suitesparse
          /usr/local/include/suitesparse /usr/include/suitesparse
          ${CMAKE_PREFIX_PATH} ENV CONDA_PREFIX
    PATH_SUFFIXES suitesparse include/suitesparse Library/include/suitesparse)
  find_library(_LS_CHOLMOD_LIB cholmod
    HINTS /opt/homebrew/opt/suite-sparse/lib /opt/homebrew/lib
          /usr/local/opt/suite-sparse/lib /usr/local/lib /usr/lib
          ${CMAKE_PREFIX_PATH} ENV CONDA_PREFIX
    PATH_SUFFIXES lib Library/lib)
  if(_LS_CHOLMOD_INCLUDE AND _LS_CHOLMOD_LIB)
    get_filename_component(_LS_CHOLMOD_LIBDIR "${_LS_CHOLMOD_LIB}" DIRECTORY)
    nanobind_add_module(_laplaciansolve_cholmod nb_cholmod_module.cpp)
    target_include_directories(_laplaciansolve_cholmod
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${_LS_CHOLMOD_INCLUDE})
    target_link_libraries(_laplaciansolve_cholmod PRIVATE ${_LS_CHOLMOD_LIB})
    if(NOT MSVC)
      target_link_options(_laplaciansolve_cholmod PRIVATE -Wl,-rpath,${_LS_CHOLMOD_LIBDIR})
      target_compile_options(_laplaciansolve_cholmod PRIVATE -O3)
    endif()
    install(TARGETS _laplaciansolve_cholmod LIBRARY DESTINATION purc/laplaciansolve)
    message(STATUS "LaplacianSolve: CHOLMOD direct solver enabled (${_LS_CHOLMOD_LIB}).")
  else()
    message(FATAL_ERROR
      "SuiteSparse/CHOLMOD is REQUIRED but was not found.  Install it and reinstall:\n"
      "  conda:         conda install -c conda-forge suitesparse\n"
      "  macOS:         brew install suite-sparse\n"
      "  Debian/Ubuntu: sudo apt install libsuitesparse-dev\n"
      "(Advanced: -DLAPLACIANSOLVE_CHOLMOD=OFF builds a single-solve-only variant "
      "without the batched OD-pair path.)")
  endif()
endif()

# ── Standalone C++ tests (ctest), built only on demand ──────────────
if(LAPLACIANSOLVE_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  enable_testing()
  add_subdirectory(tests)
endif()
