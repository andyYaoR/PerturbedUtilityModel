cmake_minimum_required(VERSION 3.18)
project(purcsolver_native LANGUAGES CXX)

# ── Language standard / options ─────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Python + nanobind discovery ─────────────────────────────────────
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Locate nanobind's CMake config via the installed Python module so this works
# both under scikit-build-core and a bare `cmake` invocation.
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
# The PURC kernels (xi*(eta) recovery, weight/active-mask + CSC assembly, and a
# full GIL-released SSN step) live in header-only templated form alongside the
# binding TU.  M0 ships only the binding + a GIL-release smoke kernel to prove
# the end-to-end build path; numerical kernels land in M1+.
nanobind_add_module(_purcsolver_core nb_core_module.cpp)
target_include_directories(_purcsolver_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Aggressive optimization for the hot kernels.  -march=native targets the build
# machine (fine for local research use; disable for portable wheels).
option(PURCSOLVER_NATIVE_ARCH "Optimize for the build machine's CPU" ON)
if(NOT MSVC)
  target_compile_options(_purcsolver_core PRIVATE -O3 -funroll-loops)
  if(PURCSOLVER_NATIVE_ARCH)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" _PS_HAS_MARCH_NATIVE)
    if(_PS_HAS_MARCH_NATIVE)
      target_compile_options(_purcsolver_core PRIVATE -march=native)
    endif()
  endif()
endif()

install(TARGETS _purcsolver_core LIBRARY DESTINATION .)
