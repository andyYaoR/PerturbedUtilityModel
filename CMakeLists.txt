cmake_minimum_required(VERSION 3.18)
project(purc_all LANGUAGES CXX)

# Top-level driver for scikit-build-core: builds BOTH native extensions in one
# install, each installing into its own package directory (see each subproject's
# install(TARGETS ... DESTINATION purc/<pkg>)).  This is what lets a single
#   pip install --no-build-isolation -e .
# build the PURC core AND the vendored laplaciansolve solver -- no separate sibling
# install step.  Each subdirectory CMake is self-contained (its own Python/nanobind
# discovery); they are independent targets with no cross-dependency.
add_subdirectory(src/purc/static_purc)
add_subdirectory(src/purc/laplaciansolve)
