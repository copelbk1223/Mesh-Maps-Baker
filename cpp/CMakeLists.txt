cmake_minimum_required(VERSION 3.21)
project(mesh_baker_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# ---------------------------------------------------------------- pybind11
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
)
FetchContent_MakeAvailable(pybind11)

# ------------------------------------------------- Embree (prebuilt binaries)
set(EMBREE_VER 4.3.3)
if(WIN32)
  set(EMBREE_URL
    "https://github.com/RenderKit/embree/releases/download/v${EMBREE_VER}/embree-${EMBREE_VER}.x64.windows.zip")
elseif(APPLE)
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(EMBREE_URL
      "https://github.com/RenderKit/embree/releases/download/v${EMBREE_VER}/embree-${EMBREE_VER}.arm64.macosx.zip")
  else()
    set(EMBREE_URL
      "https://github.com/RenderKit/embree/releases/download/v${EMBREE_VER}/embree-${EMBREE_VER}.x86_64.macosx.zip")
  endif()
else()
  set(EMBREE_URL
    "https://github.com/RenderKit/embree/releases/download/v${EMBREE_VER}/embree-${EMBREE_VER}.x86_64.linux.tar.gz")
endif()

FetchContent_Declare(embree_bin URL ${EMBREE_URL})
FetchContent_MakeAvailable(embree_bin)

find_package(embree ${EMBREE_VER} REQUIRED
  PATHS ${embree_bin_SOURCE_DIR}
  NO_DEFAULT_PATH
)

# ------------------------------------------------------------------ module
pybind11_add_module(mesh_baker_core src/module.cpp)
target_link_libraries(mesh_baker_core PRIVATE embree)

if(MSVC)
  target_compile_options(mesh_baker_core PRIVATE /O2 /fp:fast)
else()
  target_compile_options(mesh_baker_core PRIVATE -O3 -ffast-math)
endif()

# The Embree shared libs are shipped in the same folder as the module,
# so the module must look next to itself at load time.
if(UNIX AND NOT APPLE)
  set_target_properties(mesh_baker_core PROPERTIES
    INSTALL_RPATH "$ORIGIN" BUILD_WITH_INSTALL_RPATH ON)
elseif(APPLE)
  set_target_properties(mesh_baker_core PROPERTIES
    INSTALL_RPATH "@loader_path" BUILD_WITH_INSTALL_RPATH ON)
endif()
