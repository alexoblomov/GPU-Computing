# Version CMake minimum pour les machines de l'Ensimag
cmake_minimum_required (VERSION 2.8.11)

# Nom du projet
project(LabOpenCL CXX)

# Ajoute les options spécifiques pour l'ensimag
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations -std=c++0x")

# Recherche le package OpenCL
find_package(OpenCL REQUIRED)
if(NOT OPENCL_HAS_CXX)
  message(WARNING "C++ bindings not found.  Using Kronos supplied reference OpenCL v1.2 C++ bindings.")
  list(APPEND OpenCL_INCLUDE_DIRS Common)
endif()

# Ajoute les dépendences sur la bibliothèque OPENCL
set(LIBS ${LIBS} ${OpenCL_LIBRARY})
include_directories(${OpenCL_INCLUDE_DIR})
include_directories(SYSTEM ${OPENCL_INCLUDE_DIRS})
add_definitions(${OPENCL_CXX_DEFINITIONS})

include_directories(utils PUBLIC Common)

# Ajoute répertoire contenant des définitions OpenCL étendues
add_subdirectory(Exercise01)
add_subdirectory(Exercise02)
add_subdirectory(Exercise03)
add_subdirectory(Exercise04)
add_subdirectory(Exercise05)