# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake3

# The command to remove a file.
RM = /usr/bin/cmake3 -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans

# Include any dependencies generated for this target.
include Algorithm/CMakeFiles/kmeans.dir/depend.make

# Include the progress variables for this target.
include Algorithm/CMakeFiles/kmeans.dir/progress.make

# Include the compile flags for this target's objects.
include Algorithm/CMakeFiles/kmeans.dir/flags.make

Algorithm/CMakeFiles/kmeans.dir/kmeans.cpp.o: Algorithm/CMakeFiles/kmeans.dir/flags.make
Algorithm/CMakeFiles/kmeans.dir/kmeans.cpp.o: Algorithm/kmeans.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Algorithm/CMakeFiles/kmeans.dir/kmeans.cpp.o"
	cd /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/Algorithm && /opt/rh/devtoolset-8/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kmeans.dir/kmeans.cpp.o -c /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/Algorithm/kmeans.cpp

Algorithm/CMakeFiles/kmeans.dir/kmeans.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kmeans.dir/kmeans.cpp.i"
	cd /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/Algorithm && /opt/rh/devtoolset-8/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/Algorithm/kmeans.cpp > CMakeFiles/kmeans.dir/kmeans.cpp.i

Algorithm/CMakeFiles/kmeans.dir/kmeans.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kmeans.dir/kmeans.cpp.s"
	cd /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/Algorithm && /opt/rh/devtoolset-8/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/Algorithm/kmeans.cpp -o CMakeFiles/kmeans.dir/kmeans.cpp.s

# Object files for target kmeans
kmeans_OBJECTS = \
"CMakeFiles/kmeans.dir/kmeans.cpp.o"

# External object files for target kmeans
kmeans_EXTERNAL_OBJECTS =

Algorithm/kmeans: Algorithm/CMakeFiles/kmeans.dir/kmeans.cpp.o
Algorithm/kmeans: Algorithm/CMakeFiles/kmeans.dir/build.make
Algorithm/kmeans: /usr/lib64/libOpenCL.so
Algorithm/kmeans: Algorithm/CMakeFiles/kmeans.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable kmeans"
	cd /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/Algorithm && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kmeans.dir/link.txt --verbose=$(VERBOSE)
	cd /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/Algorithm && /usr/bin/cmake3 -E copy_if_different /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/Algorithm/kmeans.cl /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/Algorithm

# Rule to build all files generated by this target.
Algorithm/CMakeFiles/kmeans.dir/build: Algorithm/kmeans

.PHONY : Algorithm/CMakeFiles/kmeans.dir/build

Algorithm/CMakeFiles/kmeans.dir/clean:
	cd /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/Algorithm && $(CMAKE_COMMAND) -P CMakeFiles/kmeans.dir/cmake_clean.cmake
.PHONY : Algorithm/CMakeFiles/kmeans.dir/clean

Algorithm/CMakeFiles/kmeans.dir/depend:
	cd /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/Algorithm /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/Algorithm /user/8/.base/kennarda/home/Documents/GPU-Computing/Kmeans/Algorithm/CMakeFiles/kmeans.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Algorithm/CMakeFiles/kmeans.dir/depend

