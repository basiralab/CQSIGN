# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chris/Desktop/Cluster-GCN-Code/GKlib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64

# Include any dependencies generated for this target.
include test/CMakeFiles/fis.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/fis.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/fis.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/fis.dir/flags.make

test/CMakeFiles/fis.dir/fis.c.o: test/CMakeFiles/fis.dir/flags.make
test/CMakeFiles/fis.dir/fis.c.o: ../../test/fis.c
test/CMakeFiles/fis.dir/fis.c.o: test/CMakeFiles/fis.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object test/CMakeFiles/fis.dir/fis.c.o"
	cd /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT test/CMakeFiles/fis.dir/fis.c.o -MF CMakeFiles/fis.dir/fis.c.o.d -o CMakeFiles/fis.dir/fis.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/test/fis.c

test/CMakeFiles/fis.dir/fis.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/fis.dir/fis.c.i"
	cd /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/test/fis.c > CMakeFiles/fis.dir/fis.c.i

test/CMakeFiles/fis.dir/fis.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/fis.dir/fis.c.s"
	cd /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/test/fis.c -o CMakeFiles/fis.dir/fis.c.s

# Object files for target fis
fis_OBJECTS = \
"CMakeFiles/fis.dir/fis.c.o"

# External object files for target fis
fis_EXTERNAL_OBJECTS =

test/fis: test/CMakeFiles/fis.dir/fis.c.o
test/fis: test/CMakeFiles/fis.dir/build.make
test/fis: libGKlib.a
test/fis: test/CMakeFiles/fis.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable fis"
	cd /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fis.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/fis.dir/build: test/fis
.PHONY : test/CMakeFiles/fis.dir/build

test/CMakeFiles/fis.dir/clean:
	cd /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test && $(CMAKE_COMMAND) -P CMakeFiles/fis.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/fis.dir/clean

test/CMakeFiles/fis.dir/depend:
	cd /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chris/Desktop/Cluster-GCN-Code/GKlib /home/chris/Desktop/Cluster-GCN-Code/GKlib/test /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64 /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test/CMakeFiles/fis.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/fis.dir/depend

