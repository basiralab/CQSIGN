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
include test/CMakeFiles/strings.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/strings.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/strings.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/strings.dir/flags.make

test/CMakeFiles/strings.dir/strings.c.o: test/CMakeFiles/strings.dir/flags.make
test/CMakeFiles/strings.dir/strings.c.o: ../../test/strings.c
test/CMakeFiles/strings.dir/strings.c.o: test/CMakeFiles/strings.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object test/CMakeFiles/strings.dir/strings.c.o"
	cd /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT test/CMakeFiles/strings.dir/strings.c.o -MF CMakeFiles/strings.dir/strings.c.o.d -o CMakeFiles/strings.dir/strings.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/test/strings.c

test/CMakeFiles/strings.dir/strings.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/strings.dir/strings.c.i"
	cd /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/test/strings.c > CMakeFiles/strings.dir/strings.c.i

test/CMakeFiles/strings.dir/strings.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/strings.dir/strings.c.s"
	cd /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test && /usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/test/strings.c -o CMakeFiles/strings.dir/strings.c.s

# Object files for target strings
strings_OBJECTS = \
"CMakeFiles/strings.dir/strings.c.o"

# External object files for target strings
strings_EXTERNAL_OBJECTS =

test/strings: test/CMakeFiles/strings.dir/strings.c.o
test/strings: test/CMakeFiles/strings.dir/build.make
test/strings: libGKlib.a
test/strings: test/CMakeFiles/strings.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable strings"
	cd /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/strings.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/strings.dir/build: test/strings
.PHONY : test/CMakeFiles/strings.dir/build

test/CMakeFiles/strings.dir/clean:
	cd /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test && $(CMAKE_COMMAND) -P CMakeFiles/strings.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/strings.dir/clean

test/CMakeFiles/strings.dir/depend:
	cd /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chris/Desktop/Cluster-GCN-Code/GKlib /home/chris/Desktop/Cluster-GCN-Code/GKlib/test /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64 /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test/CMakeFiles/strings.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/strings.dir/depend

