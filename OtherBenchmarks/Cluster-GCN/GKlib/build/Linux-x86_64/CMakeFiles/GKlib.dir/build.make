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
include CMakeFiles/GKlib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/GKlib.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/GKlib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GKlib.dir/flags.make

CMakeFiles/GKlib.dir/b64.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/b64.c.o: ../../b64.c
CMakeFiles/GKlib.dir/b64.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/GKlib.dir/b64.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/b64.c.o -MF CMakeFiles/GKlib.dir/b64.c.o.d -o CMakeFiles/GKlib.dir/b64.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/b64.c

CMakeFiles/GKlib.dir/b64.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/b64.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/b64.c > CMakeFiles/GKlib.dir/b64.c.i

CMakeFiles/GKlib.dir/b64.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/b64.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/b64.c -o CMakeFiles/GKlib.dir/b64.c.s

CMakeFiles/GKlib.dir/blas.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/blas.c.o: ../../blas.c
CMakeFiles/GKlib.dir/blas.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/GKlib.dir/blas.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/blas.c.o -MF CMakeFiles/GKlib.dir/blas.c.o.d -o CMakeFiles/GKlib.dir/blas.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/blas.c

CMakeFiles/GKlib.dir/blas.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/blas.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/blas.c > CMakeFiles/GKlib.dir/blas.c.i

CMakeFiles/GKlib.dir/blas.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/blas.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/blas.c -o CMakeFiles/GKlib.dir/blas.c.s

CMakeFiles/GKlib.dir/cache.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/cache.c.o: ../../cache.c
CMakeFiles/GKlib.dir/cache.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/GKlib.dir/cache.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/cache.c.o -MF CMakeFiles/GKlib.dir/cache.c.o.d -o CMakeFiles/GKlib.dir/cache.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/cache.c

CMakeFiles/GKlib.dir/cache.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/cache.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/cache.c > CMakeFiles/GKlib.dir/cache.c.i

CMakeFiles/GKlib.dir/cache.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/cache.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/cache.c -o CMakeFiles/GKlib.dir/cache.c.s

CMakeFiles/GKlib.dir/csr.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/csr.c.o: ../../csr.c
CMakeFiles/GKlib.dir/csr.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/GKlib.dir/csr.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/csr.c.o -MF CMakeFiles/GKlib.dir/csr.c.o.d -o CMakeFiles/GKlib.dir/csr.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/csr.c

CMakeFiles/GKlib.dir/csr.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/csr.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/csr.c > CMakeFiles/GKlib.dir/csr.c.i

CMakeFiles/GKlib.dir/csr.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/csr.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/csr.c -o CMakeFiles/GKlib.dir/csr.c.s

CMakeFiles/GKlib.dir/error.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/error.c.o: ../../error.c
CMakeFiles/GKlib.dir/error.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/GKlib.dir/error.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/error.c.o -MF CMakeFiles/GKlib.dir/error.c.o.d -o CMakeFiles/GKlib.dir/error.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/error.c

CMakeFiles/GKlib.dir/error.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/error.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/error.c > CMakeFiles/GKlib.dir/error.c.i

CMakeFiles/GKlib.dir/error.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/error.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/error.c -o CMakeFiles/GKlib.dir/error.c.s

CMakeFiles/GKlib.dir/evaluate.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/evaluate.c.o: ../../evaluate.c
CMakeFiles/GKlib.dir/evaluate.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/GKlib.dir/evaluate.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/evaluate.c.o -MF CMakeFiles/GKlib.dir/evaluate.c.o.d -o CMakeFiles/GKlib.dir/evaluate.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/evaluate.c

CMakeFiles/GKlib.dir/evaluate.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/evaluate.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/evaluate.c > CMakeFiles/GKlib.dir/evaluate.c.i

CMakeFiles/GKlib.dir/evaluate.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/evaluate.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/evaluate.c -o CMakeFiles/GKlib.dir/evaluate.c.s

CMakeFiles/GKlib.dir/fkvkselect.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/fkvkselect.c.o: ../../fkvkselect.c
CMakeFiles/GKlib.dir/fkvkselect.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object CMakeFiles/GKlib.dir/fkvkselect.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/fkvkselect.c.o -MF CMakeFiles/GKlib.dir/fkvkselect.c.o.d -o CMakeFiles/GKlib.dir/fkvkselect.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/fkvkselect.c

CMakeFiles/GKlib.dir/fkvkselect.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/fkvkselect.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/fkvkselect.c > CMakeFiles/GKlib.dir/fkvkselect.c.i

CMakeFiles/GKlib.dir/fkvkselect.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/fkvkselect.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/fkvkselect.c -o CMakeFiles/GKlib.dir/fkvkselect.c.s

CMakeFiles/GKlib.dir/fs.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/fs.c.o: ../../fs.c
CMakeFiles/GKlib.dir/fs.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object CMakeFiles/GKlib.dir/fs.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/fs.c.o -MF CMakeFiles/GKlib.dir/fs.c.o.d -o CMakeFiles/GKlib.dir/fs.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/fs.c

CMakeFiles/GKlib.dir/fs.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/fs.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/fs.c > CMakeFiles/GKlib.dir/fs.c.i

CMakeFiles/GKlib.dir/fs.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/fs.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/fs.c -o CMakeFiles/GKlib.dir/fs.c.s

CMakeFiles/GKlib.dir/getopt.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/getopt.c.o: ../../getopt.c
CMakeFiles/GKlib.dir/getopt.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building C object CMakeFiles/GKlib.dir/getopt.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/getopt.c.o -MF CMakeFiles/GKlib.dir/getopt.c.o.d -o CMakeFiles/GKlib.dir/getopt.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/getopt.c

CMakeFiles/GKlib.dir/getopt.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/getopt.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/getopt.c > CMakeFiles/GKlib.dir/getopt.c.i

CMakeFiles/GKlib.dir/getopt.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/getopt.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/getopt.c -o CMakeFiles/GKlib.dir/getopt.c.s

CMakeFiles/GKlib.dir/gk_util.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/gk_util.c.o: ../../gk_util.c
CMakeFiles/GKlib.dir/gk_util.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building C object CMakeFiles/GKlib.dir/gk_util.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/gk_util.c.o -MF CMakeFiles/GKlib.dir/gk_util.c.o.d -o CMakeFiles/GKlib.dir/gk_util.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_util.c

CMakeFiles/GKlib.dir/gk_util.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/gk_util.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_util.c > CMakeFiles/GKlib.dir/gk_util.c.i

CMakeFiles/GKlib.dir/gk_util.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/gk_util.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_util.c -o CMakeFiles/GKlib.dir/gk_util.c.s

CMakeFiles/GKlib.dir/gkregex.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/gkregex.c.o: ../../gkregex.c
CMakeFiles/GKlib.dir/gkregex.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building C object CMakeFiles/GKlib.dir/gkregex.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/gkregex.c.o -MF CMakeFiles/GKlib.dir/gkregex.c.o.d -o CMakeFiles/GKlib.dir/gkregex.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/gkregex.c

CMakeFiles/GKlib.dir/gkregex.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/gkregex.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/gkregex.c > CMakeFiles/GKlib.dir/gkregex.c.i

CMakeFiles/GKlib.dir/gkregex.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/gkregex.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/gkregex.c -o CMakeFiles/GKlib.dir/gkregex.c.s

CMakeFiles/GKlib.dir/graph.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/graph.c.o: ../../graph.c
CMakeFiles/GKlib.dir/graph.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building C object CMakeFiles/GKlib.dir/graph.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/graph.c.o -MF CMakeFiles/GKlib.dir/graph.c.o.d -o CMakeFiles/GKlib.dir/graph.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/graph.c

CMakeFiles/GKlib.dir/graph.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/graph.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/graph.c > CMakeFiles/GKlib.dir/graph.c.i

CMakeFiles/GKlib.dir/graph.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/graph.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/graph.c -o CMakeFiles/GKlib.dir/graph.c.s

CMakeFiles/GKlib.dir/htable.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/htable.c.o: ../../htable.c
CMakeFiles/GKlib.dir/htable.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building C object CMakeFiles/GKlib.dir/htable.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/htable.c.o -MF CMakeFiles/GKlib.dir/htable.c.o.d -o CMakeFiles/GKlib.dir/htable.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/htable.c

CMakeFiles/GKlib.dir/htable.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/htable.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/htable.c > CMakeFiles/GKlib.dir/htable.c.i

CMakeFiles/GKlib.dir/htable.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/htable.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/htable.c -o CMakeFiles/GKlib.dir/htable.c.s

CMakeFiles/GKlib.dir/io.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/io.c.o: ../../io.c
CMakeFiles/GKlib.dir/io.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building C object CMakeFiles/GKlib.dir/io.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/io.c.o -MF CMakeFiles/GKlib.dir/io.c.o.d -o CMakeFiles/GKlib.dir/io.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/io.c

CMakeFiles/GKlib.dir/io.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/io.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/io.c > CMakeFiles/GKlib.dir/io.c.i

CMakeFiles/GKlib.dir/io.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/io.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/io.c -o CMakeFiles/GKlib.dir/io.c.s

CMakeFiles/GKlib.dir/itemsets.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/itemsets.c.o: ../../itemsets.c
CMakeFiles/GKlib.dir/itemsets.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building C object CMakeFiles/GKlib.dir/itemsets.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/itemsets.c.o -MF CMakeFiles/GKlib.dir/itemsets.c.o.d -o CMakeFiles/GKlib.dir/itemsets.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/itemsets.c

CMakeFiles/GKlib.dir/itemsets.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/itemsets.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/itemsets.c > CMakeFiles/GKlib.dir/itemsets.c.i

CMakeFiles/GKlib.dir/itemsets.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/itemsets.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/itemsets.c -o CMakeFiles/GKlib.dir/itemsets.c.s

CMakeFiles/GKlib.dir/mcore.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/mcore.c.o: ../../mcore.c
CMakeFiles/GKlib.dir/mcore.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building C object CMakeFiles/GKlib.dir/mcore.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/mcore.c.o -MF CMakeFiles/GKlib.dir/mcore.c.o.d -o CMakeFiles/GKlib.dir/mcore.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/mcore.c

CMakeFiles/GKlib.dir/mcore.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/mcore.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/mcore.c > CMakeFiles/GKlib.dir/mcore.c.i

CMakeFiles/GKlib.dir/mcore.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/mcore.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/mcore.c -o CMakeFiles/GKlib.dir/mcore.c.s

CMakeFiles/GKlib.dir/memory.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/memory.c.o: ../../memory.c
CMakeFiles/GKlib.dir/memory.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building C object CMakeFiles/GKlib.dir/memory.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/memory.c.o -MF CMakeFiles/GKlib.dir/memory.c.o.d -o CMakeFiles/GKlib.dir/memory.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/memory.c

CMakeFiles/GKlib.dir/memory.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/memory.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/memory.c > CMakeFiles/GKlib.dir/memory.c.i

CMakeFiles/GKlib.dir/memory.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/memory.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/memory.c -o CMakeFiles/GKlib.dir/memory.c.s

CMakeFiles/GKlib.dir/pqueue.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/pqueue.c.o: ../../pqueue.c
CMakeFiles/GKlib.dir/pqueue.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Building C object CMakeFiles/GKlib.dir/pqueue.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/pqueue.c.o -MF CMakeFiles/GKlib.dir/pqueue.c.o.d -o CMakeFiles/GKlib.dir/pqueue.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/pqueue.c

CMakeFiles/GKlib.dir/pqueue.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/pqueue.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/pqueue.c > CMakeFiles/GKlib.dir/pqueue.c.i

CMakeFiles/GKlib.dir/pqueue.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/pqueue.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/pqueue.c -o CMakeFiles/GKlib.dir/pqueue.c.s

CMakeFiles/GKlib.dir/random.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/random.c.o: ../../random.c
CMakeFiles/GKlib.dir/random.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_19) "Building C object CMakeFiles/GKlib.dir/random.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/random.c.o -MF CMakeFiles/GKlib.dir/random.c.o.d -o CMakeFiles/GKlib.dir/random.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/random.c

CMakeFiles/GKlib.dir/random.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/random.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/random.c > CMakeFiles/GKlib.dir/random.c.i

CMakeFiles/GKlib.dir/random.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/random.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/random.c -o CMakeFiles/GKlib.dir/random.c.s

CMakeFiles/GKlib.dir/rw.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/rw.c.o: ../../rw.c
CMakeFiles/GKlib.dir/rw.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_20) "Building C object CMakeFiles/GKlib.dir/rw.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/rw.c.o -MF CMakeFiles/GKlib.dir/rw.c.o.d -o CMakeFiles/GKlib.dir/rw.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/rw.c

CMakeFiles/GKlib.dir/rw.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/rw.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/rw.c > CMakeFiles/GKlib.dir/rw.c.i

CMakeFiles/GKlib.dir/rw.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/rw.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/rw.c -o CMakeFiles/GKlib.dir/rw.c.s

CMakeFiles/GKlib.dir/seq.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/seq.c.o: ../../seq.c
CMakeFiles/GKlib.dir/seq.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_21) "Building C object CMakeFiles/GKlib.dir/seq.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/seq.c.o -MF CMakeFiles/GKlib.dir/seq.c.o.d -o CMakeFiles/GKlib.dir/seq.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/seq.c

CMakeFiles/GKlib.dir/seq.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/seq.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/seq.c > CMakeFiles/GKlib.dir/seq.c.i

CMakeFiles/GKlib.dir/seq.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/seq.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/seq.c -o CMakeFiles/GKlib.dir/seq.c.s

CMakeFiles/GKlib.dir/sort.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/sort.c.o: ../../sort.c
CMakeFiles/GKlib.dir/sort.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_22) "Building C object CMakeFiles/GKlib.dir/sort.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/sort.c.o -MF CMakeFiles/GKlib.dir/sort.c.o.d -o CMakeFiles/GKlib.dir/sort.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/sort.c

CMakeFiles/GKlib.dir/sort.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/sort.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/sort.c > CMakeFiles/GKlib.dir/sort.c.i

CMakeFiles/GKlib.dir/sort.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/sort.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/sort.c -o CMakeFiles/GKlib.dir/sort.c.s

CMakeFiles/GKlib.dir/string.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/string.c.o: ../../string.c
CMakeFiles/GKlib.dir/string.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_23) "Building C object CMakeFiles/GKlib.dir/string.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/string.c.o -MF CMakeFiles/GKlib.dir/string.c.o.d -o CMakeFiles/GKlib.dir/string.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/string.c

CMakeFiles/GKlib.dir/string.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/string.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/string.c > CMakeFiles/GKlib.dir/string.c.i

CMakeFiles/GKlib.dir/string.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/string.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/string.c -o CMakeFiles/GKlib.dir/string.c.s

CMakeFiles/GKlib.dir/timers.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/timers.c.o: ../../timers.c
CMakeFiles/GKlib.dir/timers.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_24) "Building C object CMakeFiles/GKlib.dir/timers.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/timers.c.o -MF CMakeFiles/GKlib.dir/timers.c.o.d -o CMakeFiles/GKlib.dir/timers.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/timers.c

CMakeFiles/GKlib.dir/timers.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/timers.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/timers.c > CMakeFiles/GKlib.dir/timers.c.i

CMakeFiles/GKlib.dir/timers.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/timers.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/timers.c -o CMakeFiles/GKlib.dir/timers.c.s

CMakeFiles/GKlib.dir/tokenizer.c.o: CMakeFiles/GKlib.dir/flags.make
CMakeFiles/GKlib.dir/tokenizer.c.o: ../../tokenizer.c
CMakeFiles/GKlib.dir/tokenizer.c.o: CMakeFiles/GKlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_25) "Building C object CMakeFiles/GKlib.dir/tokenizer.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GKlib.dir/tokenizer.c.o -MF CMakeFiles/GKlib.dir/tokenizer.c.o.d -o CMakeFiles/GKlib.dir/tokenizer.c.o -c /home/chris/Desktop/Cluster-GCN-Code/GKlib/tokenizer.c

CMakeFiles/GKlib.dir/tokenizer.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/GKlib.dir/tokenizer.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/chris/Desktop/Cluster-GCN-Code/GKlib/tokenizer.c > CMakeFiles/GKlib.dir/tokenizer.c.i

CMakeFiles/GKlib.dir/tokenizer.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/GKlib.dir/tokenizer.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/chris/Desktop/Cluster-GCN-Code/GKlib/tokenizer.c -o CMakeFiles/GKlib.dir/tokenizer.c.s

# Object files for target GKlib
GKlib_OBJECTS = \
"CMakeFiles/GKlib.dir/b64.c.o" \
"CMakeFiles/GKlib.dir/blas.c.o" \
"CMakeFiles/GKlib.dir/cache.c.o" \
"CMakeFiles/GKlib.dir/csr.c.o" \
"CMakeFiles/GKlib.dir/error.c.o" \
"CMakeFiles/GKlib.dir/evaluate.c.o" \
"CMakeFiles/GKlib.dir/fkvkselect.c.o" \
"CMakeFiles/GKlib.dir/fs.c.o" \
"CMakeFiles/GKlib.dir/getopt.c.o" \
"CMakeFiles/GKlib.dir/gk_util.c.o" \
"CMakeFiles/GKlib.dir/gkregex.c.o" \
"CMakeFiles/GKlib.dir/graph.c.o" \
"CMakeFiles/GKlib.dir/htable.c.o" \
"CMakeFiles/GKlib.dir/io.c.o" \
"CMakeFiles/GKlib.dir/itemsets.c.o" \
"CMakeFiles/GKlib.dir/mcore.c.o" \
"CMakeFiles/GKlib.dir/memory.c.o" \
"CMakeFiles/GKlib.dir/pqueue.c.o" \
"CMakeFiles/GKlib.dir/random.c.o" \
"CMakeFiles/GKlib.dir/rw.c.o" \
"CMakeFiles/GKlib.dir/seq.c.o" \
"CMakeFiles/GKlib.dir/sort.c.o" \
"CMakeFiles/GKlib.dir/string.c.o" \
"CMakeFiles/GKlib.dir/timers.c.o" \
"CMakeFiles/GKlib.dir/tokenizer.c.o"

# External object files for target GKlib
GKlib_EXTERNAL_OBJECTS =

libGKlib.a: CMakeFiles/GKlib.dir/b64.c.o
libGKlib.a: CMakeFiles/GKlib.dir/blas.c.o
libGKlib.a: CMakeFiles/GKlib.dir/cache.c.o
libGKlib.a: CMakeFiles/GKlib.dir/csr.c.o
libGKlib.a: CMakeFiles/GKlib.dir/error.c.o
libGKlib.a: CMakeFiles/GKlib.dir/evaluate.c.o
libGKlib.a: CMakeFiles/GKlib.dir/fkvkselect.c.o
libGKlib.a: CMakeFiles/GKlib.dir/fs.c.o
libGKlib.a: CMakeFiles/GKlib.dir/getopt.c.o
libGKlib.a: CMakeFiles/GKlib.dir/gk_util.c.o
libGKlib.a: CMakeFiles/GKlib.dir/gkregex.c.o
libGKlib.a: CMakeFiles/GKlib.dir/graph.c.o
libGKlib.a: CMakeFiles/GKlib.dir/htable.c.o
libGKlib.a: CMakeFiles/GKlib.dir/io.c.o
libGKlib.a: CMakeFiles/GKlib.dir/itemsets.c.o
libGKlib.a: CMakeFiles/GKlib.dir/mcore.c.o
libGKlib.a: CMakeFiles/GKlib.dir/memory.c.o
libGKlib.a: CMakeFiles/GKlib.dir/pqueue.c.o
libGKlib.a: CMakeFiles/GKlib.dir/random.c.o
libGKlib.a: CMakeFiles/GKlib.dir/rw.c.o
libGKlib.a: CMakeFiles/GKlib.dir/seq.c.o
libGKlib.a: CMakeFiles/GKlib.dir/sort.c.o
libGKlib.a: CMakeFiles/GKlib.dir/string.c.o
libGKlib.a: CMakeFiles/GKlib.dir/timers.c.o
libGKlib.a: CMakeFiles/GKlib.dir/tokenizer.c.o
libGKlib.a: CMakeFiles/GKlib.dir/build.make
libGKlib.a: CMakeFiles/GKlib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_26) "Linking C static library libGKlib.a"
	$(CMAKE_COMMAND) -P CMakeFiles/GKlib.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GKlib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GKlib.dir/build: libGKlib.a
.PHONY : CMakeFiles/GKlib.dir/build

CMakeFiles/GKlib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GKlib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GKlib.dir/clean

CMakeFiles/GKlib.dir/depend:
	cd /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chris/Desktop/Cluster-GCN-Code/GKlib /home/chris/Desktop/Cluster-GCN-Code/GKlib /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64 /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64 /home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/CMakeFiles/GKlib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/GKlib.dir/depend
