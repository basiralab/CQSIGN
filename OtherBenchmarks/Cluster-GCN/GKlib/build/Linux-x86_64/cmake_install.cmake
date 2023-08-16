# Install script for directory: /home/chris/Desktop/Cluster-GCN-Code/GKlib

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/chris/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/libGKlib.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/GKlib.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_arch.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_defs.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_externs.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_getopt.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_macros.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_mkblas.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_mkmemory.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_mkpqueue.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_mkpqueue2.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_mkrandom.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_mksort.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_mkutils.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_ms_inttypes.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_ms_stat.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_ms_stdint.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_proto.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_struct.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gk_types.h"
    "/home/chris/Desktop/Cluster-GCN-Code/GKlib/gkregex.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/test/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/chris/Desktop/Cluster-GCN-Code/GKlib/build/Linux-x86_64/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
