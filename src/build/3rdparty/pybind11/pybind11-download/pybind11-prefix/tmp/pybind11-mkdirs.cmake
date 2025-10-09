# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/mnt/c/Users/lionl/OneDrive/桌面/AWS/SIC/src/build/3rdparty/pybind11/pybind11-src"
  "/mnt/c/Users/lionl/OneDrive/桌面/AWS/SIC/src/build/3rdparty/pybind11/pybind11-build"
  "/mnt/c/Users/lionl/OneDrive/桌面/AWS/SIC/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix"
  "/mnt/c/Users/lionl/OneDrive/桌面/AWS/SIC/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/tmp"
  "/mnt/c/Users/lionl/OneDrive/桌面/AWS/SIC/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp"
  "/mnt/c/Users/lionl/OneDrive/桌面/AWS/SIC/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src"
  "/mnt/c/Users/lionl/OneDrive/桌面/AWS/SIC/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/mnt/c/Users/lionl/OneDrive/桌面/AWS/SIC/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/mnt/c/Users/lionl/OneDrive/桌面/AWS/SIC/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp${cfgdir}") # cfgdir has leading slash
endif()
