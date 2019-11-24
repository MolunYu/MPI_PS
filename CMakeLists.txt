cmake_minimum_required(VERSION 3.15)
project(ps_mpi)

set(MPI_ROOT /home/yml/mpich-install)
include_directories(${MPI_ROOT}/include)

SET(CMAKE_CXX_COMPILER ${MPI_ROOT}/bin/mpicxx)
SET(CMAKE_C_COMPILER  ${MPI_ROOT}//bin/mpicc)

add_executable(ps_mpi main.cpp parameter_server.cpp parameter_server.h worker.h)