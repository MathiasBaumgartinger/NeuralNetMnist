cmake_minimum_required(VERSION 3.14)
project(Neural_Network_Mnist)

set(CMAKE_CXX_STANDARD 14)

add_executable(Neural_Network_Mnist main.cpp Network.cpp Network.h PrepareData.hpp Neuron.cpp Neuron.h)