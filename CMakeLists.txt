cmake_minimum_required(VERSION 3.9)
project(random_forest_c C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -O1")

add_executable(random-forest main.c utils/utils.c utils/utils.h model/model.c model/model.h eval/eval.c eval/eval.h)
