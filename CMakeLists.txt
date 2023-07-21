cmake_minimum_required (VERSION 3.0)
project (TTP-VFL)
#set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_FLAGS "-Wall -O3")

set(NAME "TTP-VFL")

find_path(CMAKE_FOLDER NAMES cmake/emp-tool-config.cmake)
include(${CMAKE_FOLDER}/cmake/common.cmake)
include(${CMAKE_FOLDER}/cmake/source_of_randomness.cmake)
include(${CMAKE_FOLDER}/cmake/threading.cmake)
find_package(emp-ot REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

INCLUDE_DIRECTORIES(${EMP-OT_INCLUDE_DIRS})

add_executable(ttp_vfl
    src/main.cpp
    src/utils/utils.cpp
    src/model/coordinator.cpp
    src/model/player.cpp
    src/tasks/linear_regression.cpp
    src/tasks/logistic_regression.cpp
)

target_link_libraries(ttp_vfl ${EMP-OT_LIBRARIES} Eigen3::Eigen)