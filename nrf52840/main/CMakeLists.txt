# SPDX-License-Identifier: Apache-2.0


cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(nrf52840)


target_sources(app PRIVATE src/main.c)

FILE(GLOB mylib_sources ../lib/*.c)
target_sources(app PRIVATE ${mylib_sources})

# Add the include directory
target_include_directories(app PRIVATE ../include)