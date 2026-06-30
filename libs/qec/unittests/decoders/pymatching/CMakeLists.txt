# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# External Dependencies 
# ==============================================================================

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Bug in GCC 12 leads to spurious warnings (-Wrestrict)
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
if (CMAKE_COMPILER_IS_GNUCXX
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(gtest PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:--param=evrp-mode=legacy>)
endif()
include(GoogleTest)

# ==============================================================================
add_compile_options(-Wno-attributes) 

add_executable(test_pymatching test_pymatching.cpp)
target_link_libraries(test_pymatching PRIVATE GTest::gtest_main cudaq-qec cudaq::cudaq)
add_dependencies(CUDAQXQECUnitTests test_pymatching)
gtest_discover_tests(test_pymatching)

if(CUDAQ_REALTIME_ROOT AND CUDAQ_REALTIME_INCLUDE_DIR)
  add_executable(test_pymatching_realtime test_pymatching_realtime.cpp)
  target_include_directories(test_pymatching_realtime PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAQX_QEC_SOURCE_DIR}/lib/realtime
  )
  target_compile_definitions(test_pymatching_realtime PRIVATE
    CUDAQ_REALTIME_ROOT)
  target_link_libraries(test_pymatching_realtime PRIVATE
    GTest::gtest_main
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq::cudaq
  )
  set_target_properties(test_pymatching_realtime PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins"
  )
  add_dependencies(test_pymatching_realtime cudaq-qec-pymatching)
  add_dependencies(CUDAQXQECUnitTests test_pymatching_realtime)
  gtest_discover_tests(test_pymatching_realtime)
endif()

if(CUDAQ_REALTIME_INCLUDE_DIR AND TARGET cudaq-qec-realtime-device-call)
  add_executable(test_pymatching_device_call_realtime)
  set_target_properties(test_pymatching_device_call_realtime PROPERTIES
    LINKER_LANGUAGE CXX)

  target_include_directories(test_pymatching_device_call_realtime PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
  )

  cudaqx_add_device_code(test_pymatching_device_call_realtime
    SOURCES
      test_pymatching_device_call_realtime.cpp
    COMPILER_FLAGS
      --target stim
      -frealtime-lowering
  )

  target_link_directories(test_pymatching_device_call_realtime
    PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

  # Link the realtime-lowered device wrappers (NOT cudaq-qec-realtime-decoding-
  # simulation). The plain -simulation library compiles its device_calls without
  # -frealtime-lowering and ships direct host trampolines, which would bypass the
  # host-dispatch service. The -simulation-cqr variant lowers the
  # device_calls to the realtime frame ABI so they dispatch through the
  # host-dispatch channel to the cudaq-qec-realtime-device-call service.
  target_link_libraries(test_pymatching_device_call_realtime PRIVATE
    GTest::gtest_main
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation-cqr
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-stim
  )

  set_target_properties(test_pymatching_device_call_realtime PROPERTIES
    BUILD_RPATH
      "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins;${CUDAQ_INSTALL_DIR}/lib"
  )

  add_dependencies(test_pymatching_device_call_realtime cudaq-qec-pymatching)
  add_dependencies(CUDAQXQECUnitTests test_pymatching_device_call_realtime)
  # CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch routes the device_call through the
  # host-dispatch service; CUDAQ_QEC_REALTIME_MODE=inproc_rpc then routes the
  # decode across the cudaq-realtime session ring. The two vars are joined with
  # an ESCAPED semicolon (\;) so they stay a single ENVIRONMENT value (a list of
  # two var=value entries); an unescaped ';' is split by gtest_discover_tests and
  # set_tests_properties keeps only the first, silently dropping inproc_rpc
  # (caught by the HOP2 self-check in the test).
  gtest_discover_tests(test_pymatching_device_call_realtime
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES ENVIRONMENT
      "CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch\;CUDAQ_QEC_REALTIME_MODE=inproc_rpc")
endif()
