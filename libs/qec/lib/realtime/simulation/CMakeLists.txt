# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaq-qec-realtime-decoding-simulation SHARED
  simulation_host.cpp
)

target_compile_options(cudaq-qec-realtime-decoding-simulation
  PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

cudaqx_add_device_code(cudaq-qec-realtime-decoding-simulation
  SOURCES
    simulation_device.cpp
)

target_include_directories(cudaq-qec-realtime-decoding-simulation
  PUBLIC 
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# PRIVATE, not PUBLIC: PUBLIC propagates --exclude-libs,ALL to every consumer,
# including test executables that absorb libcudaq-realtime-dispatch.a and
# dlsym() its CUDAQ_REALTIME_DISPATCH_API symbols at runtime.
target_link_options(cudaq-qec-realtime-decoding-simulation PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,ALL>
)

target_link_libraries(cudaq-qec-realtime-decoding-simulation
  PUBLIC 
    cudaqx-core
    cudaq-qec
    cudaq-qec-realtime-decoding
  PRIVATE
    cudaq::cudaq-common
)

set_target_properties(cudaq-qec-realtime-decoding-simulation PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


install(TARGETS cudaq-qec-realtime-decoding-simulation
  COMPONENT qec-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# ---------------------------------------------------------------------------
# Realtime-lowered variant of the simulation device wrappers.
#
# Same sources as cudaq-qec-realtime-decoding-simulation, but the device side is
# compiled with -frealtime-lowering, so each cudaq::device_call is rewritten to
# the realtime frame ABI and dispatched (by fnv1a(function_id)) through the
# host-dispatch channel to the cudaq-qec-realtime-device-call service -- instead
# of resolving to the in-process trampolines (which would bypass the service).
# Consumers that want device_call host-dispatch coverage link THIS library
# rather than ...-simulation. Built only when the device_call service is
# available. The wrappers are target-agnostic, so no --target is needed -- only
# -frealtime-lowering distinguishes this from the plain -simulation library.
# ---------------------------------------------------------------------------
if(CUDAQ_QEC_DEVICE_CALL_SERVICE_AVAILABLE)
  add_library(cudaq-qec-realtime-decoding-simulation-cqr SHARED
    simulation_host.cpp
  )

  target_compile_options(cudaq-qec-realtime-decoding-simulation-cqr
    PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

  cudaqx_add_device_code(cudaq-qec-realtime-decoding-simulation-cqr
    SOURCES
      simulation_device.cpp
    COMPILER_FLAGS
      -frealtime-lowering
  )

  target_include_directories(cudaq-qec-realtime-decoding-simulation-cqr
    PUBLIC
      $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
  )

  target_link_options(cudaq-qec-realtime-decoding-simulation-cqr PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,ALL>
  )

  target_link_libraries(cudaq-qec-realtime-decoding-simulation-cqr
    PUBLIC
      cudaqx-core
      cudaq-qec
      cudaq-qec-realtime-decoding
    PRIVATE
      cudaq::cudaq-common
  )

  set_target_properties(cudaq-qec-realtime-decoding-simulation-cqr PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

  install(TARGETS cudaq-qec-realtime-decoding-simulation-cqr
    COMPONENT qec-lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

