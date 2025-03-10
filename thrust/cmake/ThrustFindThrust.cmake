function(_thrust_find_thrust_multiconfig)
  # Check which systems are enabled by multiconfig:
  set(req_systems)
  if (THRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA)
    list(APPEND req_systems CUDA)
  endif()
  if (THRUST_MULTICONFIG_ENABLE_SYSTEM_CPP)
    list(APPEND req_systems CPP)
  endif()
  if (THRUST_MULTICONFIG_ENABLE_SYSTEM_TBB)
    list(APPEND req_systems TBB)
  endif()
  if (THRUST_MULTICONFIG_ENABLE_SYSTEM_OMP)
    list(APPEND req_systems OMP)
  endif()
  if (THRUST_MULTICONFIG_ENABLE_SYSTEM_HPX)
    list(APPEND req_systems HPX)
  endif()

  find_package(Thrust REQUIRED CONFIG
    NO_DEFAULT_PATH # Only check the explicit path in HINTS:
    HINTS "${CCCL_SOURCE_DIR}/lib/cmake/thrust/"
    COMPONENTS ${req_systems}
  )
endfunction()

function(_thrust_find_thrust_singleconfig)
  find_package(Thrust REQUIRED CONFIG
    NO_DEFAULT_PATH # Only check the explicit path in HINTS:
    HINTS "${CCCL_SOURCE_DIR}/lib/cmake/thrust/"
  )
  # Create target now to prepare system found flags:
  thrust_create_target(thrust FROM_OPTIONS ${THRUST_TARGET_FLAGS})
  thrust_debug_target(thrust "${THRUST_VERSION}")
endfunction()

# Build a ${THRUST_TARGETS} list containing target names for all
# requested configurations
function(thrust_find_thrust)
  if (THRUST_ENABLE_MULTICONFIG)
    _thrust_find_thrust_multiconfig()
  else()
    _thrust_find_thrust_singleconfig()
  endif()
endfunction()
