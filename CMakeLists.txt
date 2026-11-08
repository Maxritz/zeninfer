cmake_minimum_required(VERSION 3.21)
project(ZenInfer VERSION 0.4.5 LANGUAGES CXX)

option(ZEN_BUILD_AVX2 "Build AVX2/FMA3 kernels" ON)
option(ZEN_BUILD_BENCH "Build benchmark executable" ON)
option(ZEN_BUILD_SERVER "Build OpenAI-compatible HTTP server" ON)
option(ZEN_BUILD_TESTS "Build test executables" ON)
option(ZEN_BUILD_D3D12 "Build the Direct3D 12.1 compute backend (Windows/MSVC)" OFF)
option(ZEN_BUILD_HIP "Build the AMD HIP (ROCm) compute backend" OFF)
set(ZEN_D3D12_AGILITY_DIR "H:/llamadx/update/microsoft.direct3d.d3d12.1.722.0-preview" CACHE PATH
    "DirectX Agility SDK package root (enables D3D12_OPTIONS16 / GPU upload heaps)")
set(ZEN_HIP_ROCM_DIR "" CACHE PATH "ROCm install root containing bin/hipcc")
set(ZEN_HIP_ARCH "gfx1201" CACHE STRING "AMDGPU target for the HIP kernels")
set(ZEN_GGUF_FIXTURE_1 "" CACHE FILEPATH "Optional real GGUF fixture for runtime CTest")
set(ZEN_GGUF_FIXTURE_2 "" CACHE FILEPATH "Optional second real GGUF fixture for runtime CTest")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_library(zen_core STATIC src/cpu_features.cpp src/attention.cpp src/compute_pool.cpp src/diagnostics.cpp src/resource_governor.cpp src/model_ir.cpp src/chat_template.cpp src/execution_planner.cpp src/quant.cpp src/gguf.cpp src/tokenizer.cpp kernels/scalar.cpp)
target_include_directories(zen_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(zen_core PUBLIC Threads::Threads)
target_compile_definitions(zen_core PUBLIC ZEN_VERSION="0.4.5-alpha")

if(MSVC)
  target_compile_options(zen_core PRIVATE /O2 /W4 /permissive- /EHsc)
else()
  target_compile_options(zen_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

# --- Direct3D 12.1 compute backend ------------------------------------------
# Shaders compile to DXIL at build time through dxc and are embedded as byte
# arrays, so the executables need neither loose .cso files nor dxcompiler.dll.
if(ZEN_BUILD_D3D12)
  if(NOT WIN32)
    message(WARNING "ZEN_BUILD_D3D12 ignored: Direct3D 12 is Windows-only")
  else()
    find_program(ZEN_DXC_EXECUTABLE NAMES dxc dxc.exe HINTS "${ZEN_DXC_DIR}" "$ENV{ZEN_DXC_DIR}")
    find_package(Python3 COMPONENTS Interpreter)
    if(NOT ZEN_DXC_EXECUTABLE OR NOT Python3_Interpreter_FOUND)
      message(WARNING "ZEN_BUILD_D3D12 disabled: needs dxc.exe (set ZEN_DXC_DIR) and Python 3")
    else()
      set(ZEN_SHADER_OUT "${CMAKE_CURRENT_BINARY_DIR}/generated")
      file(MAKE_DIRECTORY "${ZEN_SHADER_OUT}")
      set(ZEN_D3D12_SHADER_HEADERS)
      
      # q8_0_gemv_dp4a: Shader Model 6.4 for dot4add_i8packed (decode DP4A).
      set(cso_dp4a "${ZEN_SHADER_OUT}/q8_0_gemv_dp4a.cso")
      set(header_dp4a "${ZEN_SHADER_OUT}/q8_0_gemv_dp4a_cso.h")
      add_custom_command(
        OUTPUT "${header_dp4a}"
        COMMAND "${ZEN_DXC_EXECUTABLE}" -T cs_6_4 -E GemvQ8_0Dp4a -O3 -Fo "${cso_dp4a}"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q8_0_gemv_dp4a.hlsl"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bin2header.py"
                "${cso_dp4a}" "${header_dp4a}" kQ8_0GemvDp4aCso
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q8_0_gemv_dp4a.hlsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bin2header.py"
        COMMENT "Compiling D3D12 shader q8_0_gemv_dp4a (cs_6_4)"
        VERBATIM)
      list(APPEND ZEN_D3D12_SHADER_HEADERS "${header_dp4a}")
      
      # q8_0_gemv: the fp32 decode GEMV (parity path). Compiled to kQ8_0GemvCso.
      # It is the only CSO compiled from this source; the int8 DP4A variant is a
      # separate shader (q8_0_gemv_dp4a.hlsl above), never a fallback symbol of this one.
      set(cso_fp32 "${ZEN_SHADER_OUT}/q8_0_gemv_fp32.cso")
      set(header_fp32 "${ZEN_SHADER_OUT}/q8_0_gemv_fp32_cso.h")
      add_custom_command(
        OUTPUT "${header_fp32}"
        COMMAND "${ZEN_DXC_EXECUTABLE}" -T cs_6_0 -E GemvQ8_0 -O3 -Fo "${cso_fp32}"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q8_0_gemv.hlsl"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bin2header.py"
                "${cso_fp32}" "${header_fp32}" kQ8_0GemvCso
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q8_0_gemv.hlsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bin2header.py"
        COMMENT "Compiling D3D12 shader q8_0_gemv (cs_6_0)"
        VERBATIM)
      list(APPEND ZEN_D3D12_SHADER_HEADERS "${header_fp32}")





      set(cso_quant "${ZEN_SHADER_OUT}/quantize_acts.cso")
      set(header_quant "${ZEN_SHADER_OUT}/quantize_acts_cso.h")
      add_custom_command(
        OUTPUT "${header_quant}"
        COMMAND "${ZEN_DXC_EXECUTABLE}" -T cs_6_0 -E QuantizeActs -O3 -Fo "${cso_quant}"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/quantize_acts.hlsl"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bin2header.py"
                "${cso_quant}" "${header_quant}" kQuantizeActsCso
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/quantize_acts.hlsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bin2header.py"
        COMMENT "Compiling D3D12 shader quantize_acts (cs_6_0)"
        VERBATIM)
      list(APPEND ZEN_D3D12_SHADER_HEADERS "${header_quant}")

      # The GEMM uses dot4add_i8packed in its DP4A variant, which needs Shader Model
      # 6.4; every other shader stays at 6.0. Shader model is the fourth field.
      # q8_0_gemm_v3 (BLOCK_T=8) is the speculative-verify tile: same K order
      # as v1 so contents are bit-identical, selected by ZEN_D3D12_GEMM_V3=1.
      # The *_fuseq shaders are producer-fused quantize variants (DP4A decode
      # when ZEN_D3D12_DP4A_FUSEQ=1): each also quantises its output into the
      # qact int8/scale streams (u3/u4), bit-identical to quantize_acts.
      foreach(spec "q8_0_gemm|GemmQ8_0|kQ8_0GemmCso|cs_6_0" "q8_0_gemm_i8|GemmQ8_0I8|kQ8_0GemmI8Cso|cs_6_4" "q8_0_gemm_sh|GemvQ8_0SH|kQ8_0GemmShCso|cs_6_0" "q8_0_gemm_v2|GemmQ8_0V2|kQ8_0GemmV2Cso|cs_6_0" "q8_0_gemm_v3|GemmQ8_0V3|kQ8_0GemmV3Cso|cs_6_0" "q8_0_gemm_t|GemmQ8_0T|kQ8_0GemmTCso|cs_6_0" "q8_0_gemm_rt|GemmQ8_0RT|kQ8_0GemmRtCso|cs_6_0" "q8_0_gemm_fused|GemmQ8_0Fused|kQ8_0GemmFusedCso|cs_6_0" "silu_mul|SiluMul|kSiluMulCso|cs_6_0" "rmsnorm|RmsNorm|kRmsNormCso|cs_6_0" "qk_norm|QkNorm|kQkNormCso|cs_6_0" "addn|Addn|kAddnCso|cs_6_0" "rope|Rope|kRopeCso|cs_6_0" "rope_prefill|RopePrefill|kRopePrefillCso|cs_6_0" "attention|Attention|kAttnCso|cs_6_0" "attention2|Attention2|kAttn2Cso|cs_6_0" "attention3|Attention3|kAttn3Cso|cs_6_0" "flash_attention|FlashAttentionWithCache|kFlashAttnCso|cs_6_0" "flash_attention_tile|FlashAttentionTile|kFlashAttnTileCso|cs_6_0" "flash_attention_keys|FlashAttentionKeys|kFlashAttnKeysCso|cs_6_0" "q8_0_gemm_dp4a|GemmQ8_0Dp4a|kQ8_0GemmDp4aCso|cs_6_4" "q8_0_gemm_q4k|GemmQ4_K|kQ8_0GemmQ4kCso|cs_6_0"
      "q1_0_gemm|GemmQ1_0|kQ1_0GemmCso|cs_6_0" "q6_k_gemm|GemmQ6_K|kQ6_KGemmCso|cs_6_0" "rmsnorm_fuseq|RmsNormFused|kRmsNormFusedCso|cs_6_0" "silu_fuseq|SiluMulFused|kSiluFusedCso|cs_6_0" "attention_fuseq|AttentionFused|kAttnFusedCso|cs_6_0")
        string(REPLACE "|" ";" parts "${spec}")
        list(GET parts 0 shader)
        list(GET parts 1 entry)
        list(GET parts 2 symbol)
        list(GET parts 3 shader_model)
        set(cso "${ZEN_SHADER_OUT}/${shader}.cso")
        set(header "${ZEN_SHADER_OUT}/${shader}_cso.h")
        add_custom_command(
          OUTPUT "${header}"
          COMMAND "${ZEN_DXC_EXECUTABLE}" -T ${shader_model} -E ${entry} -O3 -Fo "${cso}"
                  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.hlsl"
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bin2header.py"
                  "${cso}" "${header}" ${symbol}
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.hlsl"
                  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bin2header.py"
          COMMENT "Compiling D3D12 shader ${shader} (${shader_model})"
          VERBATIM)
        list(APPEND ZEN_D3D12_SHADER_HEADERS "${header}")
      endforeach()

      # Work Graph shaders - DISABLED: requires dxc 1.8.2411+ (Agility 1.722+) with lib_6_8 target
      # Work Graphs are in Shader Model 6.8; dxc 1.8.2411+ supports -enable-work-graph flag
      # set(ZEN_D3D12_WORK_GRAPH_SHADER_HEADERS)
      # foreach(spec "work_graph_nodes|LayerDecodeEntry|kWorkGraphLayerDecodeCso")
      #   ...
      # endforeach()

      target_sources(zen_core PRIVATE
        src/backends/d3d12/d3d12_backend.cpp src/backends/d3d12/agility_export.cpp
        ${ZEN_D3D12_SHADER_HEADERS})
      target_include_directories(zen_core PRIVATE "${ZEN_SHADER_OUT}")

      # Agility SDK: the newer d3d12.h must win over the Windows SDK's copy, and the
      # matching D3D12Core.dll has to sit beside the executables. Without both, the
      # D3D12_OPTIONS16 query below silently answers from the inbox runtime.
      if(ZEN_D3D12_AGILITY_DIR AND EXISTS "${ZEN_D3D12_AGILITY_DIR}/build/native/include/d3d12.h")
        target_include_directories(zen_core BEFORE PRIVATE
          "${ZEN_D3D12_AGILITY_DIR}/build/native/include")
        # Stage the full Agility runtime next to the binaries so every executable
        # loads it instead of falling back to the inbox system D3D12:
        #   D3D12Core.dll                 - the Agility runtime itself
        #   d3d12SDKLayers.dll            - Agility debug/validation layer
        #   D3D12StateObjectCompiler.dll  - state object compiler backing Work Graphs
        # dxcompiler.dll/dxil.dll (dxc_nuget) - runtime HLSL compiler profile support
        set(ZEN_AGILITY_BIN "${ZEN_D3D12_AGILITY_DIR}/build/native/bin/x64")
        set(ZEN_DXC_NUGET "${CMAKE_CURRENT_SOURCE_DIR}/../dxc_nuget/build/native/bin/x64")
        if(NOT EXISTS "${ZEN_DXC_NUGET}/dxcompiler.dll")
          set(ZEN_DXC_NUGET "")
        endif()
        add_custom_target(zen_agility_redist ALL
          COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                  "${ZEN_AGILITY_BIN}/D3D12Core.dll"
                  "${CMAKE_BINARY_DIR}/D3D12Core.dll"
          COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                  "${ZEN_AGILITY_BIN}/d3d12SDKLayers.dll"
                  "${CMAKE_BINARY_DIR}/d3d12SDKLayers.dll"
          COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                  "${ZEN_AGILITY_BIN}/D3D12StateObjectCompiler.dll"
                  "${CMAKE_BINARY_DIR}/D3D12StateObjectCompiler.dll"
          COMMENT "Staging Agility runtime DLLs (D3D12Core, SDKLayers, StateObjectCompiler) next to the binaries")
        if(ZEN_DXC_NUGET)
          add_custom_command(TARGET zen_agility_redist POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${ZEN_DXC_NUGET}/dxcompiler.dll"
                    "${CMAKE_BINARY_DIR}/dxcompiler.dll"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${ZEN_DXC_NUGET}/dxil.dll"
                    "${CMAKE_BINARY_DIR}/dxil.dll"
            COMMENT "  also dxcompiler.dll + dxil.dll from the DXC redistributable")
        endif()
        # DirectStorage: stage its runtime DLLs the same way, when the package is present.
        set(ZEN_DSTORAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../directstorage/native/bin/x64")
        if(NOT EXISTS "${ZEN_DSTORAGE_DIR}/dstorage.dll")
          set(ZEN_DSTORAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../directstorage/native/bin/x64")
        endif()
        if(EXISTS "${ZEN_DSTORAGE_DIR}/dstorage.dll")
          add_custom_command(TARGET zen_agility_redist POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${ZEN_DSTORAGE_DIR}/dstorage.dll"
                    "${CMAKE_BINARY_DIR}/dstorage.dll"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${ZEN_DSTORAGE_DIR}/dstoragecore.dll"
                    "${CMAKE_BINARY_DIR}/dstoragecore.dll"
            COMMENT "  also dstorage.dll + dstoragecore.dll (DirectStorage)")
        else()
          message(STATUS "DirectStorage package not found next to the build tree; staging skipped")
        endif()
        add_dependencies(zen_core zen_agility_redist)
        message(STATUS "DirectX Agility SDK enabled from ${ZEN_D3D12_AGILITY_DIR}")
      else()
        message(WARNING "Agility SDK not found: falling back to the inbox D3D12 runtime, "
                        "GPU upload heaps will report unsupported")
      endif()
      target_compile_definitions(zen_core PUBLIC ZEN_HAS_D3D12=1)
      target_link_libraries(zen_core PUBLIC d3d12 dxgi)
      message(STATUS "Direct3D 12.1 backend enabled (dxc: ${ZEN_DXC_EXECUTABLE})")

      add_executable(zen_d3d12_probe tools/d3d12_probe.cpp)
      target_link_libraries(zen_d3d12_probe PRIVATE d3d12 dxgi)

      add_executable(zen_d3d12_parity tools/d3d12_parity.cpp)
      target_link_libraries(zen_d3d12_parity PRIVATE zen_core)

      add_executable(zen_d3d12_gemm_parity tools/d3d12_gemm_parity.cpp)
      target_link_libraries(zen_d3d12_gemm_parity PRIVATE zen_core)

      add_executable(zen_d3d12_kquant_parity tools/d3d12_kquant_parity.cpp)
      target_link_libraries(zen_d3d12_kquant_parity PRIVATE zen_core)

      # Task 2 of docs/d3d12_full_backend_todo.md exported the residual add, so its
      # harness is built again and now tests the real out = a + b path instead of
      # standing in rms_norm against a reference it could never match.
      add_executable(zen_d3d12_addn_parity tools/d3d12_addn_parity.cpp)
      target_link_libraries(zen_d3d12_addn_parity PRIVATE zen_core)

      # Prefill elementwise GPU path (RMSNorm rows x hidden, SiLU gate * up) vs the
      # CPU scalar references. Fractional-lane shapes exercise the group guards.
      add_executable(zen_d3d12_rms_silu_parity tools/d3d12_rms_silu_parity.cpp)
      target_link_libraries(zen_d3d12_rms_silu_parity PRIVATE zen_core)

      # Task 5: the RoPE kernel was written but never wired, so nothing proved it ran
      # or that it matched the CPU rotation. This harness does both.
      add_executable(zen_d3d12_rope_parity tools/d3d12_rope_parity.cpp)
      target_link_libraries(zen_d3d12_rope_parity PRIVATE zen_core)

      # Tasks 4 and 6: KV cache append + RoPE + GQA attention in one submission. The
      # harness walks positions in order so the cache write path is covered too.
      add_executable(zen_d3d12_attn_parity tools/d3d12_attn_parity.cpp)
      target_link_libraries(zen_d3d12_attn_parity PRIVATE zen_core)

      # FlashAttention prefill: the whole batch in one dispatch, K/V read from the
      # device KV cache. Compares every (token, head) row against a CPU reference,
      # which is what proves the kernel is right and not merely fast.
      add_executable(zen_d3d12_flash_parity tools/d3d12_flash_parity.cpp)
      target_link_libraries(zen_d3d12_flash_parity PRIVATE zen_core)

      # Diagnostic: prints the backend's own counters next to wall-clock time, so
      # "GPU is slow" can be told apart from "host is feeding it badly".
      add_executable(zen_d3d12_metrics tools/d3d12_metrics.cpp)
      target_link_libraries(zen_d3d12_metrics PRIVATE zen_model_adapters)

      # Isolates the vocabulary projection, the largest single weight read per
      # decoded token, to show whether it runs on the GPU or the CPU.
      add_executable(zen_d3d12_lmhead_bench tools/d3d12_lmhead_bench.cpp)
      target_link_libraries(zen_d3d12_lmhead_bench PRIVATE zen_core)

      # Reports where two identically seeded generations diverge.
      add_executable(zen_d3d12_determinism tools/d3d12_determinism.cpp)
      target_link_libraries(zen_d3d12_determinism PRIVATE zen_model_adapters)

      # Isolated Q8_0 GEMM throughput at real prefill shapes, reported in TFLOP/s
      # so kernel changes can be judged without layer orchestration in the way.
      add_executable(zen_d3d12_gemm_bench tools/d3d12_gemm_bench.cpp)
      target_link_libraries(zen_d3d12_gemm_bench PRIVATE zen_core)

      # Batched prefill RoPE CPU-vs-GPU crossover bench (manual, not a ctest gate):
      # measures ms saved vs T for the prefill rope shapes so the T gate
      # (ZEN_D3D12_ROPE_PREFILL_MIN_T) is evidence-backed, not guessed.
      add_executable(zen_d3d12_rope_prefill_bench tools/d3d12_rope_prefill_bench.cpp)
      target_link_libraries(zen_d3d12_rope_prefill_bench PRIVATE zen_core)

      # Temp diagnostic harness: DP4A decode parity (layer_decode dp4a on/off vs
      # CPU double reference). Run manually; not a ctest gate.
      add_executable(zen_d3d12_dp4a_parity tools/d3d12_dp4a_parity.cpp)
      target_link_libraries(zen_d3d12_dp4a_parity PRIVATE zen_core)

      # DML FP16 GEMM probe (standalone): does DirectML hit WMMA speeds on
      # RDNA4 for prefill shapes? Run manually; not a ctest gate.
      # Note: 8.3 short paths (PROGRA~2) - ninja/cmd mangles the long
      # "Program Files (x86)" form's parens/quotes in -I flags.
      add_executable(zen_d3d12_dml_probe tools/d3d12_dml_probe.cpp)
      target_link_libraries(zen_d3d12_dml_probe PRIVATE d3d12 dxgi DirectML)
      target_include_directories(zen_d3d12_dml_probe PRIVATE
        "C:/PROGRA~2/WI3CF2~1/10/Include/10.0.26100.0/um"
        "C:/PROGRA~2/WI3CF2~1/10/Include/10.0.26100.0/shared")

      # Component-level tracer for the Sherlock workflow: separates prefill from
      # decode, prints the per-component table and an instrumentation floor.
      add_executable(zen_sherlock tools/zen_sherlock.cpp)
      target_link_libraries(zen_sherlock PRIVATE zen_model_adapters)
      if(ZEN_BUILD_TESTS)
        # Both harnesses exit 77 when no D3D12.1 device is available, and
        # SKIP_RETURN_CODE turns that into ctest "Skipped" rather than a green PASS,
        # so a machine without the hardware cannot report this suite as verified.
        add_test(NAME zen_d3d12_gemv_parity COMMAND zen_d3d12_parity)
        set_tests_properties(zen_d3d12_gemv_parity PROPERTIES
          LABELS "d3d12;quant" TIMEOUT 120
          ENVIRONMENT "ZEN_BACKEND=d3d12"
          SKIP_RETURN_CODE 77)

        add_test(NAME zen_d3d12_gemm_parity COMMAND zen_d3d12_gemm_parity)
        set_tests_properties(zen_d3d12_gemm_parity PROPERTIES
          LABELS "d3d12;quant" TIMEOUT 120
          ENVIRONMENT "ZEN_BACKEND=d3d12;ZEN_D3D12_GEMM_EXPERIMENTAL=1"
          SKIP_RETURN_CODE 77)

        add_test(NAME zen_d3d12_kquant_parity COMMAND zen_d3d12_kquant_parity)
        set_tests_properties(zen_d3d12_kquant_parity PROPERTIES
          LABELS "d3d12;quant" TIMEOUT 180
          ENVIRONMENT "ZEN_BACKEND=d3d12;ZEN_D3D12_KQUANT=1"
          SKIP_RETURN_CODE 77)

        add_test(NAME zen_d3d12_addn_parity COMMAND zen_d3d12_addn_parity)
        set_tests_properties(zen_d3d12_addn_parity PROPERTIES
          LABELS "d3d12;quant" TIMEOUT 120
          ENVIRONMENT "ZEN_BACKEND=d3d12"
          SKIP_RETURN_CODE 77)

        add_test(NAME zen_d3d12_rope_parity COMMAND zen_d3d12_rope_parity)
        set_tests_properties(zen_d3d12_rope_parity PROPERTIES
          LABELS "d3d12;rope" TIMEOUT 120
          ENVIRONMENT "ZEN_BACKEND=d3d12"
          SKIP_RETURN_CODE 77)

        add_test(NAME zen_d3d12_attn_parity COMMAND zen_d3d12_attn_parity)
        set_tests_properties(zen_d3d12_attn_parity PROPERTIES
          LABELS "d3d12;attention" TIMEOUT 120
          ENVIRONMENT "ZEN_BACKEND=d3d12"
          SKIP_RETURN_CODE 77)

        add_test(NAME zen_d3d12_flash_parity COMMAND zen_d3d12_flash_parity)
        set_tests_properties(zen_d3d12_flash_parity PROPERTIES
          LABELS "d3d12;attention" TIMEOUT 180
          ENVIRONMENT "ZEN_BACKEND=d3d12"
          SKIP_RETURN_CODE 77)

        add_test(NAME zen_d3d12_rms_silu_parity COMMAND zen_d3d12_rms_silu_parity)
        set_tests_properties(zen_d3d12_rms_silu_parity PROPERTIES
          LABELS "d3d12;elementwise" TIMEOUT 120
          ENVIRONMENT "ZEN_BACKEND=d3d12"
          SKIP_RETURN_CODE 77)
      endif()
    endif()
  endif()
endif()

# --- AMD HIP (ROCm) compute backend -----------------------------------------
# The whole backend, host logic and device kernels, is compiled by hipcc into one
# external object. CMake's Windows-Clang guard refuses to mix Clang (which HIP
# requires) with MSVC anywhere else in the same configuration, so hipcc is driven
# as a custom command - the same shape the D3D12 backend uses for dxc - and the
# resulting COFF object is linked into zen_core alongside the MSVC objects.
if(ZEN_BUILD_HIP)
  find_program(ZEN_HIPCC_EXECUTABLE NAMES hipcc hipcc.exe
    HINTS "${ZEN_HIP_ROCM_DIR}/bin" "$ENV{ZEN_HIP_ROCM_DIR}/bin")
  if(NOT ZEN_HIPCC_EXECUTABLE)
    message(WARNING "ZEN_BUILD_HIP disabled: hipcc not found (set ZEN_HIP_ROCM_DIR)")
  else()
    get_filename_component(ZEN_HIP_BIN "${ZEN_HIPCC_EXECUTABLE}" DIRECTORY)
    get_filename_component(ZEN_HIP_ROOT "${ZEN_HIP_BIN}" DIRECTORY)
    set(ZEN_HIP_GEN "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${ZEN_HIP_GEN}")
    set(ZEN_HIP_OBJ "${ZEN_HIP_GEN}/hip_backend.obj")

    add_custom_command(
      OUTPUT "${ZEN_HIP_OBJ}"
      # -fms-runtime-lib=dll selects /MD so the object's RuntimeLibrary guard
      # matches the MSVC translation units in the same archive; hipcc otherwise
      # defaults to the static CRT and the link fails LNK2038.
      COMMAND "${ZEN_HIPCC_EXECUTABLE}" --offload-arch=${ZEN_HIP_ARCH} -O3
              -fms-runtime-lib=dll
              -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
              -c "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/hip/hip_backend.hip"
              -o "${ZEN_HIP_OBJ}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/hip/hip_backend.hip"
              "${CMAKE_CURRENT_SOURCE_DIR}/include/zen/hip_backend.h"
      COMMENT "HIP-compiling hip_backend.hip for ${ZEN_HIP_ARCH}"
      VERBATIM)

    target_sources(zen_core PRIVATE "${ZEN_HIP_OBJ}")
    set_source_files_properties("${ZEN_HIP_OBJ}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    target_compile_definitions(zen_core PUBLIC ZEN_HAS_HIP=1)
    # The MSVC toolchain in this project links the shared CRT, so every target
    # here must agree with the /MD object hipcc now emits.
    set_property(TARGET zen_core PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    target_link_libraries(zen_core PUBLIC "${ZEN_HIP_ROOT}/lib/amdhip64.lib")
    message(STATUS "HIP backend enabled (hipcc: ${ZEN_HIPCC_EXECUTABLE}, arch ${ZEN_HIP_ARCH})")

    add_executable(zen_hip_parity tools/hip_parity.cpp)
    target_link_libraries(zen_hip_parity PRIVATE zen_core)

    add_executable(zen_hip_gemm_parity tools/hip_gemm_parity.cpp)
    target_link_libraries(zen_hip_gemm_parity PRIVATE zen_core)

    if(ZEN_BUILD_TESTS)
      # Both harnesses exit 77 when no HIP device is available, and SKIP_RETURN_CODE
      # turns that into ctest "Skipped" rather than a green PASS, so a machine
      # without the hardware cannot report this suite as verified.
      add_test(NAME zen_hip_gemv_parity COMMAND zen_hip_parity)
      set_tests_properties(zen_hip_gemv_parity PROPERTIES
        LABELS "hip;quant" TIMEOUT 120
        ENVIRONMENT "ZEN_BACKEND=hip"
        SKIP_RETURN_CODE 77)

      add_test(NAME zen_hip_gemm_parity COMMAND zen_hip_gemm_parity)
      set_tests_properties(zen_hip_gemm_parity PROPERTIES
        LABELS "hip;quant" TIMEOUT 120
        ENVIRONMENT "ZEN_BACKEND=hip;ZEN_HIP_GEMM_EXPERIMENTAL=1"
        SKIP_RETURN_CODE 77)
    endif()
  endif()
endif()

add_library(zen_model_adapters STATIC
  src/adapters/dense_transformer_adapter.cpp
  src/adapters/dense_transformer_runtime.cpp)
target_include_directories(zen_model_adapters PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(zen_model_adapters PUBLIC zen_core)
if(MSVC)
  target_compile_options(zen_model_adapters PRIVATE /O2 /W4 /permissive- /EHsc)
else()
  target_compile_options(zen_model_adapters PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

if(ZEN_BUILD_AVX2)
  add_library(zen_avx2 STATIC kernels/avx2.cpp)
  target_include_directories(zen_avx2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(zen_core PUBLIC ZEN_HAS_AVX2_KERNELS=1)
  target_link_libraries(zen_core PUBLIC zen_avx2)
  if(MSVC)
    target_compile_options(zen_avx2 PRIVATE /O2 /W4 /arch:AVX2 /permissive- /EHsc)
  else()
    target_compile_options(zen_avx2 PRIVATE -O3 -mavx2 -mfma -Wall -Wextra -Wpedantic)
  endif()
endif()



if(ZEN_BUILD_BENCH)
  add_executable(zen_cpu_quant_sweep_bench tools/cpu_quant_sweep_bench.cpp)
  target_link_libraries(zen_cpu_quant_sweep_bench PRIVATE zen_core)
  add_executable(zen_kquant_gemv_bench tools/kquant_gemv_bench.cpp)
  target_link_libraries(zen_kquant_gemv_bench PRIVATE zen_core)
  add_executable(zen_bench bench/main.cpp)
  target_link_libraries(zen_bench PRIVATE zen_core)
  if(ZEN_BUILD_AVX2)
    target_link_libraries(zen_bench PRIVATE zen_avx2)
  endif()
  if(MSVC)
    target_compile_options(zen_bench PRIVATE /O2 /W4 /permissive- /EHsc)
  else()
    target_compile_options(zen_bench PRIVATE -O3 -Wall -Wextra -Wpedantic)
  endif()
endif()

add_executable(zen_rms_bench tools/rms_bench.cpp)
target_link_libraries(zen_rms_bench PRIVATE zen_core)
if(ZEN_BUILD_AVX2)
  target_link_libraries(zen_rms_bench PRIVATE zen_avx2)
endif()
if(MSVC)
  target_compile_options(zen_rms_bench PRIVATE /O2 /W4 /permissive- /EHsc)
else()
  target_compile_options(zen_rms_bench PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

add_executable(zen_q8_kblk_bench tools/q8_kblk_bench.cpp)
target_link_libraries(zen_q8_kblk_bench PRIVATE zen_core)
if(ZEN_BUILD_AVX2)
  target_link_libraries(zen_q8_kblk_bench PRIVATE zen_avx2)
endif()
if(MSVC)
  target_compile_options(zen_q8_kblk_bench PRIVATE /O2 /W4 /permissive- /EHsc)
else()
  target_compile_options(zen_q8_kblk_bench PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

add_executable(zen_q8_shapes_bench tools/q8_shapes_bench.cpp)
target_link_libraries(zen_q8_shapes_bench PRIVATE zen_core)
if(ZEN_BUILD_AVX2)
  target_link_libraries(zen_q8_shapes_bench PRIVATE zen_avx2)
endif()
if(MSVC)
  target_compile_options(zen_q8_shapes_bench PRIVATE /O2 /W4 /permissive- /EHsc)
else()
  target_compile_options(zen_q8_shapes_bench PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

add_executable(zen_nonq8_fast_bench tools/nonq8_fast_bench.cpp)
target_link_libraries(zen_nonq8_fast_bench PRIVATE zen_core)
if(ZEN_BUILD_AVX2)
  target_link_libraries(zen_nonq8_fast_bench PRIVATE zen_avx2)
endif()
if(MSVC)
  target_compile_options(zen_nonq8_fast_bench PRIVATE /O2 /W4 /permissive- /EHsc)
else()
  target_compile_options(zen_nonq8_fast_bench PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

add_executable(zen_model_inspect tools/model_inspect.cpp)
target_link_libraries(zen_model_inspect PRIVATE zen_core)
add_executable(zen_mxfp4_probe tools/mxfp4_probe.cpp)
target_link_libraries(zen_mxfp4_probe PRIVATE zen_core)

add_executable(zen_tensor_check tools/tensor_check.cpp)
target_link_libraries(zen_tensor_check PRIVATE zen_core)
add_executable(zen_tokenizer_export tools/tokenizer_export.cpp)
target_link_libraries(zen_tokenizer_export PRIVATE zen_core)

add_executable(zen_generate_ids tools/generate_ids.cpp)
target_link_libraries(zen_generate_ids PRIVATE zen_model_adapters)
# Loads one model twice in a single process and compares the generated ids: exercises the
# weight-residency invalidation on reload/unload (see tools/reload_check.cpp).
add_executable(zen_reload_check tools/reload_check.cpp)
target_link_libraries(zen_reload_check PRIVATE zen_model_adapters)

add_executable(zen_speculative_timing tools/speculative_timing.cpp)
target_link_libraries(zen_speculative_timing PRIVATE zen_model_adapters)

add_executable(zen_parity_export tools/parity_export.cpp)
target_link_libraries(zen_parity_export PRIVATE zen_core)
add_executable(zen_parity_probe tools/parity_probe.cpp)
target_link_libraries(zen_parity_probe PRIVATE zen_model_adapters)
add_executable(zen_decode_parity_dump tools/decode_parity_dump.cpp)
target_link_libraries(zen_decode_parity_dump PRIVATE zen_model_adapters)

# Deterministic architecture-kernel micro-models. These are not production GGUF runtimes.
add_executable(zen_arch_microbench tools/architecture_microbench.cpp)
if(MSVC)
  target_compile_options(zen_arch_microbench PRIVATE /O2 /W4 /permissive- /EHsc)
else()
  target_compile_options(zen_arch_microbench PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

if(ZEN_BUILD_TESTS)
  enable_testing()
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    set(ZEN_QWEN2_SYNTHETIC "${CMAKE_CURRENT_BINARY_DIR}/qwen2-dense-tiny.gguf")
    add_custom_command(OUTPUT "${ZEN_QWEN2_SYNTHETIC}"
      COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_qwen2_synthetic.py" "${ZEN_QWEN2_SYNTHETIC}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_qwen2_synthetic.py"
      COMMENT "Generating deterministic Qwen2 synthetic fixture" VERBATIM)
    add_custom_target(zen_test_synthetic_fixture ALL DEPENDS "${ZEN_QWEN2_SYNTHETIC}")
  else()
    # Preserve a failing test rather than silently dropping runtime coverage.
    set(ZEN_QWEN2_SYNTHETIC "${CMAKE_CURRENT_SOURCE_DIR}/models/synthetic/qwen2-dense-tiny.gguf")
  endif()
  add_executable(zen_core_tests tests/core_tests.cpp)
  target_link_libraries(zen_core_tests PRIVATE zen_model_adapters)
  add_test(NAME zen_core_tests COMMAND zen_core_tests)
  # Coverage builds with atomic profile counters can be much slower on shared
  # CI runners; retain a bounded timeout without killing valid coverage runs.
  # These are CPU-reference assertions (exact Q8_0 GEMV/GEMV-batch parity), so the
  # D3D12 build's backend auto-opt-in must not route them to the device kernel:
  # pin the backend off, exactly as the d3d12_* parity suites pin it on.
  set_tests_properties(zen_core_tests PROPERTIES TIMEOUT 600 ENVIRONMENT "ZEN_BACKEND=cpu")
  add_executable(zen_q35_incremental_tests tests/q35_incremental_tests.cpp)
  target_link_libraries(zen_q35_incremental_tests PRIVATE zen_model_adapters)
  add_executable(zen_q8_1_layout_tests tests/q8_1_layout_tests.cpp)
  target_link_libraries(zen_q8_1_layout_tests PRIVATE zen_core)
  add_test(NAME zen_q8_1_layout_tests COMMAND zen_q8_1_layout_tests)
  set_tests_properties(zen_q8_1_layout_tests PROPERTIES LABELS "quant;gguf;security;independent" TIMEOUT 120)
  add_executable(zen_new_quant_simd_tests tests/new_quant_simd_tests.cpp)
  target_link_libraries(zen_new_quant_simd_tests PRIVATE zen_core)
  add_test(NAME zen_new_quant_simd_tests COMMAND zen_new_quant_simd_tests)
  set_tests_properties(zen_new_quant_simd_tests PROPERTIES LABELS "quant;independent;cpu;parity" TIMEOUT 120)
  add_executable(zen_new_quant_tests tests/new_quant_tests.cpp)
  target_link_libraries(zen_new_quant_tests PRIVATE zen_core)
  add_test(NAME zen_new_quant_tests COMMAND zen_new_quant_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/new_quant_ggml.bin")
  set_tests_properties(zen_new_quant_tests PROPERTIES LABELS "quant;independent;gguf;cpu" TIMEOUT 120)
  add_executable(zen_cpu_quant_sweep_tests tests/cpu_quant_sweep_tests.cpp)
  target_link_libraries(zen_cpu_quant_sweep_tests PRIVATE zen_core)
  add_test(NAME zen_cpu_quant_sweep_tests COMMAND zen_cpu_quant_sweep_tests)
  # CPU sweep by name and by intent: the D3D12 build must not hand these Q8_0
  # GEMMs to the integer device kernel, which has different (documented) precision.
  set_tests_properties(zen_cpu_quant_sweep_tests PROPERTIES LABELS "quant;parity;cpu" TIMEOUT 120 ENVIRONMENT "ZEN_BACKEND=cpu")
  add_executable(zen_q3k_cpu_tests tests/q3k_cpu_tests.cpp)
  target_link_libraries(zen_q3k_cpu_tests PRIVATE zen_core)
  add_test(NAME zen_q3k_cpu_tests COMMAND zen_q3k_cpu_tests)
  set_tests_properties(zen_q3k_cpu_tests PROPERTIES LABELS "quant;parity;q3_k" TIMEOUT 120)
  add_executable(zen_tq2_cpu_tests tests/tq2_cpu_tests.cpp)
  target_link_libraries(zen_tq2_cpu_tests PRIVATE zen_core)
  add_test(NAME zen_tq2_cpu_tests COMMAND zen_tq2_cpu_tests)
  set_tests_properties(zen_tq2_cpu_tests PROPERTIES LABELS "quant;parity;tq2" TIMEOUT 120)
  add_executable(zen_q5k_gemm_tests tests/q5k_gemm_tests.cpp)
  target_link_libraries(zen_q5k_gemm_tests PRIVATE zen_core)
  add_test(NAME zen_q5k_gemm_tests COMMAND zen_q5k_gemm_tests)
  set_tests_properties(zen_q5k_gemm_tests PROPERTIES LABELS "quant;parity;q5_k" TIMEOUT 120)
  add_executable(zen_q6k_gemm_tests tests/q6k_gemm_tests.cpp)
  target_link_libraries(zen_q6k_gemm_tests PRIVATE zen_core)
  add_test(NAME zen_q6k_gemm_tests COMMAND zen_q6k_gemm_tests)
  set_tests_properties(zen_q6k_gemm_tests PROPERTIES LABELS "quant;parity;q6_k" TIMEOUT 120)
  # Independent differential check: the C++ decoders vs the Python reference
  # (iq_reference.py) over randomly generated Q8_K/TQ/IQ blocks and their GEMV.
  add_executable(zen_iq_check tests/iq_check.cpp)
  target_link_libraries(zen_iq_check PRIVATE zen_core)
  add_test(NAME zen_iq_check COMMAND zen_iq_check "${CMAKE_CURRENT_BINARY_DIR}/iq_ref.bin")
  set_tests_properties(zen_iq_check PROPERTIES LABELS "quant;independent;differential" TIMEOUT 120 DEPENDS zen_iq_ref)
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/iq_ref.bin"
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/iq_reference.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/iq_tables.h"
            "${CMAKE_CURRENT_BINARY_DIR}/iq_ref.bin"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/ptq1_0_prism.bin"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/iq2_xxs_ggml.bin"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/ptq1_0_prism.bin"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/iq2_xxs_ggml.bin"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/iq_reference.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/iq_tables.h"
    COMMENT "Generating independent IQ decoder reference vectors (Python)")
  add_custom_target(zen_iq_ref DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/iq_ref.bin")
  add_dependencies(zen_iq_check zen_iq_ref)
  # Standalone structured GGUF fuzz harness: 4096 LCG cases plus the
  # security-review regression corpus. No crash/hang may occur.
  add_executable(zen_fuzz_gguf tests/fuzz_gguf.cpp)
  target_compile_definitions(zen_fuzz_gguf PRIVATE ZINFUZZ_STANDALONE=1)
  target_link_libraries(zen_fuzz_gguf PRIVATE zen_core)
  add_test(NAME zen_fuzz_gguf COMMAND zen_fuzz_gguf)
  set_tests_properties(zen_fuzz_gguf PROPERTIES LABELS "security;fuzz;gguf" TIMEOUT 300)
  if(Python3_Interpreter_FOUND)
    add_test(NAME zen_invalid_model COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen_invalid_model.py $<TARGET_FILE:zen_parity_probe>)
    set_tests_properties(zen_invalid_model PROPERTIES LABELS "runtime;security;gguf" TIMEOUT 60)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy" RESULT_VARIABLE ZEN_NUMPY_IMPORT_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(ZEN_NUMPY_IMPORT_RESULT EQUAL 0)
      add_test(NAME zen_q35_incremental COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/q35_incremental.py $<TARGET_FILE:zen_q35_incremental_tests> $<TARGET_FILE:zen_parity_probe>)
      set_tests_properties(zen_q35_incremental PROPERTIES LABELS "runtime;parity;qwen35;cpu" TIMEOUT 300)
      add_test(NAME zen_new_quant_runtime COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/new_quant_runtime.py $<TARGET_FILE:zen_parity_probe>)
      set_tests_properties(zen_new_quant_runtime PROPERTIES LABELS "quant;runtime;gguf;cpu" TIMEOUT 300)
      add_test(NAME zen_qwen2_numpy_parity COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen2_numpy_parity.py $<TARGET_FILE_DIR:zen_parity_probe> ${ZEN_QWEN2_SYNTHETIC} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/numpy_reference.py --prompt-ids 1,11,51,30,25,17,42,99 --length-ladder)
      set_tests_properties(zen_qwen2_numpy_parity PROPERTIES LABELS "parity;independent;numpy;qwen2" TIMEOUT 180)
      # Qwen3.5-hybrid (Bonsai 2) fixture: PTQ1_0 blocks, the prism.hadamard.*
      # contract with folded weights, gated delta net + full attention. Two
      # fixtures are generated on the fly (PTQ1_0 and F32 weights) and the
      # runtime logits are diffed against the independent reference in
      # tests/make_bonsai2_fixture.py.
      add_test(NAME zen_bonsai2_hybrid_parity COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/bonsai2_parity.py $<TARGET_FILE_DIR:zen_parity_probe> --generator ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_bonsai2_fixture.py --numeric)
      set_tests_properties(zen_bonsai2_hybrid_parity PROPERTIES LABELS "parity;independent;numpy;qwen35;bonsai" TIMEOUT 300)
    elseif(NOT ZEN_SKIP_PARITY_GUARD)
      # Previous behaviour silently dropped the independent-parity tests when
      # NumPy was not importable, so a green CTest run did not necessarily mean
      # the parity evidence had actually been produced. Register a guard test
      # that fails loudly instead (configure with -DZEN_SKIP_PARITY_GUARD=ON to
      # opt out on minimal environments).
      message(WARNING "NumPy is not importable by ${Python3_EXECUTABLE}: the qwen2/bonsai2 parity tests would be skipped. The zen_parity_prerequisites guard test will FAIL ctest until NumPy is installed (pip install numpy) or -DZEN_SKIP_PARITY_GUARD=ON is set.")
      add_test(NAME zen_parity_prerequisites
        COMMAND ${CMAKE_COMMAND} -E echo
          "FAIL: qwen2/bonsai2 parity tests require NumPy (pip install numpy); rerun cmake with -DZEN_SKIP_PARITY_GUARD=ON to skip this guard")
      set_tests_properties(zen_parity_prerequisites PROPERTIES LABELS "parity;guard")
    endif()
    # Repository governance guard was previously shipped but never executed by
    # any automated gate; run it as a test so stale status language breaks CI.
    add_test(NAME zen_governance_check COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/.zeninfer-governance-check.py")
    set_tests_properties(zen_governance_check PROPERTIES LABELS "governance;docs" TIMEOUT 60)
  endif()
  add_test(NAME zen_arch_micro_models COMMAND zen_arch_microbench
    "${CMAKE_CURRENT_SOURCE_DIR}/models/synthetic/qwen3-moe-tiny.zenmodel"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/synthetic/bitnet-tiny.zenmodel"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/synthetic/bonsai-ternary-tiny.zenmodel"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/synthetic/qwen2-dense-tiny.zenmodel"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/synthetic/qwen3-dense-tiny.zenmodel")
  set_tests_properties(zen_arch_micro_models PROPERTIES LABELS "architecture;micro-model;correctness" TIMEOUT 60)
  if(ZEN_BUILD_BENCH)
    add_test(NAME zen_bench_parity COMMAND zen_bench 1)
    set_tests_properties(zen_bench_parity PROPERTIES LABELS "numerical;benchmark" TIMEOUT 180 ENVIRONMENT "ZEN_PERF_MIN_SPEEDUP=1.05;ZEN_BENCH_QUICK=1")
  endif()
  add_executable(zen_runtime_tests tests/runtime_tests.cpp)
  target_link_libraries(zen_runtime_tests PRIVATE zen_model_adapters)
  if(TARGET zen_test_synthetic_fixture)
    add_dependencies(zen_runtime_tests zen_test_synthetic_fixture)
  endif()

  add_executable(zen_trace_test tests/zen_trace_test.cpp)
  target_link_libraries(zen_trace_test PRIVATE zen_model_adapters)
  add_test(NAME zen_runtime_qwen2_synthetic COMMAND zen_runtime_tests "${ZEN_QWEN2_SYNTHETIC}")
  set_tests_properties(zen_runtime_qwen2_synthetic PROPERTIES LABELS "runtime;gguf;synthetic;qwen2" TIMEOUT 180)
  # The same model loaded twice in one process must produce the same tokens. ZEN_BACKEND
  # makes the runtime use the D3D12 path when the machine has it, which is where the
  # weight residency lives; without it the check still covers the reload bookkeeping.
  if(TARGET zen_reload_check)
    add_test(NAME zen_reload_same_model
             COMMAND zen_reload_check "${ZEN_QWEN2_SYNTHETIC}" --prompt-ids 1,2,3,4 --max-new 4)
    set_tests_properties(zen_reload_same_model PROPERTIES
      LABELS "runtime;gguf;d3d12;reload" TIMEOUT 180
      ENVIRONMENT "ZEN_BACKEND=d3d12;ZEN_D3D12=1")
  endif()
  if(ZEN_GGUF_FIXTURE_1 AND ZEN_GGUF_FIXTURE_2)
    add_test(NAME zen_runtime_fixture_1 COMMAND zen_runtime_tests "${ZEN_GGUF_FIXTURE_1}" "${ZEN_GGUF_FIXTURE_2}")
    add_test(NAME zen_runtime_fixture_2 COMMAND zen_runtime_tests "${ZEN_GGUF_FIXTURE_2}" "${ZEN_GGUF_FIXTURE_1}")
    set_tests_properties(zen_runtime_fixture_1 zen_runtime_fixture_2 PROPERTIES LABELS "runtime;gguf;speculative" TIMEOUT 180)
  elseif(ZEN_GGUF_FIXTURE_1)
    add_test(NAME zen_runtime_fixture_1 COMMAND zen_runtime_tests "${ZEN_GGUF_FIXTURE_1}")
    set_tests_properties(zen_runtime_fixture_1 PROPERTIES LABELS "runtime;gguf" TIMEOUT 180)
  elseif(ZEN_GGUF_FIXTURE_2)
    add_test(NAME zen_runtime_fixture_2 COMMAND zen_runtime_tests "${ZEN_GGUF_FIXTURE_2}")
    set_tests_properties(zen_runtime_fixture_2 PROPERTIES LABELS "runtime;gguf" TIMEOUT 180)
  endif()
endif()

if(ZEN_BUILD_SERVER)
  add_library(zen_http STATIC src/openai_server.cpp src/model_backend.cpp src/chat_compaction.cpp)
  target_include_directories(zen_http PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(zen_http PUBLIC zen_core zen_model_adapters)
  if(ZEN_BUILD_AVX2)
    target_link_libraries(zen_http PUBLIC zen_avx2)
  endif()
  if(WIN32)
    target_link_libraries(zen_http PUBLIC ws2_32)
  endif()
  add_executable(zen_server server/main.cpp)
  target_link_libraries(zen_server PRIVATE zen_http)
  if(MSVC)
    target_compile_options(zen_http PRIVATE /O2 /W4 /permissive- /EHsc)
    target_compile_options(zen_server PRIVATE /O2 /W4 /permissive- /EHsc)
  else()
    target_compile_options(zen_http PRIVATE -O2 -Wall -Wextra -Wpedantic)
    target_compile_options(zen_server PRIVATE -O2 -Wall -Wextra -Wpedantic)
  endif()
endif()

if(ZEN_BUILD_TESTS AND ZEN_BUILD_SERVER)
  add_executable(zen_chat_compaction_tests tests/chat_compaction_tests.cpp)
  target_link_libraries(zen_chat_compaction_tests PRIVATE zen_http)
  add_test(NAME zen_chat_compaction_tests COMMAND zen_chat_compaction_tests)
  set_tests_properties(zen_chat_compaction_tests PROPERTIES LABELS "chat;compaction" TIMEOUT 30)
  add_executable(zen_mock_server tests/mock_backend_server.cpp)
  target_include_directories(zen_mock_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(zen_mock_server PRIVATE zen_core zen_http)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    if(TARGET zen_test_synthetic_fixture)
      add_dependencies(zen_server zen_test_synthetic_fixture)
      add_test(NAME zen_http_model_smoke COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/http_model_smoke.py $<TARGET_FILE:zen_server> "${ZEN_QWEN2_SYNTHETIC}")
      set_tests_properties(zen_http_model_smoke PROPERTIES LABELS "http;integration;runtime;synthetic" TIMEOUT 60)
    endif()
    add_test(NAME zen_http_api_smoke COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/http_api_smoke.py $<TARGET_FILE:zen_mock_server>)
    set_tests_properties(zen_http_api_smoke PROPERTIES LABELS "http;integration" TIMEOUT 30)
  endif()
  if(MSVC)
    target_compile_options(zen_mock_server PRIVATE /O2 /W4 /permissive- /EHsc)
  else()
    target_compile_options(zen_mock_server PRIVATE -O2 -Wall -Wextra -Wpedantic)
  endif()
endif()
