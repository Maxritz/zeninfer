cmake_minimum_required(VERSION 3.21)
project(ZenInfer VERSION 0.4.5 LANGUAGES CXX)

option(ZEN_BUILD_AVX2 "Build AVX2/FMA3 kernels" ON)
option(ZEN_BUILD_BENCH "Build benchmark executable" ON)
option(ZEN_BUILD_SERVER "Build OpenAI-compatible HTTP server" ON)
option(ZEN_BUILD_TESTS "Build test executables" ON)
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

add_executable(zen_parity_export tools/parity_export.cpp)
target_link_libraries(zen_parity_export PRIVATE zen_core)
add_executable(zen_parity_probe tools/parity_probe.cpp)
target_link_libraries(zen_parity_probe PRIVATE zen_model_adapters)

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
  add_executable(zen_core_tests tests/core_tests.cpp)
  target_link_libraries(zen_core_tests PRIVATE zen_model_adapters)
  add_test(NAME zen_core_tests COMMAND zen_core_tests)
  set_tests_properties(zen_core_tests PROPERTIES TIMEOUT 120)
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
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/iq_reference.py"
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
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy" RESULT_VARIABLE ZEN_NUMPY_IMPORT_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(ZEN_NUMPY_IMPORT_RESULT EQUAL 0)
      add_test(NAME zen_qwen2_numpy_parity COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen2_numpy_parity.py $<TARGET_FILE_DIR:zen_parity_probe> ${CMAKE_CURRENT_SOURCE_DIR}/models/synthetic/qwen2-dense-tiny.gguf ${CMAKE_CURRENT_SOURCE_DIR}/scripts/numpy_reference.py)
      set_tests_properties(zen_qwen2_numpy_parity PROPERTIES LABELS "parity;independent;numpy;qwen2" TIMEOUT 180)
      # Qwen3.5-hybrid (Bonsai 2) fixture: PTQ1_0 blocks, the prism.hadamard.*
      # contract with folded weights, gated delta net + full attention. Two
      # fixtures are generated on the fly (PTQ1_0 and F32 weights) and the
      # runtime logits are diffed against the independent reference in
      # tests/make_bonsai2_fixture.py.
      add_test(NAME zen_bonsai2_hybrid_parity COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/bonsai2_parity.py $<TARGET_FILE_DIR:zen_parity_probe> --generator ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_bonsai2_fixture.py --numeric)
      set_tests_properties(zen_bonsai2_hybrid_parity PROPERTIES LABELS "parity;independent;numpy;qwen35;bonsai" TIMEOUT 300)
    endif()
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
  add_test(NAME zen_runtime_qwen2_synthetic COMMAND zen_runtime_tests "${CMAKE_CURRENT_SOURCE_DIR}/models/synthetic/qwen2-dense-tiny.gguf")
  set_tests_properties(zen_runtime_qwen2_synthetic PROPERTIES LABELS "runtime;gguf;synthetic;qwen2" TIMEOUT 180)
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
  add_library(zen_http STATIC src/openai_server.cpp src/model_backend.cpp)
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
  add_executable(zen_mock_server tests/mock_backend_server.cpp)
  target_link_libraries(zen_mock_server PRIVATE zen_http)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME zen_http_api_smoke COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/http_api_smoke.py $<TARGET_FILE:zen_mock_server>)
    set_tests_properties(zen_http_api_smoke PROPERTIES LABELS "http;integration" TIMEOUT 30)
  endif()
  if(MSVC)
    target_compile_options(zen_mock_server PRIVATE /O2 /W4 /permissive- /EHsc)
  else()
    target_compile_options(zen_mock_server PRIVATE -O2 -Wall -Wextra -Wpedantic)
  endif()
endif()
