cmake_minimum_required(VERSION 3.20)
project(qwfnfer LANGUAGES C CXX)

# qwfnfer -- a purpose-built inference engine for Qwen3.8-Flash-Next (GGUF arch "qwen4exp").
#
# We own the model graph and the memory hierarchy; ggml supplies the quantized
# math kernels. The entire performance thesis of this engine lives in
# qwfn_io / qwfn_expert_cache / qwfn_ple: fetch expert slices at their natural
# 627KB-921KB granularity over io_uring instead of 4KiB demand paging.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Where llama.cpp/ggml lives. Override with -DLLAMA_CPP_ROOT=...
set(LLAMA_CPP_ROOT "" CACHE PATH "llama.cpp source root (built, with patches/ applied)")
set(LLAMA_CPP_BUILD "${LLAMA_CPP_ROOT}/build/bin"   CACHE PATH "llama.cpp built libraries")
set(QWFN_ARCH_FLAGS -march=native)

if(NOT EXISTS "${LLAMA_CPP_ROOT}/ggml/include/ggml.h")
  message(FATAL_ERROR "ggml headers not found under ${LLAMA_CPP_ROOT}. Pass -DLLAMA_CPP_ROOT=<path>.")
endif()

add_library(ggml_base SHARED IMPORTED)
set_target_properties(ggml_base PROPERTIES IMPORTED_LOCATION "${LLAMA_CPP_BUILD}/libggml-base.so.0")
add_library(ggml_all SHARED IMPORTED)
set_target_properties(ggml_all PROPERTIES IMPORTED_LOCATION "${LLAMA_CPP_BUILD}/libggml.so.0")

# libllama: the chat tools borrow its tokenizer (opened vocab_only, so no
# weights are loaded), and llama-nll scores against llama.cpp itself.
add_library(llama_ref SHARED IMPORTED)
set_target_properties(llama_ref PROPERTIES IMPORTED_LOCATION "${LLAMA_CPP_BUILD}/libllama.so.0")

find_package(Threads REQUIRED)

# Rule 10 of docs/CODING_RULES.md: every warning on, in this project's code. Third-party
# headers are SYSTEM includes so their own warnings do not count; scripts/check_rules.sh
# builds with -DQWFN_WERROR=ON. Warning flags do not change the generated code.
option(QWFN_WERROR "Treat warnings in this project's code as errors" OFF)
set(QWFN_WARN -Wall -Wextra -Wpedantic -Wshadow)
if(QWFN_WERROR)
  list(APPEND QWFN_WARN -Werror)
endif()

# liburing is mandatory: it is the whole point of the engine.
find_path(URING_INCLUDE_DIR liburing.h)
find_library(URING_LIBRARY NAMES uring liburing)
if(NOT URING_INCLUDE_DIR OR NOT URING_LIBRARY)
  message(FATAL_ERROR "liburing not found. Install liburing (Arch: pacman -S liburing).")
endif()
message(STATUS "liburing: ${URING_LIBRARY}")

add_library(qwfn_core STATIC
  src/qwfn_model.cpp
  src/qwfn_io.cpp
  src/qwfn_expert_cache.cpp
  src/qwfn_ple.cpp
  src/qwfn_weights.cpp
  src/qwfn_graph.cpp
  src/qwfn_state.cpp
  src/qwfn_engine.cpp
  src/qwfn_prefill.cpp
  src/qwfn_vision.cpp
)
target_include_directories(qwfn_core PUBLIC src)
target_include_directories(qwfn_core SYSTEM PUBLIC
  "${LLAMA_CPP_ROOT}/vendor/stb"
  "${LLAMA_CPP_ROOT}/ggml/include"
  "${LLAMA_CPP_ROOT}/ggml/src"       # ggml-impl.h: the graph struct, for the cached decode graphs' uid
  "${URING_INCLUDE_DIR}"
)
target_link_libraries(qwfn_core PUBLIC ggml_base ggml_all Threads::Threads "${URING_LIBRARY}")
target_compile_options(qwfn_core PRIVATE -O3 ${QWFN_ARCH_FLAGS} ${QWFN_WARN})



add_executable(qwfn-inspect tools/qwfn_inspect.cpp)
target_link_libraries(qwfn-inspect PRIVATE qwfn_core)
target_compile_options(qwfn-inspect PRIVATE -O3 ${QWFN_ARCH_FLAGS} ${QWFN_WARN})

# qwfn_vocab.cpp is deliberately NOT in qwfn_core: only the binaries that
# tokenize carry a libllama dependency.
# cpp-httplib 0.54 is split header/source, so its .cpp joins the target.
add_executable(qwfn-server tools/qwfn_server.cpp src/qwfn_vocab.cpp
               "${LLAMA_CPP_ROOT}/vendor/cpp-httplib/httplib.cpp")
target_include_directories(qwfn-server SYSTEM PRIVATE
  "${LLAMA_CPP_ROOT}/include"
  "${LLAMA_CPP_ROOT}/vendor"
  "${LLAMA_CPP_ROOT}/vendor/cpp-httplib")
target_link_libraries(qwfn-server PRIVATE qwfn_core llama_ref)
target_compile_options(qwfn-server PRIVATE -O2 ${QWFN_ARCH_FLAGS} ${QWFN_WARN})
set_source_files_properties("${LLAMA_CPP_ROOT}/vendor/cpp-httplib/httplib.cpp" PROPERTIES COMPILE_OPTIONS "-w")   # vendored

add_executable(qwfn-chat tools/qwfn_chat.cpp src/qwfn_vocab.cpp)
target_include_directories(qwfn-chat SYSTEM PRIVATE "${LLAMA_CPP_ROOT}/include")
target_link_libraries(qwfn-chat PRIVATE qwfn_core llama_ref)
target_compile_options(qwfn-chat PRIVATE -O3 ${QWFN_ARCH_FLAGS} ${QWFN_WARN})

add_executable(qwfn-tok tools/qwfn_tok.cpp src/qwfn_vocab.cpp)
target_include_directories(qwfn-tok SYSTEM PRIVATE "${LLAMA_CPP_ROOT}/include")
target_link_libraries(qwfn-tok PRIVATE qwfn_core llama_ref)
target_compile_options(qwfn-tok PRIVATE -O2 ${QWFN_ARCH_FLAGS} ${QWFN_WARN})

add_executable(qwfn-gen tools/qwfn_gen.cpp)
target_link_libraries(qwfn-gen PRIVATE qwfn_core)
target_compile_options(qwfn-gen PRIVATE -O3 ${QWFN_ARCH_FLAGS} ${QWFN_WARN})


add_library(ggml_cpu_lib SHARED IMPORTED)
set_target_properties(ggml_cpu_lib PROPERTIES IMPORTED_LOCATION "${LLAMA_CPP_BUILD}/libggml-cpu.so.0")

add_executable(llama-nll tools/llama_nll.cpp)
target_include_directories(llama-nll SYSTEM PRIVATE "${LLAMA_CPP_ROOT}/include" "${LLAMA_CPP_ROOT}/ggml/include")
target_link_libraries(llama-nll PRIVATE llama_ref ggml_base ggml_all ggml_cpu_lib)
target_compile_options(llama-nll PRIVATE ${QWFN_WARN})

add_executable(qwfn-snaptest tools/qwfn_snaptest.cpp)
target_link_libraries(qwfn-snaptest PRIVATE qwfn_core)
target_compile_options(qwfn-snaptest PRIVATE ${QWFN_WARN})
