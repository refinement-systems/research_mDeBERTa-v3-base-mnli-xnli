cmake_minimum_required(VERSION 3.28)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(nli LANGUAGES C CXX)

include(GNUInstallDirs)
include(CheckLanguage)
include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

if (APPLE)
    option(NLI_ENABLE_COREML "Prefer CoreML on supported Apple builds" ON)
else()
    option(NLI_ENABLE_COREML "Prefer CoreML on supported Apple builds" OFF)
endif()
option(NLI_BUILD_TESTS "Build nli unit tests" ON)

set(NLI_USE_COREML OFF)
if (APPLE AND NLI_ENABLE_COREML)
    set(NLI_USE_COREML ON)
endif()

if (NLI_USE_COREML)
    check_language(OBJCXX)
    if (CMAKE_OBJCXX_COMPILER)
        enable_language(OBJCXX)
    else()
        message(FATAL_ERROR "CoreML support requires an Objective-C++ compiler.")
    endif()
endif()

set(CMAKE_DISABLE_FIND_PACKAGE_ONNX TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_onnx TRUE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_OBJC OFF CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_OPSCHEMA_LIB OFF CACHE BOOL "" FORCE)
set(onnxruntime_DISABLE_RTTI OFF CACHE BOOL "" FORCE)
set(onnxruntime_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(onnxruntime_ENABLE_TRAINING OFF CACHE BOOL "" FORCE)
set(onnxruntime_USE_COREML ${NLI_USE_COREML} CACHE BOOL "" FORCE)

function(nli_prepare_absl_for_sentencepiece)
    if(NOT TARGET absl::base)
        return()
    endif()

    set(_nli_absl_shim_dir "${CMAKE_BINARY_DIR}/cmake/package-shims/absl")
    file(MAKE_DIRECTORY "${_nli_absl_shim_dir}")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/absl-compat-config.cmake.in"
        "${_nli_absl_shim_dir}/abslConfig.cmake"
        COPYONLY
    )
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/absl-compat-config.cmake.in"
        "${_nli_absl_shim_dir}/absl-config.cmake"
        COPYONLY
    )

    set(absl_DIR "${_nli_absl_shim_dir}" CACHE PATH
        "Path to the local absl compatibility shim used by SentencePiece" FORCE
    )
endfunction()

function(nli_prepare_protobuf_for_sentencepiece)
    if(TARGET protobuf::protoc)
        return()
    endif()

    if(TARGET protoc)
        add_executable(protobuf::protoc ALIAS protoc)
        return()
    endif()

    set(_nli_protoc_path
        "${CMAKE_BINARY_DIR}/_deps/protoc_binary-src/bin/protoc${CMAKE_EXECUTABLE_SUFFIX}"
    )
    if(EXISTS "${_nli_protoc_path}")
        add_executable(protobuf::protoc IMPORTED GLOBAL)
        set_target_properties(protobuf::protoc PROPERTIES
            IMPORTED_LOCATION "${_nli_protoc_path}"
        )
    endif()
endfunction()

set(_nli_saved_skip_install_rules "${CMAKE_SKIP_INSTALL_RULES}")
set(CMAKE_SKIP_INSTALL_RULES TRUE)

add_subdirectory(
    subprojects/onnxruntime/cmake
    ${CMAKE_BINARY_DIR}/subprojects/onnxruntime
    EXCLUDE_FROM_ALL
)

nli_prepare_absl_for_sentencepiece()
nli_prepare_protobuf_for_sentencepiece()

if (DEFINED CMAKE_FIND_PACKAGE_REDIRECTS_DIR AND EXISTS "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}")
    list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}")
endif()

set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(SPM_ABSL_PROVIDER package CACHE STRING "" FORCE)
set(SPM_PROTOBUF_PROVIDER package CACHE STRING "" FORCE)
set(SPM_USE_BUILTIN_PROTOBUF OFF CACHE BOOL "" FORCE)

add_subdirectory(
    subprojects/sentencepiece
    ${CMAKE_BINARY_DIR}/subprojects/sentencepiece
    EXCLUDE_FROM_ALL
)

set(CMAKE_SKIP_INSTALL_RULES "${_nli_saved_skip_install_rules}")

add_library(nli_common
    src/command_line.cpp
    src/nli_inference.cpp
    src/session_factory.cpp
    src/topical_chat.cpp
)
target_include_directories(nli_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/onnxruntime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/onnxruntime/include/onnxruntime/core/session
    ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/sentencepiece/src
)
target_link_libraries(nli_common PUBLIC onnxruntime sentencepiece nlohmann_json::nlohmann_json)

if (NLI_USE_COREML)
    target_compile_definitions(nli_common PUBLIC NLI_HAS_COREML_PROVIDER=1)
endif()

if (UNIX AND NOT APPLE)
    target_link_libraries(nli_common PUBLIC m)
endif()

add_executable(nli
    src/main.cpp
)
target_link_libraries(nli PRIVATE nli_common)

add_executable(nli-topical-chat
    src/topical_chat_main.cpp
)
target_link_libraries(nli-topical-chat PRIVATE nli_common)

if (NLI_BUILD_TESTS)
    enable_testing()

    add_executable(nli_session_factory_test
        tests/session_factory_test.cpp
    )
    target_link_libraries(nli_session_factory_test PRIVATE nli_common)

    add_executable(nli_topical_chat_test
        tests/topical_chat_test.cpp
    )
    target_link_libraries(nli_topical_chat_test PRIVATE nli_common)
    target_compile_definitions(nli_topical_chat_test PRIVATE
        NLI_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(
        NAME session_factory_test
        COMMAND nli_session_factory_test
    )
    add_test(
        NAME topical_chat_test
        COMMAND nli_topical_chat_test
    )

    add_custom_target(nli_tests
        DEPENDS nli_session_factory_test nli_topical_chat_test
    )

    add_custom_target(nli_check
        COMMAND $<TARGET_FILE:nli_session_factory_test>
        COMMAND $<TARGET_FILE:nli_topical_chat_test>
        DEPENDS nli_session_factory_test nli_topical_chat_test
        USES_TERMINAL
    )
endif()

install(TARGETS nli nli-topical-chat
    COMPONENT runtime
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
