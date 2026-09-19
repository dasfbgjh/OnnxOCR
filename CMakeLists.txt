cmake_minimum_required(VERSION 3.10)
project(OnnxOCR
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC: 使用UTF-8源码和执行字符集，避免C4819警告
if(MSVC)
    add_compile_options(/utf-8)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime-1.21.0/OnnxRuntimeWrapper.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv-static/OpenCVWrapperConfig.cmake)

# --- OpenCV setup ---
set(OpenCV_STATIC ON)
include(${OpenCV_DIR}/OpenCVConfig.cmake)

# --- ONNX Runtime setup ---
include(${OnnxRuntime_DIR}/OnnxRuntimeConfig.cmake)

# --- Build shared library ---
add_library(OnnxOCR SHARED
    src/common/onnx_session.cpp
    src/common/utils.cpp
    src/common/pre_process.cpp
    src/common/ctc_decode.cpp
    src/text/db_post_process.cpp
    src/text/text_detector.cpp
    src/text/text_recognizer.cpp
    src/text/text_classifier.cpp
    src/text/rapid_orientation.cpp
    src/text/text_system.cpp
    src/plate/license_plate.cpp
    src/layout/layout_analyzer.cpp
    src/layout/doc_layout_analyzer.cpp
    src/layout/doclayout_yolo_analyzer.cpp
    src/table/table_recognizer.cpp
    src/table/table_cls.cpp
    src/table/unet_table_rec.cpp
    src/ocr.cpp
)

target_include_directories(OnnxOCR
    PRIVATE
    ${OnnxRuntime_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/text
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plate
    ${CMAKE_CURRENT_SOURCE_DIR}/src/table
    ${CMAKE_CURRENT_SOURCE_DIR}/src/layout

    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

target_compile_definitions(OnnxOCR PRIVATE OCR_EXPORTS)

target_link_libraries(OnnxOCR
    PRIVATE
    ${OnnxRuntime_LIBS}
    ${OpenCV_LIBS}
)

if(WIN32)
    if(MSVC)
        target_link_libraries(OnnxOCR PRIVATE ws2_32 advapi32 user32)
        set_property(TARGET OnnxOCR PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        target_compile_options(OnnxOCR PRIVATE $<$<CONFIG:Release>:/MT> $<$<CONFIG:Debug>:/MTd>)
    endif()

    # 复制 onnxruntime.dll 到输出目录（仅动态库版本需要）
    if(EXISTS "${OnnxRuntime_DLL}")
        add_custom_command(TARGET OnnxOCR POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OnnxRuntime_DLL}"
            $<TARGET_FILE_DIR:OnnxOCR>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OnnxRuntime_PROVIDERS_SHARED_DLL}"
            $<TARGET_FILE_DIR:OnnxOCR>
            COMMENT "Copying onnxruntime.dll to output directory"
        )
    endif()

    set_target_properties(OnnxOCR PROPERTIES
        OUTPUT_NAME "OnnxOCR"
        SUFFIX ".dll"
    )
else()
    target_link_libraries(OnnxOCR PRIVATE pthread)
endif()

# --- Set output directory ---
set_target_properties(OnnxOCR PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    POSITION_INDEPENDENT_CODE ON
)

# --- Install rules ---
install(TARGETS OnnxOCR
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES lib/ocr.h lib/ocr_dynamic_loader.h lib/markdown_generator.h DESTINATION include)
install(FILES lib/ocr_dynamic_loader.cpp lib/markdown_generator.cpp DESTINATION src)


add_executable(test test/test.cpp)
target_link_libraries(test PRIVATE OnnxOCR)
set_target_properties(test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    POSITION_INDEPENDENT_CODE ON
)
