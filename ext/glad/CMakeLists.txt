add_library(glad src/glad.c)

target_include_directories(glad PUBLIC
    $<BUILD_INTERFACE:${OPENGL_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(glad PUBLIC ${OPENGL_LIBRARIES})

if (UNIX AND NOT APPLE)
    target_link_libraries(glad PUBLIC dl)
endif()

