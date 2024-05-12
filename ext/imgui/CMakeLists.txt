
set(IMGUI_SOURCES 
    imgui.cpp 
    imgui_demo.cpp
    imgui_draw.cpp 
    imgui_tables.cpp 
    imgui_widgets.cpp
    imgui_impl_opengl3.cpp
    imgui_impl_glfw.cpp
    imgui_spectrum.cpp
)

add_library(imgui STATIC ${IMGUI_SOURCES})

target_link_libraries(imgui glfw)

target_include_directories(imgui PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
