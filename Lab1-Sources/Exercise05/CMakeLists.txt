cmake_minimum_required (VERSION 2.8.11)

set(EXEC "pi")

add_executable(${EXEC} pi.cpp  ${EMBEDDED_OPENCL_KERNELS})

# Ajoute la dépendence sur les fichiers clh
target_link_libraries(${EXEC} PUBLIC ${OpenCL_LIBRARY})

#add_custom_command(TARGET ${EXEC} POST_BUILD
#                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                       ${CMAKE_CURRENT_SOURCE_DIR}/pi.cl
#                       $<TARGET_FILE_DIR:${EXEC}>
#                   )