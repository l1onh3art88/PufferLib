emcc -o build/game.html tcg.c -Os -Wall ./raylib/src/libraylib.a -I./raylib/src -L. -L./raylib/src/libraylib.a -sASSERTIONS=2 -gsource-map -s USE_GLFW=3 -sUSE_WEBGL2=1 -s ASYNCIFY -sFILESYSTEM -s FORCE_FILESYSTEM=1 --shell-file ./raylib/src/minshell.html -DPLATFORM_WEB -DGRAPHICS_API_OPENGL_ES3
