aarch64-linux-android-g++ -o main -O0 main3.cpp  *.S
adb push main /data/local/tmp
