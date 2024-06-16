@echo off

if not exist build mkdir build

pushd build

cl /EHsc /O2 /std:c++17 ..\main.cpp /link /out:rt.exe

popd

