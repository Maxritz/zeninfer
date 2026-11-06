@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 10.0.26100.0
if errorlevel 1 exit /b 1
cd /d H:\llamadx\update\zeninfer
set "CMAKE_EXE=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
"%CMAKE_EXE%" -S . -B build-dx12 -G Ninja -DCMAKE_BUILD_TYPE=Release -DZEN_BUILD_D3D12=ON -DZEN_DXC_DIR=H:/llamadx/dxc/bin/x64 -DZEN_BUILD_TESTS=ON
if errorlevel 1 exit /b 1
"%CMAKE_EXE%" --build build-dx12
exit /b %errorlevel%
