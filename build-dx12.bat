@echo off
setlocal
rem Append, never prepend: Strawberry ships DLLs that can shadow SDK tools.
set "PATH=%PATH%;C:\Strawberry\c\bin"
rem Pin the SDK: the 10.0.28000.0 kit on this box ships an mt.exe that fails to load.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 10.0.26100.0
if errorlevel 1 exit /b 1
cd /d H:\llamadx\update\zeninfer
cmake -S . -B build-dx12 -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DZEN_BUILD_D3D12=ON -DZEN_DXC_DIR=H:/llamadx/dxc/bin/x64 -DZEN_BUILD_TESTS=ON
if errorlevel 1 exit /b 1
cmake --build build-dx12
exit /b %errorlevel%
