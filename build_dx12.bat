@echo off
call "C:\Progra~1\Micros~1\18\Commun~1\VC\Auxiliary\Build\vcvarsall.bat" x64
set CMAKE_C_COMPILER=cl.exe
set CMAKE_CXX_COMPILER=cl.exe
mkdir H:\llamadx\update\zeninfer\build-dx12 2>nul
cd H:\llamadx\update\zeninfer\build-dx12
"C:\Progra~1\Micros~1\18\Commun~1\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -G Ninja -DZEN_BUILD_D3D12=ON -DZEN_BUILD_TESTS=ON -DZEN_DXC_DIR=H:/llamadx/dxc/bin/x64 -DZEN_D3D12_AGILITY_DIR=H:/llamadx/update/microsoft.direct3d.d3d12.1.722.0-preview ..