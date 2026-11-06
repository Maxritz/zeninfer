// DirectX Agility SDK activation.
//
// The D3D12 loader looks for two exported symbols in the *executable* (not in a DLL)
// before it creates a device. If D3D12SDKVersion is present it loads D3D12Core.dll
// from the directory named by D3D12SDKPath instead of using the inbox runtime, which
// is what makes D3D12_OPTIONS16 (GPU upload heaps / ReBAR) and enhanced barriers
// available.
//
// These live in a static library, so the object is only linked into an executable if
// something references it. d3d12_backend.cpp calls agility_version(), which both
// keeps the object alive in every executable that links zen_core and lets the backend
// report which runtime is actually loaded.
extern "C" {

// 722 == Agility SDK 1.722.0-preview, the package provided alongside this project.
__declspec(dllexport) extern const unsigned int D3D12SDKVersion = 722;

// Relative to the executable. D3D12Core.dll is copied next to the binaries at build
// time, so the current directory is the only place it needs to be found.
__declspec(dllexport) extern const char* D3D12SDKPath = ".\\";

}  // extern "C"

namespace zen::d3d12 {

// Pulled in by the backend so the linker cannot discard this translation unit.
unsigned agility_version() noexcept { return D3D12SDKVersion; }

}  // namespace zen::d3d12
