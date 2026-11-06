// Standalone D3D12 adapter/feature probe.
//
// RECONSTRUCTED: the original version of this file was overwritten by accident
// and could not be recovered (the tree is not under version control). What is
// known about it is only its link line - "d3d12 dxgi", no zen_core - so it was a
// raw device probe that printed adapter and feature-level facts without the
// backend's help. This version reproduces that role. If the original had extra
// output, this is not it; treat the fields below as a floor, not a replica.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>

using Microsoft::WRL::ComPtr;

namespace {

const char* feature_level_name(D3D_FEATURE_LEVEL level) {
    switch (level) {
        case D3D_FEATURE_LEVEL_12_2: return "12.2";
        case D3D_FEATURE_LEVEL_12_1: return "12.1";
        case D3D_FEATURE_LEVEL_12_0: return "12.0";
        case D3D_FEATURE_LEVEL_11_1: return "11.1";
        case D3D_FEATURE_LEVEL_11_0: return "11.0";
        default: return "other";
    }
}

}  // namespace

int main() {
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        std::fprintf(stderr, "CreateDXGIFactory1 failed\n");
        return 1;
    }

    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc))) {
            continue;
        }
        // Skip the WARP/basic-render software adapters; they are not testable
        // targets for this backend.
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }
        std::printf("adapter[%u] %ls\n", i, desc.Description);
        std::printf("  vendor=0x%04x device=0x%04x\n", desc.VendorId, desc.DeviceId);
        std::printf("  dedicated_vram=%.1f MB shared=%.1f MB\n",
                    double(desc.DedicatedVideoMemory) / (1024.0 * 1024.0),
                    double(desc.SharedSystemMemory) / (1024.0 * 1024.0));

        const D3D_FEATURE_LEVEL wanted[] = {D3D_FEATURE_LEVEL_12_2, D3D_FEATURE_LEVEL_12_1,
                                             D3D_FEATURE_LEVEL_12_0};
        for (D3D_FEATURE_LEVEL level : wanted) {
            ComPtr<ID3D12Device> device;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), level, IID_PPV_ARGS(&device)))) {
                std::printf("  max_feature_level=%s\n", feature_level_name(level));
                D3D12_FEATURE_DATA_D3D12_OPTIONS opts{};
                if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts,
                                                          sizeof(opts)))) {
                    std::printf("  resource_binding_tier=%d\n", int(opts.ResourceBindingTier));
                }
                break;
            }
        }
    }
    return 0;
}
