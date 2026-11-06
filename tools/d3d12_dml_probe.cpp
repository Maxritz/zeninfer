// DML FP16 GEMM probe (v1 API) — DEAD END on this host/driver (2026-11-06):
// FP16 GEMM compiles (descriptors=4) but CreateCommandRecorder removes the
// device (0x887a0005 DEVICE_REMOVED), with and without staged Agility DLLs.
// Kept as a diagnostic; not a ctest gate. Prefill fast path must be scalar
// tiling (see q8_0_gemm_v2) or a future DML/runtime that supports FP16 GEMM
// on RDNA4. Dense (nullptr) strides are required for valid tensor descs here.
#include <windows.h>
#include <windowsx.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <DirectML.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
bool ok(const char* what, HRESULT hr) {
    if (FAILED(hr)) {
        std::printf("[dml-probe] FAIL %s hr=0x%lx\n", what, (unsigned long)hr);
        return false;
    }
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    const UINT tokens = argc > 1 ? (UINT)std::atoi(argv[1]) : 128;
    const UINT N = 8192, K = 2048;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 5;

    ComPtr<IDXGIFactory6> factory;
    if (!ok("factory", CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) return 1;
    ComPtr<IDXGIAdapter> adapter;
    if (!ok("enum", factory->EnumAdapters(0, &adapter))) return 1; { DXGI_ADAPTER_DESC d{}; adapter->GetDesc(&d); char nb[256]; WideCharToMultiByte(CP_UTF8,0,d.Description,-1,nb,256,nullptr,nullptr); printf("[dml-probe] adapter0 vendor=0x%x vram=%llu MB desc=%s\n", d.VendorId, (unsigned long long)(d.DedicatedVideoMemory/(1024ull*1024ull)), nb); } const UINT nb2=4; ComPtr<IDXGIAdapter> bestad;
    ComPtr<ID3D12Device> device;
    if (!ok("device", D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_2,
                                        IID_PPV_ARGS(&device)))) return 1;
    ComPtr<IDMLDevice> dml_device;
    if (!ok("dml-create",
            DMLCreateDevice1(device.Get(), DML_CREATE_DEVICE_FLAG_NONE, DML_FEATURE_LEVEL_3_0,
                            IID_PPV_ARGS(&dml_device)))) return 1;
    std::printf("[dml-probe] device+dml OK (T=%u N=%u K=%u)\n", tokens, N, K);

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ComPtr<ID3D12CommandQueue> queue;
    if (!ok("queue", device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)))) return 1;
    ComPtr<ID3D12CommandAllocator> alloc;
    if (!ok("alloc", device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                    IID_PPV_ARGS(&alloc)))) return 1;
    ComPtr<ID3D12GraphicsCommandList> list;
    if (!ok("list", device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                              alloc.Get(), nullptr,
                                              IID_PPV_ARGS(&list)))) return 1;
    ComPtr<ID3D12Fence> fence;
    if (!ok("fence", device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                         IID_PPV_ARGS(&fence)))) return 1;
    HANDLE evt = CreateEventExW(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
    if (!evt) { std::printf("[dml-probe] FAIL event\n"); return 1; }

    auto make_buf = [&](SIZE_T bytes, ComPtr<ID3D12Resource>& r) {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = bytes;
        d.Height = 1;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.SampleDesc = {1, 0};
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        HRESULT hr = device->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &d,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&r));
        return ok("committed", hr);
    };
    const SIZE_T abyte = (SIZE_T)tokens * K * 2ull;
    const SIZE_T bbyte = (SIZE_T)N * K * 2ull;
    const SIZE_T cbyte = (SIZE_T)tokens * N * 2ull;   // HALF out for probe (timing)
    ComPtr<ID3D12Resource> a_res, b_res, c_res;
    if (!make_buf(abyte, a_res) || !make_buf(bbyte, b_res) || !make_buf(cbyte, c_res)) return 1;

    const UINT a_sizes[2] = {tokens, K};
    const UINT a_strides[2] = {K * 2u, 2u};
    const UINT b_sizes[2] = {N, K};
    const UINT b_strides[2] = {K * 2u, 2u};
    const UINT c_sizes[2] = {tokens, N};
    const UINT c_strides[2] = {N * 2u, 2u};

    DML_BUFFER_TENSOR_DESC a_desc{DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 2, a_sizes, nullptr, abyte, 0};
    DML_BUFFER_TENSOR_DESC b_desc{DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 2, b_sizes, nullptr, bbyte, 0};
    DML_BUFFER_TENSOR_DESC c_desc{DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 2, c_sizes, nullptr, cbyte, 0};
    DML_TENSOR_DESC a_tens{DML_TENSOR_TYPE_BUFFER, &a_desc};
    DML_TENSOR_DESC b_tens{DML_TENSOR_TYPE_BUFFER, &b_desc};
    DML_TENSOR_DESC c_tens{DML_TENSOR_TYPE_BUFFER, &c_desc};

    DML_GEMM_OPERATOR_DESC gemm{&a_tens, &b_tens, nullptr, &c_tens, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_TRANSPOSE, 1.0f, 0.0f, nullptr};
    DML_OPERATOR_DESC op_desc{DML_OPERATOR_GEMM, &gemm};
    ComPtr<IDMLOperator> op;
    if (!ok("op", dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op)))) return 1;
    ComPtr<IDMLCompiledOperator> compiled;
    if (!ok("compile", dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE, IID_PPV_ARGS(&compiled)))) return 1;
    DML_BINDING_PROPERTIES props = compiled->GetBindingProperties();
    std::printf("[dml-probe] descriptors=%u temp=%llu persistent=%llu\n",
                props.RequiredDescriptorCount,
                (unsigned long long)props.TemporaryResourceSize,
                (unsigned long long)props.PersistentResourceSize);

    // Descriptor heap for bindings.
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = props.RequiredDescriptorCount ? props.RequiredDescriptorCount : 4;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    if (!ok("heap", device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)))) return 1;
    DML_BINDING_TABLE_DESC tdesc{compiled.Get(), heap->GetCPUDescriptorHandleForHeapStart(),
                                 heap->GetGPUDescriptorHandleForHeapStart(),
                                 hd.NumDescriptors};
    ComPtr<IDMLBindingTable> table;
    if (!ok("table", dml_device->CreateBindingTable(&tdesc, IID_PPV_ARGS(&table)))) return 1;
    DML_BUFFER_BINDING ba{a_res.Get(), 0, abyte};
    DML_BUFFER_BINDING bb{b_res.Get(), 0, bbyte};
    DML_BUFFER_BINDING bc{c_res.Get(), 0, cbyte};
    DML_BINDING_DESC inb[2] = {{DML_BINDING_TYPE_BUFFER, &ba}, {DML_BINDING_TYPE_BUFFER, &bb}};
    DML_BINDING_DESC outb[1] = {{DML_BINDING_TYPE_BUFFER, &bc}};
    table->BindInputs(2, inb);
    table->BindOutputs(1, outb);
    // Temp/persistent resources (probe: compile likely needs none; allocate if asked).
    ComPtr<ID3D12Resource> tmp_res, psl_res;
    if (props.TemporaryResourceSize) {
        if (!make_buf((SIZE_T)props.TemporaryResourceSize, tmp_res)) return 1;
        DML_BUFFER_BINDING bt{tmp_res.Get(), 0, (SIZE_T)props.TemporaryResourceSize};
        DML_BINDING_DESC bd{DML_BINDING_TYPE_BUFFER, &bt};
        table->BindTemporaryResource(&bd);
    }
    if (props.PersistentResourceSize) {
        if (!make_buf((SIZE_T)props.PersistentResourceSize, psl_res)) return 1;
        DML_BUFFER_BINDING bp{psl_res.Get(), 0, (SIZE_T)props.PersistentResourceSize};
        DML_BINDING_DESC bd{DML_BINDING_TYPE_BUFFER, &bp};
        table->BindPersistentResource(&bd);
    }
    ComPtr<IDMLCommandRecorder> recorder;
    if (!ok("recorder", dml_device->CreateCommandRecorder(IID_PPV_ARGS(&recorder)))) return 1;

    UINT64 fence_val = 0;
    auto exec_once = [&]() -> HRESULT {
        if (FAILED(alloc->Reset())) return E_FAIL;
        if (FAILED(list->Reset(alloc.Get(), nullptr))) return E_FAIL;
        recorder->RecordDispatch(list.Get(), compiled.Get(), table.Get());
        if (FAILED(list->Close())) return E_FAIL;
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        HRESULT hr = queue->Signal(fence.Get(), ++fence_val);
        if (FAILED(hr)) return hr;
        return fence->SetEventOnCompletion(fence_val, evt);
    };
    if (!ok("warm", exec_once())) return 1;
    if (fence->GetCompletedValue() < fence_val) WaitForSingleObject(evt, INFINITE);
    if (FAILED(device->GetDeviceRemovedReason())) { std::printf("[dml-probe] removed after warm\n"); return 1; }

    double best = 1e9;
    for (int i = 0; i < iters; ++i) {
        const auto t = Clock::now();
        if (!ok("iter", exec_once())) return 1;
        if (fence->GetCompletedValue() < fence_val) WaitForSingleObject(evt, INFINITE);
        const double ms = ms_since(t);
        best = ms < best ? ms : best;
        std::printf("[dml-probe] iter %d %.2f ms\n", i, ms);
    }
    const double flops = 2.0 * double(tokens) * double(N) * double(K);
    std::printf("[dml-probe] T=%u 8192x2048 best=%.2f ms %.2f TFLOP/s (scalar ~3.0)\n",
                tokens, best, flops / (best / 1000.0) / 1e12);
    CloseHandle(evt);
    return 0;
}
