// Load-time stubs for RCCL 2.30-era symbols that torch 2.11/2.12 (ROCm builds)
// reference eagerly but RCCL 2.27.7 lacks. vLLM tensor-parallel inference never
// calls these (symmetric-memory / device-API / suspend paths); they return
// ncclInternalError(3) so accidental use fails loudly rather than corrupting.
// Build: g++ -O2 -shared -fPIC devapistub.cpp -o libnccl_devapi_stub.so
// If your torch needs more symbols (check with:
//   nm -D --undefined-only libtorch_hip.so | grep nccl), add them here.
extern "C" {
int ncclDevCommCreate(void*, void*, void*)              { return 3; }
int ncclDevCommDestroy(void*, void*)                    { return 3; }
int ncclGetLsaMultimemDevicePointer(void*, void*, void*){ return 3; }
int ncclPutSignal(void*, void*, void*, void*, void*)    { return 3; }
int ncclSignal(void*, void*, void*, void*)              { return 3; }
int ncclWaitSignal(void*, void*, void*)                 { return 3; }
int ncclCommMemStats(void*, void*)                      { return 3; }
int ncclCommResume(void*)                               { return 3; }
int ncclCommSuspend(void*)                              { return 3; }
}
