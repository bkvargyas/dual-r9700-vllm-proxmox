// Stub for the C++-mangled ncclCommDump(ncclComm*, unordered_map<string,string>&)
// that some torch/ROCm builds bind eagerly but RCCL 2.27.7 does not export.
// Build: g++ -O2 -shared -fPIC -std=c++17 dumpstub.cpp -o libnccldump_stub.so
#include <string>
#include <unordered_map>
struct ncclComm;
typedef int ncclResult_t_compat;
ncclResult_t_compat ncclCommDump(ncclComm*, std::unordered_map<std::string, std::string>&) { return 0; }
