# Dual AMD R9700 vLLM Serving on Proxmox — with working GPU P2P and DFlash2

A reproducible guide for serving **Qwen3.8-27B at 154–159 tok/s combined single-stream
decode (~210 on json/math), ~4,700 tok/s prefill, and a 922k-token KV cache** on two
AMD Radeon AI PRO R9700s (gfx1201/RDNA4) passed through to a Proxmox VM — including the
four fixes required to make **GPU↔GPU P2P work inside a VM**, which as far as we know
had no public end-to-end recipe before.

Everything here was measured on real hardware in August 2026, and every adopted config
passed output-quality gates (not just throughput runs).

## Results

Two serving profiles, same hardware and P2P plumbing:

- **FP8 + DFlash2** (the original guide): the compose file below.
- **MXFP4-W4A8 + DFlash2-FP8** ([`mxfp4-dflash/`](mxfp4-dflash/), added 2026-08-29):
  **+45% single-stream decode and 4.6× the KV capacity** (922k tokens) on vLLM 0.28,
  via [ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4)'s
  kernels. Full comparison: [benchmarks/RESULTS-MXFP4-DFLASH.md](benchmarks/RESULTS-MXFP4-DFLASH.md).

Measured like-for-like with [BetterBench](https://github.com/GGZ14/BetterBench) 0.4.0
(0.4.0 measures real stream-update gaps — its numbers are NOT comparable with the
0.2.3-era tables in [benchmarks/RESULTS.md](benchmarks/RESULTS.md)):

| metric (BetterBench 0.4.0) | FP8 + DFlash2 | MXFP4-W4A8 + DFlash2-FP8 |
|---|--:|--:|
| combined decode t/s (weighted) | 106.2 | **154–159** |
| best category (json/math) | 145.9 | **~210** |
| aggregate @ 8 / @ 16 streams | 435 / 404 | **465 / 494** |
| cold prefill @ 128k | 3,376 t/s | **4,085 t/s** |
| GPU KV cache | ~202k tokens | **922k tokens** |

The original 0.2.3-era table (naive vs tuned FP8) remains in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Hardware / host

- 2× AMD Radeon AI PRO R9700 32GB (gfx1201, RDNA4), each PCIe 5.0 x16
- AMD EPYC 7543 (Zen3 — matters: its root complex supports PCIe AtomicOps), 64GB+ RAM for the VM
- Proxmox VE 9.2, pve-qemu 11.x, q35 + OVMF VM, `cpu: host`
- Guest: Debian 13 base (any modern distro), but see the **kernel swap** below

## The four things that block P2P in a VM (and their fixes)

GPU↔GPU P2P under VFIO passthrough fails for four independent, stacked reasons. All
four fixes are required. Skipping any one gives you either no P2P or a broken RCCL.

### 1. Both GPUs must share an emulated PCIe switch

Linux `pci_p2pdma` only permits peer DMA between devices under a **common upstream
bridge** (QEMU's emulated Q35 host bridge is not on the kernel's allowlist). Proxmox's
normal `hostpciN` entries put each GPU on its own root port → P2P refused.

Fix: remove the `hostpciN:` lines from the VM config and build a shared switch via raw
QEMU args (see [`vm-config/vm.conf.example`](vm-config/vm.conf.example)):

```
args: -fw_cfg name=opt/ovmf/X-PciMmio64Mb,string=131072
  -device pcie-root-port,id=p2prp,bus=pcie.0,chassis=90,slot=90,x-speed=32,x-width=16
  -device x3130-upstream,id=p2pup,bus=p2prp
  -device xio3130-downstream,id=p2pdn1,bus=p2pup,chassis=91,slot=1
  -device xio3130-downstream,id=p2pdn2,bus=p2pup,chassis=92,slot=2
  -device vfio-pci,host=0000:XX:00.0,bus=p2pdn1,addr=0x0
  -device vfio-pci,host=0000:YY:00.0,bus=p2pdn2,addr=0x0
```

The `X-PciMmio64Mb=131072` fw_cfg gives OVMF a 128GB MMIO window for the two 32GB
BARs (Resizable BAR / Above-4G decoding must be on in host BIOS). Because the GPUs are
now raw `args` devices, Proxmox will NOT auto-bind them to vfio-pci — make it
persistent yourself:

```
# /etc/modprobe.d/vfio.conf on the Proxmox host
options vfio-pci ids=1002:7551 disable_idle_d3=1
softdep amdgpu pre: vfio-pci
```

### 2. amdgpu refuses the emulated switch's Gen1 link

QEMU's `xio3130-downstream` ports advertise PCIe Gen1 x1 and (unlike `pcie-root-port`)
have no `x-speed`/`x-width` properties. The R9700's SMU fails init against that
("Attempt to override pcie params failed!" → `Fatal error during GPU init`).

Fix: force the link caps in the **guest** kernel cmdline:

```
amdgpu.pcie_gen_cap=0x001F001F amdgpu.pcie_lane_cap=0x003F003F
```

(e.g. drop a file in `/etc/default/grub.d/` and `update-grub`.)

### 3. The guest kernel must have `CONFIG_HSA_AMD_P2P=y`

Debian compiles KFD P2P **out** (`CONFIG_HSA_AMD_P2P is not set`), so no amount of
topology fixing helps. Ubuntu ≥ 24.04 kernels have it on. On Debian, the easiest fix is
the [XanMod kernel](https://xanmod.org/) (Debian-native repo, `CONFIG_HSA_AMD_P2P=y`):

```
apt install linux-xanmod-x64v3   # x64v3 for Zen3+
```

Verify after reboot: `torch.cuda.can_device_access_peer(0,1)` → `True`, and the
radiance startup banner shows `P2P access : ENABLED 0↔1 ✓`.

### 4. RCCL crashes without PCIe atomics — rebuild it

Here's the trap: the emulated switch that enables P2P **kills PCIe AtomicOps** (the
switch ports advertise no atomics routing; guest dmesg shows
`amdgpu: PCIE atomic ops is not supported`). Stock RCCL kernels carry a `hostcall`
metadata declaration that makes ROCr **refuse to dispatch them without atomics**:

```
HIP failure 'the operation cannot be performed in the present state'  (rccl enqueue.cc)
```

This is [ROCm/ROCm#6520](https://github.com/rocm/rocm/issues/6520). It is NOT fixed as
of RCCL 2.30.7 / ROCm 10.1 nightlies (we tested). The verified workaround (credit:
[cadamcat/dual-radeon-vllm](https://github.com/cadamcat/dual-radeon-vllm)) is to
rebuild RCCL 2.27.7 with `NDEBUG` forced into the device compile, which strips the
hostcall declaration:

```bash
git clone --depth 1 -b release/rocm-rel-7.1.1.1 https://github.com/ROCm/rccl.git
cd rccl
sed -i '/^project(rccl CXX)/a add_compile_definitions(NDEBUG)' CMakeLists.txt
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGPU_TARGETS=gfx1201 -DAMDGPU_TARGETS=gfx1201 \
      -DCMAKE_PREFIX_PATH=/opt/rocm -DBUILD_TESTS=OFF
ninja -C build          # ~40–85 min, ~18GB RAM peak
```

Build it inside the same image you'll serve with (ABI match). If the image is pruned,
you may need: `mkdir -p /opt/rocm/.info && echo 7.14.0-0 > /opt/rocm/.info/version`,
`apt install cmake libdrm-dev libnuma-dev patchelf`, and the rocm_smi headers from
[ROCm/rocm_smi_lib](https://github.com/ROCm/rocm_smi_lib) copied to
`/opt/rocm/rocm_smi/include`.

Verify the result has **zero** `hidden_hostcall_buffer` entries:

```bash
llvm-objdump --offloading librccl.so.1.0          # extracts device images
llvm-readelf --notes librccl.so.1.0.*gfx1201* | grep -c hidden_hostcall_buffer   # must be 0
```

Newer torch builds eagerly bind RCCL-2.30-era symbols that 2.27.7 lacks. Build the two
tiny stub libraries in [`rccl-stubs/`](rccl-stubs/) and wire them in:

```bash
g++ -O2 -shared -fPIC -std=c++17 dumpstub.cpp    -o libnccldump_stub.so
g++ -O2 -shared -fPIC             devapistub.cpp -o libnccl_devapi_stub.so
cp librccl.so.1.0 librccl-final.so
patchelf --add-needed libnccldump_stub.so     librccl-final.so
patchelf --add-needed libnccl_devapi_stub.so  librccl-final.so
```

Then bind-mount all three over the image's `librccl.so.1` (see the compose file).
Check symbol coverage against *your* torch:
`nm -D --undefined-only libtorch_hip.so | grep nccl` vs the exports — extend
`devapistub.cpp` with any gaps.

## The serving stack

Image: [`magiccodingman/vllm-radiance`](https://github.com/magiccodingman/vllm-radiance)
(vLLM 0.28 fork of [StillDeadcode/vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance),
hand-tuned gfx1201 kernels). Models:

- Target: `Qwen/Qwen3.8-27B-FP8` (or `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` for a
  +42% KV-capacity / multi-agent profile — see Variants below)
- Drafter: [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
  — block-diffusion speculative decoding, ~1.4× decode over MTP

The full compose file is [`docker-compose.yml`](docker-compose.yml). The
hard-won, non-obvious settings:

| setting | value | why |
|---|---|---|
| drafter `attention_backend` | `TRITON_ATTN` | the combination that works; and prefer vLLM ≥ 0.28 native DFlash2 — the early 0.27 backport we first ran corrupted output (periodic ~2-request garbage window every ~14 requests). (ggz14's later 0.27.1 backport measures clean, but native needs no maintenance.) |
| target `--attention-backend` | `R4D` | radiance's hand-written gfx1201 kernels |
| `--max-num-batched-tokens` | `8192` | bigger chunks amortize prefill… |
| `RADIANCE_AR_MAX_KB` | `98304` | …but ONLY with this raised: the P2P one-shot all-reduce silently falls back to RCCL for messages over the default 48MB cap, and an 8192-token chunk sends 84MB. Without this, chunk 8192 is a net loss. |
| `--max-num-seqs` | `8` | 8 vs 4 nearly doubles aggregate throughput, no single-stream cost |
| `--kv-offloading-size` | **omit** | 2 TP workers × 16GB shm arenas don't fit in a 32GB `/dev/shm` → crash-loop. If you want a host KV tier, use ≤8. |
| `RADIANCE_USE_R4D_AR` | `1` (default) | the P2P one-shot all-reduce — the payoff for all the P2P work. **Only with the rebuilt RCCL**; with stock RCCL on this topology, workers die rather than fall back. |

## Operational gotchas (learned the hard way)

1. **Purge the torch compile cache on any RADIANCE\_\*/speculative/chunk-size change.**
   The cache key ignores these; stale graphs crash the engine with
   `IndexError ... copy_misaligned_inputs`. `rm -rf <cache-mount>/vllm/torch_compile_cache`.
2. **Stale `/dev/shm/psm_*` segments** from unclean stops starve shared memory and can
   fail boots. `rm /dev/shm/psm_*` with the container stopped.
3. `RADIANCE_FAST_DRAFT=1` + `RADIANCE_DYNAMIC_DRAFT=0` **crashes at boot** on a dflash
   drafter (w4-packed 1-D weight reaches the stock GEMM). Leave both defaulted.
4. P2P bandwidth through the EPYC fabric is only ~28 GB/s (same as host-staged) when
   the GPUs sit on different physical root complexes — the win is the one-shot
   all-reduce's latency and 6-bit payload compression, not raw bandwidth. Measure,
   don't assume.
5. A dedicated **DHCP reservation** for the VM saves you rediscovering its IP via
   `qm agent <vmid> network-get-interfaces`.
6. **vLLM's startup memory gate compares `util × total` against *current* free VRAM**,
   so ~60 MiB of idle VRAM use (a VM console framebuffer) can make `GPU_UTIL≥0.97`
   permanently unbootable. Boot at 0.95 and size KV with an explicit
   `--kv-cache-memory` pin instead — the pin overrides util and reclaims the 1–3 GiB
   vLLM's own profiler leaves on the table (derivation recipe in
   [`mxfp4-dflash/run_mxfp4_dflash.sh`](mxfp4-dflash/run_mxfp4_dflash.sh)).
7. **On the MXFP4-W4A8+dflash profile**, `--enable-per-request-metrics` +
   `--enable-prompt-tokens-details` cost a periodic **~0.5 s stream stall**
   (update-gap p99 523.8 ms → 34.7 ms without; A/B'd). The FP8 profile ran the same
   flags with clean streaming (p99 56.8 ms), so this is a flags × profile interaction —
   if you enable them, check your update-gap p99 before trusting the stream.
8. Trim `cudagraph_capture_sizes` to `max_num_seqs × (num_speculative_tokens + 1)` —
   the default ladder captures ~1 GiB/GPU of graphs that can never replay.
9. This lineage returns the thinking trace in a `reasoning` field (not
   `reasoning_content`) on both vLLM 0.27 and 0.28 images — point clients at either.

## Variants

- **MXFP4-W4A8 + DFlash2-FP8** ([`mxfp4-dflash/`](mxfp4-dflash/)) — supersedes the old
  MXFP4+MTP variant: it keeps MXFP4's KV capacity (now 922k tokens) AND beats the FP8
  profile's decode by +45% instead of trading it away, by pairing the 4-bit target with
  an **FP8** DFlash2 drafter (that quant choice is a settled result upstream: a 4-bit
  drafter costs more acceptance than it saves in bandwidth) plus ggz14's int2
  draft/verify heads and small-M decode GEMM. AWQ quality recovery is 99–102% of BF16
  (AMD's numbers); our own output gates (GSM8K-paired upstream, cache-hit correctness,
  needle retrieval, dup-8gram scans) all pass.
- **Skip the whole P2P saga**: run the container in an LXC on the Proxmox host instead
  of a VM (host kernel drives the GPUs → native atomics + native P2P, stock RCCL
  works). You give up VM isolation/snapshots; the Proxmox kernel already has
  `CONFIG_HSA_AMD_P2P=y`.

## Credits

- [StillDeadcode/vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance) — the gfx1201 kernel work this all builds on
- [ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4) — the MXFP4-W4A8 + DFlash2-FP8 stack the fast profile ports (int2 draft/verify heads, small-M decode GEMM, GDN merge, libr4d fixes), and [BetterBench](https://github.com/GGZ14/BetterBench)
- [magiccodingman/vllm-radiance](https://github.com/magiccodingman/vllm-radiance) — the vLLM 0.28 fork + MXFP4/W4A8
- [cadamcat/dual-radeon-vllm](https://github.com/cadamcat/dual-radeon-vllm) — root-caused the RCCL hostcall/atomics bug ([ROCm#6520](https://github.com/rocm/rocm/issues/6520)) and published the rebuild recipe
- z-lab / incoai — the DFlash2 drafter checkpoint; [tcclaviger](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8) — its FP8 quant; AMD Quark team — the AWQ-MXFP4 quant

*Setup debugged and documented with Claude Code.*
