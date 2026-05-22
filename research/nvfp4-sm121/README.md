# NVFP4 Tensor-Core GEMM on consumer Blackwell (DGX Spark / GB10, sm_121)

A from-scratch, validated **NVFP4 (block-scaled FP4) tensor-core GEMM** written in
Mojo, running on the NVIDIA GB10 (sm_121) via the native warp-level block-scaled
FP4 MMA. Built 2026-05-22 by Light of Baldr (Prime + Adam).

## Status
- **Functionally complete + validated.** `fp4_tc_gemm_nvfp4.mojo`: 64×64×256 with
  random per-block e4m3 scales, **bit-exact** vs a host block-scaled reference
  (maxrel = 0.0, 4096/4096 elements).
- Remaining work is **performance only** (smem staging, multi-tile-per-warp,
  pipelining). The current kernel is one warp per 16×8 output tile — correct,
  not yet peak.
- **Not upstream-mergeable yet**: depends on a CUDA-13.1+ driver (PTX 9.1 JIT for
  the FP4 MMA) and the `sm_121a` target — see patches below. Shared as research.

## The instruction
`mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`
- Confirmed available on sm_121a (ptxas), requires PTX `.version` 9.1+ → CUDA
  13.1+ driver to JIT.

## Fragment layout (m16n8k64, e2m1)
lane: `gid = lane>>2` (0–7), `tid = lane&3` (0–3). K split into halves [0:32),[32:64).
- A (row-major MxK): `a0=A[gid, tid*8:+8]  a1=A[gid+8, tid*8:+8]  a2=A[gid, tid*8+32:+8]  a3=A[gid+8, tid*8+32:+8]`
- B (row-major NxK): `b0=B[gid, tid*8:+8]  b1=B[gid, tid*8+32:+8]` (col = gid)
- C/D (16x8 f32): `d0=C[gid,2tid] d1=C[gid,2tid+1] d2=C[gid+8,2tid] d3=C[gid+8,2tid+1]`
- Packing: byte low-nibble = even k, high-nibble = odd k; aligned u32 loads.

## Scale-factor distribution (PTX ISA mma block-scaling, scale_vec::4X)
- Selectors `{byte-id, thread-id}` all 0.
- SF_A: lanes with `lane%4 ∈ {0,1}` contribute (p=0 → A_scale row gid, p=1 → row gid+8).
- SF_B: lanes with `lane%4 == 0` contribute (B_scale col gid).
- Each contributing lane's scale-data b32 = that row/col's 4 block-scales
  (one e4m3 byte per 16-element block; byte b → block b of the k64 chunk).

## Build prerequisites (sm_121 / GB10)
Stdlib/kernel patches (see `patches/`):
1. `info.mojo`: add sm_121 to arch predicates (`_SM_121X_ARCHS` + `_is_sm_121x`).
2. `info.mojo`: DGX Spark target → `sm_121a` (+ptx91 for the FP4 MMA).
3. `simd.mojo`: route sm_121 float8→float casts to the bit-manip fallback.
4. `fp4_quantization.mojo` + `fp4_utils.mojo`: route block_scaled_matmul to the
   naive path on consumer Blackwell + immutable read-only kernel inputs.
- Driver: CUDA 13.1+ (we ran 595.71.05 / CUDA 13.2). Secure Boot: dkms module
  signing key (MOK) must be enrolled.

## Files (build progression)
- `fp4_mma_spike.mojo` — one m16n8k64 FP4 MMA tile (all-1.0).
- `fp4_tc_gemm_kloop.mojo` — full-K accumulation.
- `fp4_tc_gemm_real.mojo` — single tile, real data, host-ref validated.
- `fp4_tc_gemm_general.mojo` — general tiled GEMM, unit scales.
- `fp4_tc_gemm_nvfp4.mojo` — **complete: per-block e4m3 scales, validated.**
