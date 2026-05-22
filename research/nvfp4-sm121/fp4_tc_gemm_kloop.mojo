# Stage-3 e2e kernel build, step 1: FP4 tensor-core GEMM with a full K-loop.
# One warp computes a 16x8 output tile, accumulating across K in steps of 64
# using the native block-scaled FP4 MMA (mma.sync m16n8k64 kind::mxf4nvf4).
# All-1.0 inputs + unit scales over K columns => every C element == K (layout-
# insensitive), which proves the multi-chunk tensor-core accumulation loop.

from std.gpu.host import DeviceContext
from std.gpu import thread_idx
from std.sys._assembly import inlined_assembly
from std.sys import _RegisterPackType
from std.memory import UnsafePointer


def fp4_mma_kloop(c_out: UnsafePointer[Float32, MutAnyOrigin], k_chunks: Int):
    # All-1.0 packed FP4 (0x2 e2m1 = 1.0) and e4m3 1.0 scale (0x38).
    var a = UInt32(0x22222222)
    var b = UInt32(0x22222222)
    var sc = UInt32(0x38383838)
    var c0 = Float32(0)
    var c1 = Float32(0)
    var c2 = Float32(0)
    var c3 = Float32(0)
    # Accumulate across K in steps of 64 (each mma is m16n8k64).
    for _ in range(k_chunks):
        var r = inlined_assembly[
            (
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
                ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {$0,$1,$2,$3},"
                " {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, {$14}, {0, 0},"
                " {$15}, {0, 0};"
            ),
            _RegisterPackType[Float32, Float32, Float32, Float32],
            constraints="=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
        ](a, a, a, a, b, b, c0, c1, c2, c3, sc, sc)
        c0 = r[0]
        c1 = r[1]
        c2 = r[2]
        c3 = r[3]
    var tid = thread_idx.x
    c_out[tid * 4 + 0] = c0
    c_out[tid * 4 + 1] = c1
    c_out[tid * 4 + 2] = c2
    c_out[tid * 4 + 3] = c3


def main() raises:
    with DeviceContext() as ctx:
        comptime k_chunks = 4  # K = 64 * 4 = 256
        comptime expected = Float32(64 * k_chunks)  # all-1.0 => C = K
        var out = ctx.enqueue_create_buffer[DType.float32](128)
        var outh = ctx.enqueue_create_host_buffer[DType.float32](128)
        ctx.enqueue_function[fp4_mma_kloop](
            out, k_chunks, grid_dim=1, block_dim=32
        )
        ctx.synchronize()
        ctx.enqueue_copy(outh, out)
        ctx.synchronize()
        var bad = 0
        for i in range(128):
            if abs(outh[i] - expected) > 0.5:
                bad += 1
        print(
            "FP4 K-loop GEMM: K=",
            64 * k_chunks,
            " C[0]=",
            outh[0],
            " expected=",
            expected,
            " bad=",
            bad,
            "/128",
        )
        if bad == 0:
            print("FP4 TENSOR-CORE K-ACCUMULATION WORKS")
        _ = out^
