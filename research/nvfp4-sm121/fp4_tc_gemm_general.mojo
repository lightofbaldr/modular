# Stage-3: GENERAL tiled FP4 tensor-core GEMM (warp per 16x8 output tile, K-loop).
# Validated vs host reference on random e2m1 data, unit scales. M=64,N=64,K=256.
# Builds on the proven m16n8k64 fragment layout + K-accumulation.

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx
from std.sys._assembly import inlined_assembly
from std.sys import _RegisterPackType
from std.memory import UnsafePointer
from std.random import random_ui64

comptime M = 64
comptime N = 64
comptime K = 256
comptime KB = K // 2


def fp4_tc_gemm(
    c_out: UnsafePointer[Float32, MutAnyOrigin],
    a_bytes: UnsafePointer[UInt8, MutAnyOrigin],
    b_bytes: UnsafePointer[UInt8, MutAnyOrigin],
):
    var wm = Int(block_idx.y)  # m-tile (16 rows)
    var wn = Int(block_idx.x)  # n-tile (8 cols)
    var lane = Int(thread_idx.x)
    var gid = lane >> 2
    var tid = lane & 3
    var sc = UInt32(0x38383838)
    var mrow = wm * 16
    var ncol = wn * 8

    @parameter
    def ld(p: UnsafePointer[UInt8, MutAnyOrigin], row: Int, k: Int) -> UInt32:
        return (p + (row * KB + (k >> 1))).bitcast[UInt32]()[0]

    var c0 = Float32(0)
    var c1 = Float32(0)
    var c2 = Float32(0)
    var c3 = Float32(0)
    for k0 in range(0, K, 64):
        var a0 = ld(a_bytes, mrow + gid, k0 + tid * 8)
        var a1 = ld(a_bytes, mrow + gid + 8, k0 + tid * 8)
        var a2 = ld(a_bytes, mrow + gid, k0 + tid * 8 + 32)
        var a3 = ld(a_bytes, mrow + gid + 8, k0 + tid * 8 + 32)
        var b0 = ld(b_bytes, ncol + gid, k0 + tid * 8)
        var b1 = ld(b_bytes, ncol + gid, k0 + tid * 8 + 32)
        var r = inlined_assembly[
            (
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
                ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {$0,$1,$2,$3},"
                " {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, {$14}, {0, 0},"
                " {$15}, {0, 0};"
            ),
            _RegisterPackType[Float32, Float32, Float32, Float32],
            constraints="=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
        ](a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sc, sc)
        c0 = r[0]
        c1 = r[1]
        c2 = r[2]
        c3 = r[3]

    c_out[(mrow + gid) * N + ncol + 2 * tid] = c0
    c_out[(mrow + gid) * N + ncol + 2 * tid + 1] = c1
    c_out[(mrow + gid + 8) * N + ncol + 2 * tid] = c2
    c_out[(mrow + gid + 8) * N + ncol + 2 * tid + 1] = c3


def e2m1_decode(nib: Int) -> Float32:
    var mag = SIMD[DType.float32, 8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)[nib & 7]
    return -mag if (nib & 8) != 0 else mag


def main() raises:
    with DeviceContext() as ctx:
        var a_h = ctx.enqueue_create_host_buffer[DType.uint8](M * KB)
        var b_h = ctx.enqueue_create_host_buffer[DType.uint8](N * KB)
        for i in range(M * KB):
            a_h[i] = UInt8(Int(random_ui64(0, 255)))
        for i in range(N * KB):
            b_h[i] = UInt8(Int(random_ui64(0, 255)))

        var cref = ctx.enqueue_create_host_buffer[DType.float32](M * N)
        for m in range(M):
            for n in range(N):
                var acc = Float32(0)
                for kb in range(KB):
                    var ab = Int(a_h[m * KB + kb])
                    var bb = Int(b_h[n * KB + kb])
                    acc += e2m1_decode(ab & 0xF) * e2m1_decode(bb & 0xF)
                    acc += e2m1_decode(ab >> 4) * e2m1_decode(bb >> 4)
                cref[m * N + n] = acc

        var a_d = ctx.enqueue_create_buffer[DType.uint8](M * KB)
        var b_d = ctx.enqueue_create_buffer[DType.uint8](N * KB)
        var c_d = ctx.enqueue_create_buffer[DType.float32](M * N)
        var c_h = ctx.enqueue_create_host_buffer[DType.float32](M * N)
        ctx.enqueue_copy(a_d, a_h)
        ctx.enqueue_copy(b_d, b_h)
        ctx.enqueue_function[fp4_tc_gemm](
            c_d, a_d.unsafe_ptr(), b_d.unsafe_ptr(),
            grid_dim=(N // 8, M // 16), block_dim=32,
        )
        ctx.synchronize()
        ctx.enqueue_copy(c_h, c_d)
        ctx.synchronize()

        var bad = 0
        var maxdiff = Float32(0)
        for i in range(M * N):
            var d = abs(c_h[i] - cref[i])
            if d > maxdiff:
                maxdiff = d
            if d > 0.5:
                bad += 1
        print("GENERAL FP4 TC GEMM", M, "x", N, "x", K, ": bad=", bad, "/", M * N, " maxdiff=", maxdiff)
        if bad == 0:
            print("GENERAL FP4 TENSOR-CORE GEMM CORRECT")
        _ = a_d^
        _ = b_d^
        _ = c_d^
