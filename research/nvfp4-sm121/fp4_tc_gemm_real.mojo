# Stage-3: real-data FP4 tensor-core GEMM, single m16n8k64 tile, validated vs a
# host reference. Random e2m1 inputs + unit scales => layout must be correct to
# match. M=16, N=8, K=64. C is f32 (mma D fragment).
#
# Derived m16n8k64 e2m1 fragment layout (from documented m16n8k32 int8, scaled):
#   lane: gid = lane>>2 (0..7), tid = lane&3 (0..3). K split into halves [0:32),[32:64).
#   A (16x64 row-major): a0=A[gid, tid*8:+8]  a1=A[gid+8, tid*8:+8]
#                        a2=A[gid, tid*8+32:+8]  a3=A[gid+8, tid*8+32:+8]
#   B (8x64  row-major NxK): b0=B[gid, tid*8:+8]  b1=B[gid, tid*8+32:+8]
#   C/D (16x8 f32): d0=C[gid,2tid] d1=C[gid,2tid+1] d2=C[gid+8,2tid] d3=C[gid+8,2tid+1]

from std.gpu.host import DeviceContext
from std.gpu import thread_idx
from std.sys._assembly import inlined_assembly
from std.sys import _RegisterPackType
from std.memory import UnsafePointer
from std.random import random_ui64

comptime M = 16
comptime N = 8
comptime K = 64
comptime KB = K // 2  # bytes per row (2 e2m1 per byte)


def fp4_tc_tile(
    c_out: UnsafePointer[Float32, MutAnyOrigin],
    a_bytes: UnsafePointer[UInt8, MutAnyOrigin],
    b_bytes: UnsafePointer[UInt8, MutAnyOrigin],
):
    var lane = Int(thread_idx.x)
    var gid = lane >> 2
    var tid = lane & 3
    var sc = UInt32(0x38383838)  # unit e4m3 scales

    # load 4 bytes (8 e2m1) as a b32 from a packed row at element offset k
    @parameter
    def ld(p: UnsafePointer[UInt8, MutAnyOrigin], row: Int, k: Int) -> UInt32:
        return (p + (row * KB + (k >> 1))).bitcast[UInt32]()[0]

    var a0 = ld(a_bytes, gid, tid * 8)
    var a1 = ld(a_bytes, gid + 8, tid * 8)
    var a2 = ld(a_bytes, gid, tid * 8 + 32)
    var a3 = ld(a_bytes, gid + 8, tid * 8 + 32)
    var b0 = ld(b_bytes, gid, tid * 8)
    var b1 = ld(b_bytes, gid, tid * 8 + 32)

    var r = inlined_assembly[
        (
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
            ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {$0,$1,$2,$3},"
            " {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, {$14}, {0, 0},"
            " {$15}, {0, 0};"
        ),
        _RegisterPackType[Float32, Float32, Float32, Float32],
        constraints="=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
    ](a0, a1, a2, a3, b0, b1, Float32(0), Float32(0), Float32(0), Float32(0), sc, sc)

    c_out[gid * N + 2 * tid] = r[0]
    c_out[gid * N + 2 * tid + 1] = r[1]
    c_out[(gid + 8) * N + 2 * tid] = r[2]
    c_out[(gid + 8) * N + 2 * tid + 1] = r[3]


def e2m1_decode(nib: Int) -> Float32:
    var mag = SIMD[DType.float32, 8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)[nib & 7]
    return -mag if (nib & 8) != 0 else mag


def main() raises:
    with DeviceContext() as ctx:
        var a_h = ctx.enqueue_create_host_buffer[DType.uint8](M * KB)
        var b_h = ctx.enqueue_create_host_buffer[DType.uint8](N * KB)
        # random e2m1 bytes
        for i in range(M * KB):
            a_h[i] = UInt8(Int(random_ui64(0, 255)))
        for i in range(N * KB):
            b_h[i] = UInt8(Int(random_ui64(0, 255)))

        # host reference: decode + GEMM C[m,n] = sum_k A[m,k]*B[n,k]
        var cref = ctx.enqueue_create_host_buffer[DType.float32](M * N)
        for m in range(M):
            for n in range(N):
                var acc = Float32(0)
                for kb in range(KB):
                    var ab = Int(a_h[m * KB + kb])
                    var bb = Int(b_h[n * KB + kb])
                    # low nibble = even k, high nibble = odd k
                    acc += e2m1_decode(ab & 0xF) * e2m1_decode(bb & 0xF)
                    acc += e2m1_decode(ab >> 4) * e2m1_decode(bb >> 4)
                cref[m * N + n] = acc

        var a_d = ctx.enqueue_create_buffer[DType.uint8](M * KB)
        var b_d = ctx.enqueue_create_buffer[DType.uint8](N * KB)
        var c_d = ctx.enqueue_create_buffer[DType.float32](M * N)
        var c_h = ctx.enqueue_create_host_buffer[DType.float32](M * N)
        ctx.enqueue_copy(a_d, a_h)
        ctx.enqueue_copy(b_d, b_h)
        ctx.enqueue_function[fp4_tc_tile](
            c_d, a_d.unsafe_ptr(), b_d.unsafe_ptr(), grid_dim=1, block_dim=32
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
        print("real-data FP4 TC GEMM 16x8x64: bad=", bad, "/", M * N, " maxdiff=", maxdiff)
        print("kernel C[0..4]=", c_h[0], c_h[1], c_h[2], c_h[3])
        print("ref    C[0..4]=", cref[0], cref[1], cref[2], cref[3])
        if bad == 0:
            print("REAL-DATA FP4 TENSOR-CORE GEMM CORRECT")
        _ = a_d^
        _ = b_d^
        _ = c_d^
