# Standalone NVFP4 naive-block-scaled-matmul smoke test for consumer Blackwell
# (sm_120 / sm_121). Imports ONLY the arch-agnostic naive kernel — deliberately
# avoids any sm100/tcgen05 imports so it can compile for sm_120a/sm_121.
# Goal: prove naive_block_scaled_matmul compiles + launches + produces finite
# output on consumer Blackwell. Correctness-vs-reference comes later.

from std.gpu.host import DeviceContext
from std.math import ceildiv, align_up
from std.random import rand
from layout import TileTensor, Coord, row_major, Idx
from linalg.fp4_quantization import naive_block_scaled_matmul
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    set_scale_factor,
)
from std.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.time import perf_counter_ns


def main() raises:
    comptime M = 256
    comptime N = 256
    comptime K = 256
    comptime a_type = DType.uint8          # packed two FP4-E2M1 per byte
    comptime c_type = DType.bfloat16
    comptime scales_dtype = NVFP4_SF_DTYPE  # float8_e4m3fn block scales
    comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE  # 16

    with DeviceContext() as ctx:
        var a_shape = row_major(Coord(Idx[M](), Idx[K // 2]()))
        var b_shape = row_major(Coord(Idx[N](), Idx[K // 2]()))
        var c_shape = row_major(Coord(Idx[M](), Idx[N]()))

        var a_dev = ctx.enqueue_create_buffer[a_type](M * (K // 2))
        var b_dev = ctx.enqueue_create_buffer[a_type](N * (K // 2))
        var c_dev = ctx.enqueue_create_buffer[c_type](M * N)
        var a = TileTensor(a_dev, a_shape)
        var b = TileTensor(b_dev, b_shape)
        var c = TileTensor(c_dev, c_shape)

        var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](M * (K // 2))
        var b_host_ptr = ctx.enqueue_create_host_buffer[a_type](N * (K // 2))
        var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](M * N)
        var a_host = TileTensor(a_host_ptr, a_shape)
        var b_host = TileTensor(b_host_ptr, b_shape)
        var c_host = TileTensor(c_host_ptr, c_shape)
        # Known-answer inputs: byte 0x22 = two e2m1 values of 1.0 each.
        for i in range(a_host.num_elements()):
            a_host.ptr[i] = UInt8(0x22)
        for i in range(b_host.num_elements()):
            b_host.ptr[i] = UInt8(0x22)

        var a_sc_shape = row_major(
            Coord(
                Idx[ceildiv(M, SF_MN_GROUP_SIZE)](),
                Idx[ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)](),
                Idx[SF_ATOM_M[0]](),
                Idx[SF_ATOM_M[1]](),
                Idx[SF_ATOM_K](),
            )
        )
        var b_sc_shape = row_major(
            Coord(
                Idx[ceildiv(N, SF_MN_GROUP_SIZE)](),
                Idx[ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)](),
                Idx[SF_ATOM_M[0]](),
                Idx[SF_ATOM_M[1]](),
                Idx[SF_ATOM_K](),
            )
        )
        var a_sc_total = a_sc_shape.product()
        var b_sc_total = b_sc_shape.product()

        var a_sc_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](a_sc_total)
        var b_sc_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](b_sc_total)
        var a_sc_host = TileTensor(a_sc_host_ptr, a_sc_shape)
        var b_sc_host = TileTensor(b_sc_host_ptr, b_sc_shape)
        # All block scales = 1.0 (e4m3); per-tensor alpha = 1.0 below.
        for i in range(a_sc_host.num_elements()):
            a_sc_host.ptr[i] = Scalar[scales_dtype](1.0)
        for i in range(b_sc_host.num_elements()):
            b_sc_host.ptr[i] = Scalar[scales_dtype](1.0)

        var a_sc_dev = ctx.enqueue_create_buffer[scales_dtype](a_sc_total)
        var b_sc_dev = ctx.enqueue_create_buffer[scales_dtype](b_sc_total)
        var a_sc = TileTensor(a_sc_dev, a_sc_shape)
        var b_sc = TileTensor(b_sc_dev, b_sc_shape)

        ctx.enqueue_copy(a_dev, a_host_ptr)
        ctx.enqueue_copy(b_dev, b_host_ptr)
        ctx.enqueue_copy(a_sc_dev, a_sc_host_ptr)
        ctx.enqueue_copy(b_sc_dev, b_sc_host_ptr)

        naive_block_scaled_matmul[
            scaling_kind=UMMAKind.KIND_MXF4NVF4,
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=True,
        ](
            c.to_layout_tensor(),
            a.to_layout_tensor(),
            b.to_layout_tensor(),
            a_sc.to_layout_tensor(),
            b_sc.to_layout_tensor(),
            ctx,
            1.0,
        )
        ctx.synchronize()

        ctx.enqueue_copy(c_host_ptr, c_dev)
        ctx.synchronize()

        print("naive NVFP4 GEMM ran on consumer Blackwell")
        comptime expected = Float32(K)  # all-1.0 inputs, unit scales => C = K
        var ok = 0
        var bad = 0
        var first_bad = Float32(0)
        for i in range(c_host.num_elements()):
            var v = Float32(c_host.ptr[i])
            if abs(v - expected) <= 0.5:
                ok += 1
            else:
                if bad == 0:
                    first_bad = v
                bad += 1
        print("correctness: expected", expected, " ok", ok, " bad", bad, " C[0]", Float32(c_host.ptr[0]))
        if bad == 0:
            print("CORRECTNESS PASS")
        else:
            print("CORRECTNESS FAIL first_bad", first_bad)

        # ---- perf (naive CUDA-core reference kernel; tensor-core path is Stage 3) ----
        comptime iters = 200
        for _ in range(10):  # warmup
            naive_block_scaled_matmul[
                scaling_kind=UMMAKind.KIND_MXF4NVF4,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                transpose_b=True,
            ](
                c.to_layout_tensor(), a.to_layout_tensor(), b.to_layout_tensor(),
                a_sc.to_layout_tensor(), b_sc.to_layout_tensor(), ctx, 1.0,
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(iters):
            naive_block_scaled_matmul[
                scaling_kind=UMMAKind.KIND_MXF4NVF4,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                transpose_b=True,
            ](
                c.to_layout_tensor(), a.to_layout_tensor(), b.to_layout_tensor(),
                a_sc.to_layout_tensor(), b_sc.to_layout_tensor(), ctx, 1.0,
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var us = Float64(t1 - t0) / 1000.0 / Float64(iters)
        var gflops = (2.0 * Float64(M) * Float64(N) * Float64(K) / 1.0e9) / (
            us / 1.0e6
        )
        print("perf:", us, "us/call,", gflops, "GFLOP/s  (M=N=K=256, naive ref)")
        _ = a_dev^
        _ = b_dev^
        _ = c_dev^
        _ = a_sc_dev^
        _ = b_sc_dev^
