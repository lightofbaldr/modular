# Stage-3 spike: issue ONE native block-scaled FP4 tensor-core MMA
# (mma.sync.aligned.m16n8k64 kind::mxf4nvf4) on consumer Blackwell (sm_121).
# All-1.0 inputs (FP4 0x2=1.0 packed as 0x22222222; e4m3 1.0=0x38) with K=64
# => every C element must equal 64.0, independent of fragment layout. This
# proves the FP4 tensor-core instruction executes end-to-end through Mojo.

from std.gpu.host import DeviceContext
from std.gpu import thread_idx
from std.sys._assembly import inlined_assembly
from std.sys import _RegisterPackType
from std.memory import UnsafePointer


def fp4_mma_tile(c_out: UnsafePointer[Float32, MutAnyOrigin]):
    var a = UInt32(0x22222222)  # 8 packed e2m1 values of 1.0
    var b = UInt32(0x22222222)
    var sc = UInt32(0x38383838)  # 4 packed e4m3 values of 1.0
    var r = inlined_assembly[
        (
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
            ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {$0,$1,$2,$3},"
            " {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, {$14}, {0, 0},"
            " {$15}, {0, 0};"
        ),
        _RegisterPackType[Float32, Float32, Float32, Float32],
        constraints="=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
    ](
        a, a, a, a,
        b, b,
        Float32(0), Float32(0), Float32(0), Float32(0),
        sc, sc,
    )
    var tid = thread_idx.x
    c_out[tid * 4 + 0] = r[0]
    c_out[tid * 4 + 1] = r[1]
    c_out[tid * 4 + 2] = r[2]
    c_out[tid * 4 + 3] = r[3]


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[DType.float32](128)
        var outh = ctx.enqueue_create_host_buffer[DType.float32](128)
        ctx.enqueue_function[fp4_mma_tile](out, grid_dim=1, block_dim=32)
        ctx.synchronize()
        ctx.enqueue_copy(outh, out)
        ctx.synchronize()
        var bad = 0
        for i in range(128):
            if abs(outh[i] - 64.0) > 0.5:
                bad += 1
        print("FP4 MMA tile: C[0]=", outh[0], " bad=", bad, "/128 (expect 64.0)")
        if bad == 0:
            print("FP4 TENSOR-CORE MMA WORKS ON sm_121")
        _ = out^
