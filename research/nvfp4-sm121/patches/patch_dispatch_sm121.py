p = "max/kernels/src/linalg/fp4_quantization.mojo"
s = open(p).read()

# Anchor everything strictly INSIDE `def block_scaled_matmul[` so the
# `var m` / asserts in other functions are never matched.
fn = s.index("\ndef block_scaled_matmul[")

# 2) remove the hard SM100-only assert (matched within this function only)
hard_assert = (
    "    comptime assert (\n"
    "        ctx.default_device_info.compute == B200.compute\n"
    '    ), "This kernel is only supported on SM100"\n\n'
)
ha = s.index(hard_assert, fn)
s = s[:ha] + s[ha + len(hard_assert):]

# 3) wrap the SM100 body (from the first `var m` in this fn to the fn end) and
#    add the sm_121 -> naive branch.
i0 = s.index("    var m = Int(c.dim[0]())\n", fn)
i1 = s.index("\n\n\n@always_inline\ndef quantize_dynamic_block_scaled", i0)
body = s[i0:i1]
indented = "\n".join((("    " + ln) if ln.strip() else ln) for ln in body.split("\n"))

branch = (
    "    comptime if ctx.default_device_info.compute == B200.compute:\n"
    + indented
    + "\n"
    + "    elif ctx.default_device_info.compute >= 12.0:\n"
    + "        # Consumer Blackwell (sm_120 / sm_121, compute 12.x) lacks the SM100\n"
    + "        # warp-specialized FP4 path; use the arch-agnostic CUDA-core naive\n"
    + "        # kernel. (Datacenter Blackwell / B200 is compute 10.0, handled above.)\n"
    + "        comptime assert (\n"
    + "            elementwise_compute_lambda_fn is None\n"
    + '        ), "compute-lambda epilogue not supported on the sm_121 NVFP4 path"\n'
    + "        naive_block_scaled_matmul[\n"
    + "            scaling_kind = UMMAKind.KIND_MXF4NVF4,\n"
    + "            SF_VECTOR_SIZE=SF_VECTOR_SIZE,\n"
    + "            transpose_b=transpose_b,\n"
    + "            elementwise_lambda_fn=elementwise_lambda_fn,\n"
    + "        ](\n"
    + "            c.to_layout_tensor(),\n"
    + "            a.to_layout_tensor(),\n"
    + "            b.to_layout_tensor(),\n"
    + "            a_scales.to_layout_tensor(),\n"
    + "            b_scales.to_layout_tensor(),\n"
    + "            ctx,\n"
    + "            tensor_sf,\n"
    + "        )\n"
    + "    else:\n"
    + "        comptime assert False, (\n"
    + '            "block_scaled_matmul is only supported on SM100 (B200) or"\n'
    + '            " sm_121 (consumer Blackwell / GB10)"\n'
    + "        )"
)

s = s[:i0] + branch + s[i1:]
open(p, "w").write(s)
print("dispatch wired (anchored within block_scaled_matmul)")
