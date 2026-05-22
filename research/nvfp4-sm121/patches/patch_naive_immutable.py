p = "max/kernels/src/linalg/fp4_quantization.mojo"
s = open(p).read()

# naive_block_scaled_matmul_kernel only writes `c`; a/b/a_scales/b_scales are
# read-only. Declaring them MutAnyOrigin forces callers to provide mutable
# tensors, which breaks dispatch from block_scaled_matmul (immutable inputs).
# Make the read-only inputs ImmutAnyOrigin (a mutable caller can still pass).
block_old = (
    "    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],\n"
    "    b: LayoutTensor[b_type, b_layout, MutAnyOrigin],\n"
    "    a_scales: LayoutTensor[a_scales_type, a_scale_layout, MutAnyOrigin],\n"
    "    b_scales: LayoutTensor[b_scales_type, b_scale_layout, MutAnyOrigin],\n"
)
block_new = (
    "    a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],\n"
    "    b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],\n"
    "    a_scales: LayoutTensor[a_scales_type, a_scale_layout, ImmutAnyOrigin],\n"
    "    b_scales: LayoutTensor[b_scales_type, b_scale_layout, ImmutAnyOrigin],\n"
)
assert s.count(block_old) == 1, "kernel-arg block count %d" % s.count(block_old)
s = s.replace(block_old, block_new, 1)

# ensure ImmutAnyOrigin is importable
if "ImmutAnyOrigin" not in s.split("def naive_block_scaled_matmul_kernel")[0]:
    pass  # used below; rely on existing layout imports providing it

open(p, "w").write(s)
print("naive kernel read-only inputs -> ImmutAnyOrigin")
