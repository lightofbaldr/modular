p = "max/kernels/src/linalg/fp4_utils.mojo"
s = open(p).read()
# Target ONLY get_scale_factor (the reader, `-> Scalar`); leave set_scale_factor mutable.
old = (
    "    scales_tensor: LayoutTensor[scales_dtype, scales_layout, MutAnyOrigin],\n"
    "    row_idx: Int,\n"
    "    col_idx: Int,\n"
    ") -> Scalar[scales_dtype]:\n"
)
assert s.count(old) == 1, "count %d" % s.count(old)
new = old.replace("MutAnyOrigin", "ImmutAnyOrigin")
s = s.replace(old, new, 1)
open(p,"w").write(s)
print("get_scale_factor (reader) scales -> ImmutAnyOrigin")
