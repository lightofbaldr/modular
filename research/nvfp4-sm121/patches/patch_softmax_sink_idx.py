p = "max/kernels/src/nn/softmax.mojo"
s = open(p).read()

# Per-head sink index bug: rows are laid out [shape[0]=batch*heads, mid..., axis].
# row_idx maps to outer-dim-0 coord = row_idx // (num_rows // shape[0]); the head
# is that coord mod num_heads. The old `row_idx % num_heads` is only correct when
# the middle dims collapse (seq_q == 1, i.e. decode), so prefill rows pick the
# wrong per-head sink.
old = "umod(row_idx, Int(sink_weights.dim[0]()))"
assert s.count(old) == 1, "anchor count %d" % s.count(old)
new = "umod(row_idx // ufloordiv(num_rows, shape[0]), Int(sink_weights.dim[0]()))"
s = s.replace(old, new, 1)

open(p, "w").write(s)
print("softmax.mojo per-head sink index fixed")
