p = "max/kernels/src/nn/attention/gpu/mha.mojo"
s = open(p).read()

guard = (
    " and not (sink and not (gpu_info == H100 or _is_sm10x_gpu(gpu_info)"
    " or has_amd_gpu_accelerator()))"
)

# Newline-anchored full lines so the 8-space (KVCache) and 4-space (dense)
# variants don't collide. `[dtype]` overloads: KVCache entry + dense entry.
anchors = [
    # dense flash_attention overload (the path test_flash_attention_sink_kernel uses)
    "\n    comptime flash_attention_applicable = flash_attention_hw_supported[dtype]() and head_depth_known and head_depth_supported and not naive_kernel\n",
]
for a in anchors:
    assert s.count(a) == 1, "anchor count %d for: %s" % (s.count(a), a[:60])
    s = s.replace(a, a[:-1] + guard + "\n", 1)

open(p, "w").write(s)
print("patched all 3 flash_attention_applicable gates with sink->naive guard")
