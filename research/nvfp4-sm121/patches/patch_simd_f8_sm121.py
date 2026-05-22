p = "mojo/stdlib/std/builtin/simd.mojo"
s = open(p).read()

# 1) import _is_sm_121x alongside _is_sm_9x_or_newer
imp = "    _is_sm_9x_or_newer,\n"
assert s.count(imp) == 1, "import anchor count %d" % s.count(imp)
s = s.replace(imp, imp + "    _is_sm_121x,\n", 1)

# 2) exclude sm_121 from the two float8->float hardware pop.cast fast-paths;
#    sm_121's pop.cast(f8->f16/f32) lowering is unimplemented, so route it to
#    the pure bit-manipulation fallback (always correct, just not hw-accelerated).
gate = "    comptime if _is_sm_9x_or_newer() and dtype in (\n"
n = s.count(gate)
assert n == 2, "gate anchor count %d (expected 2)" % n
s = s.replace(
    gate,
    "    comptime if _is_sm_9x_or_newer() and not _is_sm_121x() and dtype in (\n",
)

open(p, "w").write(s)
print("simd.mojo: sm_121 routed to bit-manip float8 conversion (%d gates)" % n)
