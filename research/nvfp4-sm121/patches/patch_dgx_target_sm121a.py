p = "mojo/stdlib/std/gpu/host/info.mojo"
s = open(p).read()

# The DGX Spark (GB10, sm_121) codegen target used the base `sm_121` arch, which
# rejects architecture-specific Blackwell instructions (e.g. cvt.f16x2.e2m1x2 for
# FP4). sm_100 and sm_120 both correctly use the `a` (arch-specific) variant.
# Switch sm_121 to sm_121a so PTX emits `.target sm_121a` and the GB10 JIT accepts
# arch-specific instructions.
repls = [
    ('`arch = "sm_121", `,', '`arch = "sm_121a", `,'),
    ('`features = "+ptx88,+sm_121", `,', '`features = "+ptx88,+sm_121a", `,'),
    ('`tune_cpu = "sm_121", `,', '`tune_cpu = "sm_121a", `,'),
]
for old, new in repls:
    assert s.count(old) == 1, "anchor count %d for %s" % (s.count(old), old)
    s = s.replace(old, new, 1)

open(p, "w").write(s)
print("info.mojo: DGX Spark target -> sm_121a (arch-specific)")
