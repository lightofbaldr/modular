p = "mojo/stdlib/std/gpu/host/info.mojo"
s = open(p).read()
# Revert ONLY the DGX Spark codegen target sm_121a -> sm_121 (keep arch-predicate).
repls = [
    ('arch = "sm_121a", ', 'arch = "sm_121", '),
    ('features = "+ptx88,+sm_121a", ', 'features = "+ptx88,+sm_121", '),
    ('tune_cpu = "sm_121a", ', 'tune_cpu = "sm_121", '),
]
for a, b in repls:
    assert s.count(a) == 1, "count %d for %r" % (s.count(a), a)
    s = s.replace(a, b, 1)
open(p, "w").write(s)
print("reverted DGX target to base sm_121")
