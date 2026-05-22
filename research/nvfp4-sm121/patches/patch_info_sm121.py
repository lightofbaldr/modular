p = "mojo/stdlib/std/sys/info.mojo"
s = open(p).read()
assert "_SM_121X_ARCHS" not in s, "already patched"

anchor = '''@always_inline("nodebug")
def _has_sm_120x() -> Bool:
    return _has_nvidia_gpu_any[_SM_120X_ARCHS]()'''

addition = anchor + '''


comptime _SM_121X_ARCHS: List[String] = ["sm_121", "sm_121a"]


@always_inline("nodebug")
def _is_sm_121x() -> Bool:
    return _is_nvidia_gpu_any[_SM_121X_ARCHS]()


@always_inline("nodebug")
def _has_sm_121x() -> Bool:
    return _has_nvidia_gpu_any[_SM_121X_ARCHS]()'''

assert s.count(anchor) == 1, "anchor not unique"
s = s.replace(anchor, addition, 1)

s = s.replace(
    "def _is_sm_120x_or_newer() -> Bool:\n    return _is_sm_120x()",
    "def _is_sm_120x_or_newer() -> Bool:\n    return _is_sm_120x() or _is_sm_121x()",
    1,
)
s = s.replace(
    "def _has_sm_120x_or_newer() -> Bool:\n    return _has_sm_120x()",
    "def _has_sm_120x_or_newer() -> Bool:\n    return _has_sm_120x() or _has_sm_121x()",
    1,
)

open(p, "w").write(s)
print("info.mojo patched with sm_121 arch set + predicates")
