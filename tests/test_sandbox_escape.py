"""
test_sandbox_escape.py — SEC-2 verification
Confirms that the combined validate_python() + execute_python() pipeline
blocks all known sandbox escape patterns.

Security model (two layers):
  Layer 1 — validate_python() AST visitor:
    Blocks __dunder__ attribute access, __build_class__, forbidden imports.
    This is where __subclasses__, __class__, __bases__ etc. are caught.

  Layer 2 — execute_python() builtins restriction:
    Exec namespace has no __build_class__, no __import__, no os/sys.
    Catches attempts that somehow bypass the AST (e.g. dynamically built names).

The test verifies the full pipeline for AST-caught patterns, and the exec
layer alone for builtins-caught patterns — matching actual production usage.

ExecutionResult fields (validation.py):
    success: bool
    result: list[dict] | None       (SQL path output)
    dataframe: Any | None           (Pandas path output — pd.DataFrame or scalar)
    error_type: str | None
    error_message: str | None
    row_count: int
    columns: list[str]
"""

import sys

sys.path.insert(0, "analyst_copilot")

import pandas as pd
from validation import execute_python, validate_python

df = pd.DataFrame({"a": [1, 2, 3]})

# ---------------------------------------------------------------------------
# Helper: run the full two-layer pipeline and report blocked/escaped
# ---------------------------------------------------------------------------


def run_full_pipeline(name: str, code: str) -> bool:
    """
    Run validate_python() then execute_python().
    Returns True if the attempt was blocked at either layer.
    """
    schema_cols = {"a"}
    df_names = {"df"}

    # Layer 1: AST check
    vr = validate_python(code, schema_columns=schema_cols, dataframe_refs=df_names)
    if not vr.valid:
        print(f"  PASS (blocked at AST): {name}")
        print(f"    Reason: {vr.error_message[:80]}")
        return True

    # Layer 2: exec builtins restriction
    result = execute_python(code=code, dataframe_refs={"df": df})
    if not result.success:
        print(f"  PASS (blocked at exec): {name}")
        print(f"    Error: {(result.error_message or '')[:80]}")
        return True

    # Escaped both layers
    # Use correct field: 'dataframe' for Pandas path, 'result' for SQL path
    output = result.dataframe if result.dataframe is not None else result.result
    print(f"  FAIL (escaped!): {name}")
    print(f"    Output: {str(output)[:80]}")
    return False


# ---------------------------------------------------------------------------
# Attempts that must be caught by the AST visitor (Layer 1)
# ---------------------------------------------------------------------------
AST_LAYER_ATTEMPTS = [
    # __subclasses__ traversal — caught by visit_Attribute dunder check
    ("subclasses_via_instance", "result = ().__class__.__bases__[0].__subclasses__()"),
    # __class__ access — caught by visit_Attribute dunder check
    ("class_access_via_instance", "t = type\nresult = df.__class__"),
    # __globals__ access — caught by visit_Attribute dunder check
    ("globals_via_func", "def f(): pass\nresult = f.__globals__['__builtins__']"),
    # __reduce__ — caught by visit_Attribute dunder check
    ("reduce_pickle_vector", "import pickle\nresult = df.__reduce__()"),
]

# ---------------------------------------------------------------------------
# Attempts that must be caught by exec builtins restriction (Layer 2)
# Validate_python may or may not catch these; exec must block them regardless.
# ---------------------------------------------------------------------------
EXEC_LAYER_ATTEMPTS = [
    # __build_class__ removed from _SAFE_BUILTINS (SEC-2 fix)
    ("direct_build_class", "__build_class__(lambda: None, 'Exploit')"),
    # class definition — requires __build_class__
    ("class_definition", "class Exploit: pass\nresult = Exploit"),
    # __import__ via builtins dict
    ("import_via_builtins_dict", "result = __builtins__['__import__']('os').getcwd()"),
]

print("=== SEC-2: Sandbox escape attempts (all must be blocked) ===")
print()
print("-- AST-layer attempts (validate_python catches these) --")
all_blocked = True
for name, code in AST_LAYER_ATTEMPTS:
    blocked = run_full_pipeline(name, code)
    all_blocked = all_blocked and blocked

print()
print("-- Exec-layer attempts (execute_python builtins restriction catches these) --")
for name, code in EXEC_LAYER_ATTEMPTS:
    # These bypass validate_python intentionally to test exec-layer isolation
    result = execute_python(code=code, dataframe_refs={"df": df})
    blocked = not result.success
    all_blocked = all_blocked and blocked
    print(f"  {'PASS (blocked)' if blocked else 'FAIL (escaped!)'}: {name}")
    if not blocked:
        output = result.dataframe if result.dataframe is not None else result.result
        print(f"    Output: {str(output)[:80]}")
    else:
        print(f"    Error: {(result.error_message or '')[:80]}")

print()
print(f"{'All escape attempts blocked.' if all_blocked else 'FAILURES — sandbox has gaps!'}")
if not all_blocked:
    sys.exit(1)
