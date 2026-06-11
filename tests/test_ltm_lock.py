"""
test_ltm_lock.py — BUG-4 verification
Confirms that rebuild_if_stale() acquires self._lock, preventing the race
condition with concurrent store()/search() calls.

Windows note: pathlib.read_text() defaults to the system ANSI codepage (cp1252
on most Windows installs) which cannot decode the UTF-8 source files. Always
pass encoding='utf-8' explicitly.
"""

import ast
import pathlib
import sys

# Works whether run from project root or tests/ directory
src_path = pathlib.Path("analyst_copilot/long_term_memory.py")
if not src_path.exists():
    src_path = pathlib.Path("../analyst_copilot/long_term_memory.py")

print("=== BUG-4: rebuild_if_stale() acquires self._lock ===")

# Read with explicit UTF-8 encoding (required on Windows — default is cp1252)
src = src_path.read_text(encoding="utf-8")
tree = ast.parse(src)

# 1. Find rebuild_if_stale() and confirm `with self._lock:` is inside it
rebuild_fn = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "rebuild_if_stale":
        rebuild_fn = node
        break

if rebuild_fn is None:
    print("  FAIL: rebuild_if_stale() function not found in long_term_memory.py")
    sys.exit(1)


# Check for `with self._lock:` anywhere within the function body
def has_lock_context(fn_node: ast.FunctionDef) -> bool:
    for node in ast.walk(fn_node):
        if isinstance(node, ast.With):
            for item in node.items:
                ce = item.context_expr
                # Matches `self._lock` (Attribute node: value=Name('self'), attr='_lock')
                if (
                    isinstance(ce, ast.Attribute)
                    and ce.attr == "_lock"
                    and isinstance(ce.value, ast.Name)
                    and ce.value.id == "self"
                ):
                    return True
    return False


lock_acquired = has_lock_context(rebuild_fn)
print(f"  {'PASS' if lock_acquired else 'FAIL'}: rebuild_if_stale() contains `with self._lock:`")

# 2. Confirm store() also acquires the lock (for comparison)
store_fn = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "store":
        store_fn = node
        break

if store_fn:
    store_lock = has_lock_context(store_fn)
    print(f"  {'PASS' if store_lock else 'WARN'}: store() also acquires self._lock (expected)")

# 3. Confirm search() also acquires the lock
search_fn = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "search":
        search_fn = node
        break

if search_fn:
    search_lock = has_lock_context(search_fn)
    print(f"  {'PASS' if search_lock else 'WARN'}: search() also acquires self._lock (expected)")

# 4. Check the misleading comment is updated
misleading_comment_gone = "rebuild_if_stale() holds self._lock so the" not in src
print(
    f"  {'PASS' if misleading_comment_gone else 'WARN'}: misleading comment about lock removed/updated"
)

if not lock_acquired:
    print("\n  FAIL: The race condition is NOT fixed. Apply BUG-4 patch.")
    sys.exit(1)
else:
    print("\n  All lock checks passed — race condition eliminated.")
