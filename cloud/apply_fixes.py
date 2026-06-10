#!/usr/bin/env python3
"""
apply_fixes.py — one-shot post-iCloud-migration patch script.

Run from cloud/ after you've moved the repo out of iCloud:

    cd ~/dev/final_proj/Decentr_my_own/cloud
    python3 apply_fixes.py

What it does:
  1. Removes the top-level "meta" key from build_inventory() in orchestrate.py
     (Pydantic models with extra='forbid' reject unknown keys, so "meta" would
      crash start-run if it ever reached the node).
  2. Replaces provision/selectel.py with the corrected version that uses
     Keystone v3 password auth instead of the broken "token" (API-key) method.
  3. Fixes the hardcoded overlay="tailscale" in build_inventory() for the
     subnet provisioning path (private IPs, no Tailscale needed).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# Fix 1: remove "meta" block from build_inventory() in orchestrate.py
# ---------------------------------------------------------------------------

def fix_orchestrate_meta(path: Path) -> bool:
    text = path.read_text()

    # Match the "meta": { ... }, block (multi-line, with trailing comma).
    # We look for the pattern inside a dict literal — ends at the closing brace
    # followed by optional whitespace and a comma.
    pattern = re.compile(
        r'[ \t]*"meta"\s*:\s*\{[^}]*\}\s*,?\s*\n',
        re.MULTILINE | re.DOTALL,
    )
    new_text, n = pattern.subn("", text)
    if n == 0:
        # Try a slightly more permissive multi-line match
        pattern2 = re.compile(
            r'[ \t]*"meta"\s*:\s*\{.*?\},?\s*\n',
            re.MULTILINE | re.DOTALL,
        )
        new_text, n = pattern2.subn("", text)

    if n == 0:
        print(f"  [skip]  'meta' key not found in {path.name} — already fixed or pattern changed.")
        return False

    path.write_text(new_text)
    print(f"  [OK]    Removed 'meta' block ({n} occurrence(s)) from {path.name}")
    return True


# ---------------------------------------------------------------------------
# Fix 2: replace provision/selectel.py with corrected password-auth version
# ---------------------------------------------------------------------------

def fix_selectel(cloud_dir: Path) -> bool:
    src  = cloud_dir / "provision" / "selectel_new.py"
    dst  = cloud_dir / "provision" / "selectel.py"

    if not src.exists():
        print(f"  [skip]  {src.name} not found — run from cloud/ directory.")
        return False

    shutil.copy2(src, dst)
    src.unlink()
    os.chmod(dst, 0o755)
    print(f"  [OK]    Replaced provision/selectel.py with corrected password-auth version.")
    return True


# ---------------------------------------------------------------------------
# Fix 3: overlay field — replace hardcoded "tailscale" with pool-aware value
# ---------------------------------------------------------------------------

def fix_overlay(path: Path) -> bool:
    text = path.read_text()

    # Look for the line that sets overlay inside build_inventory.
    # Typical pattern (from our generated code):
    #   "overlay": "tailscale",
    # We want to make it conditional on whether the pool has a tailscale_network
    # field or similar. Safest minimal fix: change to "subnet" which is what
    # the Terraform-provisioned nodes use (they talk over private subnet IPs).
    #
    # If the codebase actually reads overlay from the experiment config, skip.
    if '"overlay": "tailscale"' not in text and '"overlay":"tailscale"' not in text:
        print(f"  [skip]  overlay='tailscale' literal not found — may already be dynamic.")
        return False

    # Replace the hardcoded tailscale with a reference to exp.overlay (with fallback).
    # We add a helper one line above the return statement in build_inventory.
    # Strategy: add `_overlay = getattr(exp, "overlay", "subnet")` and use it.
    new_text = text.replace(
        '"overlay": "tailscale"',
        '"overlay": getattr(exp, "overlay", "subnet")',
    ).replace(
        '"overlay":"tailscale"',
        '"overlay": getattr(exp, "overlay", "subnet")',
    )

    if new_text == text:
        print(f"  [skip]  overlay replacement produced no change (unexpected).")
        return False

    path.write_text(new_text)
    print(f"  [OK]    Made overlay field dynamic in build_inventory().")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cloud_dir = HERE  # we expect to be run from cloud/

    orchestrate = cloud_dir / "orchestrate.py"
    if not orchestrate.exists():
        sys.exit(
            f"ERROR: {orchestrate} not found.\n"
            "Run this script from the cloud/ directory:\n"
            "  cd .../Decentr_my_own/cloud && python3 apply_fixes.py"
        )

    print("Applying pending fixes to Decentr cloud toolchain...\n")

    changed = []

    print("1/3  orchestrate.py — remove top-level 'meta' key from build_inventory()")
    if fix_orchestrate_meta(orchestrate):
        changed.append("orchestrate.py (meta removed)")

    print("\n2/3  provision/selectel.py — replace with password-auth version")
    if fix_selectel(cloud_dir):
        changed.append("provision/selectel.py (auth fixed)")

    print("\n3/3  orchestrate.py — make overlay field pool-aware (not hardcoded 'tailscale')")
    if fix_overlay(orchestrate):
        changed.append("orchestrate.py (overlay dynamic)")

    print()
    if changed:
        print("Done. Changed files:")
        for f in changed:
            print(f"  • {f}")
        print("\nNext steps:")
        print("  • Verify: python3 orchestrate.py check --pool pool.yaml.example --dry-run")
        print("  • Set env vars (SEL_DOMAIN_NAME etc.) and test: python3 provision/selectel.py list-flavors")
    else:
        print("Nothing changed — all fixes already applied or patterns not matched.")
        print("If you expected changes, the file may still be iCloud-evicted. Check:")
        print("  ls -la orchestrate.py  # size should be > 0")


if __name__ == "__main__":
    main()
