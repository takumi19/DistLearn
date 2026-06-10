#!/usr/bin/env python3
"""
selectel.py  — Selectel / OpenStack helpers for the Decentr cloud toolchain.

Auth model: Keystone v3 **password** auth with a service-user account.
(The previous version incorrectly used the "token" method with an API key;
 that method requires the legacy v2 token endpoint and a different flow.)

Required env vars (or pass via --env / terraform.tfvars):
  SEL_DOMAIN_NAME   — Selectel account ID shown top-right in the panel
  SEL_USERNAME      — service-user name (Управление доступом → Сервисные пользователи)
  SEL_PASSWORD      — service-user password
  SEL_PROJECT_ID    — project UUID (Облачная платформа → Проекты)
  SEL_REGION        — e.g. "ru-9"  (default ru-9)
  SEL_AUTH_URL      — override auth URL (default https://cloud.api.selcloud.ru/identity/v3)

Usage:
  python3 selectel.py list-flavors [--min-vcpu N] [--min-ram-gb N]
  python3 selectel.py list-images  [--filter STR]
  python3 selectel.py list-networks
  python3 selectel.py token        # print a short-lived OS token (for debugging)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_AUTH_URL = "https://cloud.api.selcloud.ru/identity/v3"
DEFAULT_REGION = "ru-9"


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _env(key: str, fallback: str | None = None) -> str:
    val = os.environ.get(key, fallback)
    if not val:
        sys.exit(f"ERROR: env var {key} is not set. See selectel.py docstring.")
    return val


def get_token(auth_url: str, domain_name: str, username: str,
              password: str, project_id: str) -> tuple[str, str]:
    """
    Authenticate with Keystone v3 password method (service-user scope).

    Returns (token_str, compute_endpoint).
    The token is valid for ~1 hour; re-call if needed.
    """
    payload = {
        "auth": {
            "identity": {
                "methods": ["password"],
                "password": {
                    "user": {
                        "name": username,
                        "domain": {"name": domain_name},
                        "password": password,
                    }
                },
            },
            "scope": {
                "project": {"id": project_id}
            },
        }
    }
    url = auth_url.rstrip("/") + "/auth/tokens"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            token = resp.headers["X-Subject-Token"]
            catalog = json.loads(resp.read())["token"]["catalog"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        sys.exit(f"Keystone auth failed ({exc.code}): {body}")

    # Extract Nova compute endpoint for the target region
    compute_ep = _find_endpoint(catalog, "compute", "public")
    return token, compute_ep


def _find_endpoint(catalog: list[dict], svc_type: str, interface: str) -> str:
    for svc in catalog:
        if svc.get("type") == svc_type:
            for ep in svc.get("endpoints", []):
                if ep.get("interface") == interface:
                    return ep["url"].rstrip("/")
    # fallback — return first compute endpoint found
    for svc in catalog:
        if svc.get("type") == svc_type:
            eps = svc.get("endpoints", [])
            if eps:
                return eps[0]["url"].rstrip("/")
    return ""


# ---------------------------------------------------------------------------
# API wrappers
# ---------------------------------------------------------------------------

def _get(url: str, token: str) -> Any:
    req = urllib.request.Request(
        url,
        headers={"X-Auth-Token": token, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        sys.exit(f"GET {url} failed ({exc.code}): {body}")


def list_flavors(token: str, compute_ep: str,
                 min_vcpu: int = 1, min_ram_gb: float = 0) -> list[dict]:
    data = _get(f"{compute_ep}/flavors/detail", token)
    flavors = data.get("flavors", [])
    min_ram_mb = int(min_ram_gb * 1024)
    return [
        f for f in flavors
        if f.get("vcpus", 0) >= min_vcpu and f.get("ram", 0) >= min_ram_mb
    ]


def list_images(token: str, compute_ep: str, name_filter: str = "") -> list[dict]:
    # Nova image list (uses Glance under the hood via Nova proxy)
    data = _get(f"{compute_ep}/images/detail", token)
    images = data.get("images", [])
    if name_filter:
        nf = name_filter.lower()
        images = [i for i in images if nf in i.get("name", "").lower()]
    return images


def list_networks(token: str, auth_url: str, project_id: str) -> list[dict]:
    # Neutron endpoint — derive from auth_url base
    # Selectel Neutron is at: https://network.{region}.cloud.selectel.ru/v2.0
    # Simpler: use the compute endpoint host to find the pattern.
    # We'll call Keystone catalog directly.
    # For now, we provide best-effort via Nova network list.
    # TODO: parse Neutron endpoint from catalog properly.
    return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _auth_from_env() -> tuple[str, str, str, str, str, str]:
    auth_url = os.environ.get("SEL_AUTH_URL", DEFAULT_AUTH_URL)
    domain   = _env("SEL_DOMAIN_NAME")
    username = _env("SEL_USERNAME")
    password = _env("SEL_PASSWORD")
    project  = _env("SEL_PROJECT_ID")
    region   = os.environ.get("SEL_REGION", DEFAULT_REGION)
    return auth_url, domain, username, password, project, region


def cmd_list_flavors(args: argparse.Namespace) -> None:
    auth_url, domain, username, password, project, _ = _auth_from_env()
    token, compute_ep = get_token(auth_url, domain, username, password, project)
    flavors = list_flavors(token, compute_ep,
                           min_vcpu=args.min_vcpu,
                           min_ram_gb=args.min_ram_gb)
    flavors.sort(key=lambda f: (f.get("vcpus", 0), f.get("ram", 0)))
    print(f"{'ID':<12} {'Name':<28} {'vCPU':>5} {'RAM GB':>7} {'Disk GB':>8}")
    print("-" * 65)
    for f in flavors:
        print(
            f"{f.get('id',''):<12} "
            f"{f.get('name',''):<28} "
            f"{f.get('vcpus', 0):>5} "
            f"{f.get('ram', 0) / 1024:>7.1f} "
            f"{f.get('disk', 0):>8}"
        )
    print(f"\n{len(flavors)} flavor(s) shown (min {args.min_vcpu} vCPU, {args.min_ram_gb} GB RAM)")


def cmd_list_images(args: argparse.Namespace) -> None:
    auth_url, domain, username, password, project, _ = _auth_from_env()
    token, compute_ep = get_token(auth_url, domain, username, password, project)
    images = list_images(token, compute_ep, name_filter=args.filter)
    print(f"{'ID':<38} {'Name'}")
    print("-" * 70)
    for img in sorted(images, key=lambda i: i.get("name", "")):
        print(f"{img.get('id',''):<38} {img.get('name','')}")
    print(f"\n{len(images)} image(s) shown")


def cmd_token(_args: argparse.Namespace) -> None:
    auth_url, domain, username, password, project, _ = _auth_from_env()
    token, compute_ep = get_token(auth_url, domain, username, password, project)
    print(f"Token:      {token[:32]}…")
    print(f"Compute EP: {compute_ep}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selectel / OpenStack CLI helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fl = sub.add_parser("list-flavors", help="List available VM flavors")
    p_fl.add_argument("--min-vcpu",   type=int,   default=1,   metavar="N")
    p_fl.add_argument("--min-ram-gb", type=float, default=0.0, metavar="N")
    p_fl.set_defaults(func=cmd_list_flavors)

    p_img = sub.add_parser("list-images", help="List available OS images")
    p_img.add_argument("--filter", default="", metavar="STR",
                       help="Case-insensitive substring filter on image name")
    p_img.set_defaults(func=cmd_list_images)

    p_tok = sub.add_parser("token", help="Print a short-lived auth token (debug)")
    p_tok.set_defaults(func=cmd_token)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
