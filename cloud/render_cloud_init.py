#!/usr/bin/env python3
"""
render_cloud_init.py — render a cloud-init template by substituting
{{PLACEHOLDER}} tokens with values from secrets.env.

Usage:
  # Terraform/subnet mode (no Tailscale):
  python3 render_cloud_init.py \
    --template provision/terraform/cloud-init.subnet.yaml.template \
    --out provision/terraform/cloud-init.rendered.yaml

  # Manual/Tailscale mode:
  python3 render_cloud_init.py \
    --template cloud-init.yaml.template \
    --out cloud-init.rendered.yaml
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


def _load_env(env_path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines from a shell env file (ignores comments, blanks).

    Inline comments (KEY=value  # comment) are stripped — anything after
    unquoted whitespace + '#' is treated as a comment.
    """
    env: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip().strip('"').strip("'")
        # Strip trailing inline comment: "value  # comment" → "value"
        # Only when comment is preceded by whitespace (not inside a value)
        import re as _re
        val = _re.sub(r'\s+#.*$', '', val).strip()
        env[key.strip()] = val
    return env


def _read_pubkey(path: str) -> str:
    p = Path(os.path.expanduser(path))
    if not p.exists():
        sys.exit(f"ERROR: SSH public key not found: {p}")
    return p.read_text().strip()


def _read_deploy_key(path: str) -> str:
    """Read private key; return empty string if path is blank (public repo)."""
    path = path.strip()
    if not path:
        return ""
    p = Path(os.path.expanduser(path))
    if not p.exists():
        sys.exit(f"ERROR: Deploy key not found: {p}")
    return p.read_text()


def _indent(text: str, spaces: int = 6) -> str:
    """Indent every line of text by `spaces` spaces (for cloud-init write_files)."""
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def _repo_host(url: str) -> str:
    """Extract hostname from git URL (SSH or HTTPS)."""
    url = url.strip()
    if url.startswith("git@"):
        # git@github.com:user/repo.git  →  github.com
        return url.split("@", 1)[1].split(":")[0]
    try:
        return urlparse(url).hostname or "github.com"
    except Exception:
        return "github.com"


def render(template_path: Path, env: dict[str, str]) -> str:
    tpl = template_path.read_text()

    # Resolve values
    ssh_pubkey          = _read_pubkey(env.get("ORCHESTRATOR_SSH_PUBKEY_PATH", ""))
    results_key         = _read_deploy_key(env.get("RESULTS_DEPLOY_KEY_PATH", ""))
    project_key         = _read_deploy_key(env.get("PROJECT_DEPLOY_KEY_PATH", ""))
    results_repo_url    = env.get("RESULTS_REPO_URL", "")
    project_repo_url    = env.get("PROJECT_REPO_URL", "")

    subs: dict[str, str] = {
        "ORCHESTRATOR_SSH_PUBKEY":        ssh_pubkey,
        "RESULTS_DEPLOY_KEY_INDENTED":    _indent(results_key) if results_key else "        # no deploy key",
        "PROJECT_DEPLOY_KEY_INDENTED":    _indent(project_key) if project_key else "        # no deploy key",
        "RESULTS_REPO_URL":               results_repo_url,
        "RESULTS_REPO_HOST":              _repo_host(results_repo_url),
        "PROJECT_REPO_URL":               project_repo_url,
        "PROJECT_REPO_HOST":              _repo_host(project_repo_url),
        "PROJECT_REPO_BRANCH":            env.get("PROJECT_REPO_BRANCH", "main"),
        "RESULTS_GIT_NAME":               env.get("RESULTS_GIT_NAME",  "Decentr Bot"),
        "RESULTS_GIT_EMAIL":              env.get("RESULTS_GIT_EMAIL", "bot@example.com"),
        "TAILSCALE_AUTH_KEY":             env.get("TAILSCALE_AUTH_KEY", ""),
    }

    missing = []
    def _replace(m: re.Match) -> str:
        key = m.group(1)
        if key not in subs:
            missing.append(key)
            return m.group(0)
        return subs[key]

    rendered = re.sub(r"\{\{([A-Z0-9_]+)\}\}", _replace, tpl)

    if missing:
        sys.exit(f"ERROR: unknown placeholder(s) in template: {missing}")
    return rendered


def main() -> None:
    ap = argparse.ArgumentParser(description="Render cloud-init template from secrets.env")
    ap.add_argument("--template", required=True, help="Path to *.yaml.template")
    ap.add_argument("--out",      required=True, help="Output path (.rendered.yaml)")
    ap.add_argument("--secrets",  default="secrets.env",
                    help="Path to secrets.env (default: secrets.env in CWD)")
    args = ap.parse_args()

    secrets_path = Path(args.secrets)
    if not secrets_path.exists():
        # Try relative to the cloud/ dir (script location)
        secrets_path = Path(__file__).parent / "secrets.env"
    if not secrets_path.exists():
        sys.exit(f"ERROR: secrets.env not found. Copy secrets.env.example → secrets.env and fill it in.")

    env      = _load_env(secrets_path)
    rendered = render(Path(args.template), env)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    print(f"Rendered → {out}  ({len(rendered)} bytes)")


if __name__ == "__main__":
    main()
