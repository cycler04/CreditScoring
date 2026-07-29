#!/usr/bin/env python3
"""Validate the shared agent knowledge layer and thin Codex adapter."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = ROOT / ".agents"
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
NUMBERED_FILE = re.compile(r"^(\d{2})_[^/]+\.md$")


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_numbering(errors: list[str]) -> None:
    for directory in (AGENTS_DIR, AGENTS_DIR / "workflows"):
        numbers = sorted(
            int(match.group(1))
            for path in directory.glob("*.md")
            if (match := NUMBERED_FILE.match(path.name))
        )
        if numbers and numbers != list(range(1, len(numbers) + 1)):
            fail(errors, f"non-contiguous numbering in {directory.relative_to(ROOT)}: {numbers}")


def validate_links(errors: list[str]) -> None:
    for path in [ROOT / "AGENTS.md", *AGENTS_DIR.rglob("*.md")]:
        text = path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                fail(
                    errors,
                    f"broken link: {path.relative_to(ROOT)} -> {raw_target}",
                )


def validate_skill(skill: Path, errors: list[str]) -> None:
    skill_md = skill / "SKILL.md"
    ui_yaml = skill / "agents" / "openai.yaml"
    if not skill_md.is_file():
        fail(errors, f"missing {skill_md.relative_to(ROOT)}")
        return
    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---\n") or text.count("---") < 2:
        fail(errors, f"invalid frontmatter in {skill_md.relative_to(ROOT)}")
    for key in ("name:", "description:"):
        if key not in text.split("---", 2)[1]:
            fail(errors, f"missing {key} in {skill_md.relative_to(ROOT)}")
    if "[TODO" in text:
        fail(errors, f"unresolved TODO in {skill_md.relative_to(ROOT)}")
    if not ui_yaml.is_file():
        fail(errors, f"missing {ui_yaml.relative_to(ROOT)}")


def validate_toml(errors: list[str]) -> None:
    for path in ROOT.rglob("*.toml"):
        if ".git" in path.parts:
            continue
        try:
            tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            fail(errors, f"invalid TOML {path.relative_to(ROOT)}: {exc}")


def validate_gitignore(errors: list[str]) -> None:
    if not (ROOT / ".git").exists():
        return
    targets = [
        "AGENTS.md",
        ".agents/01_overview.md",
        ".agents/memory/MEMORY.md",
        ".agents/skills/research-paper/SKILL.md",
        ".codex/config.toml",
    ]
    ignored: list[str] = []
    for target in targets:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", target],
            cwd=ROOT,
            check=False,
        )
        if result.returncode == 0:
            ignored.append(target)
    if ignored:
        fail(errors, f"agent infrastructure is ignored: {', '.join(ignored)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also validate TOML and gitignore")
    args = parser.parse_args()
    errors: list[str] = []
    validate_numbering(errors)
    validate_links(errors)
    for skill in sorted((AGENTS_DIR / "skills").iterdir()):
        if skill.is_dir():
            validate_skill(skill, errors)
    if args.full:
        validate_toml(errors)
        validate_gitignore(errors)
    if errors:
        print("\n".join(f"ERROR: {message}" for message in errors), file=sys.stderr)
        return 1
    print("Workspace agent infrastructure: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
