# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scan a whitelist export and unpacked assets without displaying matched values."""

import argparse
import gzip
import json
import re
import subprocess
import tarfile
import tomllib
from pathlib import Path

RULES = {
    "private_machine_path": re.compile(rb"/(?:home|media/data[0-9]+)/[^\s\x00\"']+"),
    "private_network_address": re.compile(
        rb"(?<![\w.-])(?:10\.(?:\d+\.){2}\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)(?![\w.-])"
    ),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_url": re.compile(rb"https?://[^/\s\"']+:[^/\s\"']+@"),
    "internal_operation_record": re.compile(
        b"(?:" + b"codex" + b'_reviews|"' + b"session" + b'_id"\\s*:|GPU-[a-fA-F0-9]{8})'
    ),
}


def inspect_stream(stream, findings, name, package_versions):
    for line_number, line in enumerate(stream, 1):
        for category, pattern in RULES.items():
            if category == "private_network_address":
                if name == "uv.lock" and line.startswith(b'version = "'):
                    value = line.decode().split('"')[1]
                    if value in package_versions.values():
                        continue
                if name == "environment/reference.json":
                    try:
                        field = json.loads(b"{" + line.strip().rstrip(b",") + b"}")
                    except (ValueError, UnicodeDecodeError):
                        field = {}
                    if field and all(package_versions.get(k) == v for k, v in field.items()):
                        continue
            if pattern.search(line):
                findings.append(dict(file=name, line=line_number, category=category))


def inspect_file(path, findings, label=None, package_versions=None):
    name = label or path.name
    if path.is_symlink():
        findings.append(dict(file=name, category="symlink_requires_review"))
        return
    package_versions = package_versions or {}
    if path.name.endswith((".tar.gz", ".tar.xz", ".tar")):
        with tarfile.open(path) as archive:
            for member in archive:
                if member.uid or member.gid or member.uname or member.gname:
                    findings.append(dict(file=name, category="archive_owner_metadata"))
                if member.issym() or member.islnk():
                    findings.append(dict(file=name, category="archive_link"))
                for category, pattern in RULES.items():
                    if pattern.search(member.name.encode()):
                        findings.append(dict(file=name, category=category))
                if member.isfile():
                    with archive.extractfile(member) as stream:
                        inspect_stream(stream, findings, name + "/" + member.name, package_versions)
    else:
        stream = gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
        with stream:
            inspect_stream(stream, findings, name, package_versions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("outputs/privacy_audit.json"))
    args = parser.parse_args()
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=args.root).decode().split("\0")
    findings: list[dict] = []
    lock = tomllib.loads((args.root / "uv.lock").read_text())
    package_versions = {p["name"]: p["version"] for p in lock["package"]}
    for name in filter(None, files):
        inspect_file(args.root / name, findings, name, package_versions)
    for path in args.artifact:
        inspect_file(path, findings, path.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(files_checked=len(files) - 1, findings=findings), indent=2) + "\n")
    print(f"Privacy audit: {len(findings)} findings; values are never printed")
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
