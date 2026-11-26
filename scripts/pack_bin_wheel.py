"""Pack prebuilt executables into a wheel that installs them into the environment's bin directory."""

import argparse
import base64
import hashlib
import io
import zipfile
from pathlib import Path


def record_line(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", required=True, help="wheel platform tag, e.g. manylinux_2_28_x86_64")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--license", required=True, help="SPDX license expression")
    parser.add_argument("--bin", action="append", type=Path, required=True)
    parser.add_argument("--license-file", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    dist = args.name.replace("-", "_")
    data_dir = f"{dist}-{args.version}.data"
    info_dir = f"{dist}-{args.version}.dist-info"
    tag = f"py3-none-{args.platform}"

    metadata = "\n".join(
        [
            "Metadata-Version: 2.4",
            f"Name: {args.name}",
            f"Version: {args.version}",
            f"Summary: {args.summary}",
            f"License-Expression: {args.license}",
            *(f"License-File: {path.name}" for path in args.license_file),
            "",
        ]
    )
    wheel = f"Wheel-Version: 1.0\nGenerator: pack_bin_wheel.py\nRoot-Is-Purelib: false\nTag: {tag}\n"

    entries: list[tuple[str, bytes, int]] = [
        (f"{data_dir}/scripts/{path.name}", path.read_bytes(), 0o755) for path in args.bin
    ]
    entries += [(f"{info_dir}/licenses/{path.name}", path.read_bytes(), 0o644) for path in args.license_file]
    entries += [(f"{info_dir}/METADATA", metadata.encode(), 0o644), (f"{info_dir}/WHEEL", wheel.encode(), 0o644)]
    record = [record_line(name, data) for name, data, _ in entries] + [f"{info_dir}/RECORD,,"]
    entries.append((f"{info_dir}/RECORD", ("\n".join(record) + "\n").encode(), 0o644))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{dist}-{args.version}-{tag}.whl"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = (0o100000 | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    out.write_bytes(buffer.getvalue())
    print(out)


if __name__ == "__main__":
    main()
