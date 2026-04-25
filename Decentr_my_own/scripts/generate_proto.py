from __future__ import annotations

from pathlib import Path

from grpc_tools import protoc


ROOT = Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT / "decentr_my_own" / "comm" / "proto"
PROTO_FILE = PROTO_DIR / "peer.proto"


def main() -> int:
    code = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={PROTO_DIR}",
            f"--grpc_python_out={PROTO_DIR}",
            str(PROTO_FILE),
        ]
    )
    if code != 0:
        return code

    grpc_file = PROTO_DIR / "peer_pb2_grpc.py"
    text = grpc_file.read_text(encoding="utf-8")
    text = text.replace("import peer_pb2 as peer__pb2", "from . import peer_pb2 as peer__pb2")
    grpc_file.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
