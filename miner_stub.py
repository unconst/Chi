"""
Minimal Lemma miner stub for local integration testing.

Features:
- Exposes POST /solve over HTTP.
- Optionally verifies validator Epistula request headers.
- Returns Lean proof code for known demo challenges.
- Signs response body with miner hotkey Epistula headers.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from bittensor_wallet import Wallet

HOST = os.getenv("MINER_HOST", "0.0.0.0")
PORT = int(os.getenv("MINER_PORT", "8080"))
WALLET_NAME = os.getenv("WALLET_NAME", "default")
HOTKEY_NAME = os.getenv("HOTKEY_NAME", "default")
VERIFY_VALIDATOR_EPISTULA = os.getenv("VERIFY_VALIDATOR_EPISTULA", "false").lower() in {
    "1",
    "true",
    "yes",
}
EPISTULA_MAX_SKEW_SECONDS = int(os.getenv("EPISTULA_MAX_SKEW_SECONDS", "60"))

wallet = Wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)


def create_epistula_headers(hotkey: Any, body: bytes) -> dict[str, str]:
    ts = str(time.time_ns())
    msg = f"{ts}.{hashlib.sha256(body).hexdigest()}".encode("utf-8")
    sig = hotkey.sign(msg)
    sig_hex = sig.hex() if isinstance(sig, bytes) else bytes(sig).hex()
    return {
        "X-Epistula-Timestamp": ts,
        "X-Epistula-Signature": sig_hex,
        "X-Epistula-Hotkey": hotkey.ss58_address,
    }


def verify_epistula_headers(headers: dict[str, str], body: bytes) -> tuple[bool, str]:
    ts = headers.get("X-Epistula-Timestamp", "")
    sig = headers.get("X-Epistula-Signature", "")
    hotkey = headers.get("X-Epistula-Hotkey", "")
    if not ts or not sig or not hotkey:
        return False, "missing epistula headers"
    try:
        ts_ns = int(ts)
    except ValueError:
        return False, "invalid timestamp"
    if abs(time.time_ns() - ts_ns) > EPISTULA_MAX_SKEW_SECONDS * 1_000_000_000:
        return False, "timestamp skew too large"
    try:
        from substrateinterface import Keypair  # type: ignore

        msg = f"{ts}.{hashlib.sha256(body).hexdigest()}".encode("utf-8")
        kp = Keypair(ss58_address=hotkey)
        if not kp.verify(msg, bytes.fromhex(sig)):
            return False, "signature verify failed"
    except Exception as exc:
        return False, f"verify exception: {exc}"
    return True, "ok"


def solve_challenge(challenge_id: str) -> str:
    # Demo-only canned proofs for current CHALLENGE_BANK in validator.py.
    if challenge_id in ("nat_add_zero_right", "nat_zero_add_left", "nat_mul_one_right"):
        return "by simp"
    if challenge_id == "nat_add_assoc":
        return "by simpa [Nat.add_assoc]"
    return "by simp"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/solve":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if VERIFY_VALIDATOR_EPISTULA:
            ok, reason = verify_epistula_headers(dict(self.headers.items()), raw)
            if not ok:
                self.send_response(401)
                self.end_headers()
                self.wfile.write(reason.encode("utf-8"))
                return

        try:
            req = json.loads(raw.decode("utf-8"))
            challenge_id = str(req.get("challenge_id", "")).strip()
        except Exception:
            self.send_response(400)
            self.end_headers()
            return

        proof_code = solve_challenge(challenge_id)
        response_obj = {
            "proof_code": proof_code,
            "metadata": {"miner": wallet.hotkey.ss58_address, "challenge_id": challenge_id},
        }
        response_body = json.dumps(response_obj).encode("utf-8")
        sig_headers = create_epistula_headers(wallet.hotkey, response_body)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        for k, v in sig_headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Miner stub running on http://{HOST}:{PORT}")
    server.serve_forever()
