"""
Lemma subnet validator.

What this validator measures:
    Miner ability to solve deterministic Lean 4 theorem challenges with valid,
    concise, low-dependency proofs.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import logging
import math
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import bittensor as bt
import click
from bittensor_wallet import Wallet
from challenge_engine import Challenge, choose_challenge, choose_challenges

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

HEARTBEAT_TIMEOUT_SECONDS = 600
BLOCK_TIME_SECONDS = 12
DEFAULT_EMA_ALPHA = 0.35
STATE_PATH = Path(os.getenv("VALIDATOR_STATE_PATH", "data/validator_state.json"))
ROUND_LOG_PATH = Path(os.getenv("ROUND_LOG_PATH", "data/round_results.jsonl"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("MINER_REQUEST_TIMEOUT", "45"))
LEAN_TIMEOUT_SECONDS = float(os.getenv("LEAN_TIMEOUT_SECONDS", "30"))
MAX_RESPONSE_CHARS = int(os.getenv("MAX_RESPONSE_CHARS", "16000"))
MAX_PROOF_CHARS_FOR_SCORING = int(os.getenv("MAX_PROOF_CHARS_FOR_SCORING", "12000"))
MAX_PROOF_LINES = int(os.getenv("MAX_PROOF_LINES", "300"))
MAX_IMPORT_LINES = int(os.getenv("MAX_IMPORT_LINES", "4"))
COMMITMENT_REFRESH_BLOCKS = int(os.getenv("COMMITMENT_REFRESH_BLOCKS", "20"))
BATCH_SIZE = int(os.getenv("CHALLENGE_BATCH_SIZE", "6"))
DIFFICULTY_WEIGHTING_ENABLED = os.getenv("DIFFICULTY_WEIGHTING_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
}
EPISTULA_SIGN_REQUESTS = os.getenv("EPISTULA_SIGN_REQUESTS", "true").lower() in {
    "1",
    "true",
    "yes",
}
EPISTULA_VERIFY_RESPONSES = os.getenv("EPISTULA_VERIFY_RESPONSES", "true").lower() in {
    "1",
    "true",
    "yes",
}
EPISTULA_STRICT_VERIFY = os.getenv("EPISTULA_STRICT_VERIFY", "true").lower() in {
    "1",
    "true",
    "yes",
}
EPISTULA_MAX_SKEW_SECONDS = int(os.getenv("EPISTULA_MAX_SKEW_SECONDS", "60"))
LEAN_MEMORY_LIMIT_MB = int(os.getenv("LEAN_MEMORY_LIMIT_MB", "1024"))
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() in {"1", "true", "yes"}
METRICS_HOST = os.getenv("METRICS_HOST", "0.0.0.0")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9109"))

METRICS_LOCK = threading.Lock()
METRICS: dict[str, Any] = {
    "start_time_ns": time.time_ns(),
    "attempts_total": 0,
    "valid_total": 0,
    "invalid_total": 0,
    "epistula_fail_total": 0,
    "timeout_total": 0,
    "http_error_total": 0,
    "weights_set_success_total": 0,
    "weights_set_fail_total": 0,
    "last_round_block": 0,
    "last_round_batch_size": 0,
    "per_uid": {},
}


def heartbeat_monitor(last_heartbeat: list[float], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        time.sleep(5)
        if time.time() - last_heartbeat[0] > HEARTBEAT_TIMEOUT_SECONDS:
            logger.error(
                "No heartbeat detected in %ss. Restarting process.",
                HEARTBEAT_TIMEOUT_SECONDS,
            )
            logging.shutdown()
            os.execv(sys.executable, [sys.executable, *sys.argv])


def _metrics_snapshot() -> dict[str, Any]:
    with METRICS_LOCK:
        return json.loads(json.dumps(METRICS))


def _render_prometheus(metrics: dict[str, Any]) -> str:
    lines = [
        "# TYPE lemma_validator_attempts_total counter",
        f"lemma_validator_attempts_total {metrics.get('attempts_total', 0)}",
        "# TYPE lemma_validator_valid_total counter",
        f"lemma_validator_valid_total {metrics.get('valid_total', 0)}",
        "# TYPE lemma_validator_invalid_total counter",
        f"lemma_validator_invalid_total {metrics.get('invalid_total', 0)}",
        "# TYPE lemma_validator_epistula_fail_total counter",
        f"lemma_validator_epistula_fail_total {metrics.get('epistula_fail_total', 0)}",
        "# TYPE lemma_validator_timeout_total counter",
        f"lemma_validator_timeout_total {metrics.get('timeout_total', 0)}",
        "# TYPE lemma_validator_http_error_total counter",
        f"lemma_validator_http_error_total {metrics.get('http_error_total', 0)}",
        "# TYPE lemma_validator_weights_set_success_total counter",
        f"lemma_validator_weights_set_success_total {metrics.get('weights_set_success_total', 0)}",
        "# TYPE lemma_validator_weights_set_fail_total counter",
        f"lemma_validator_weights_set_fail_total {metrics.get('weights_set_fail_total', 0)}",
        "# TYPE lemma_validator_last_round_block gauge",
        f"lemma_validator_last_round_block {metrics.get('last_round_block', 0)}",
        "# TYPE lemma_validator_last_round_batch_size gauge",
        f"lemma_validator_last_round_batch_size {metrics.get('last_round_batch_size', 0)}",
    ]
    per_uid = metrics.get("per_uid", {})
    for uid, vals in per_uid.items():
        lines.append(f'lemma_validator_uid_instant_score{{uid="{uid}"}} {vals.get("instant", 0.0)}')
        lines.append(f'lemma_validator_uid_ema_score{{uid="{uid}"}} {vals.get("ema", 0.0)}')
        lines.append(f'lemma_validator_uid_valid_rate{{uid="{uid}"}} {vals.get("valid_rate", 0.0)}')
    return "\n".join(lines) + "\n"


def _start_metrics_server() -> threading.Thread | None:
    if not METRICS_ENABLED:
        return None

    class MetricsHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/metrics":
                data = _render_prometheus(_metrics_snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path == "/metrics.json":
                data = json.dumps(_metrics_snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    def _serve() -> None:
        with http.server.ThreadingHTTPServer((METRICS_HOST, METRICS_PORT), MetricsHandler) as server:
            logger.info("Metrics server listening on http://%s:%s", METRICS_HOST, METRICS_PORT)
            server.serve_forever()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t


def parse_miner_endpoints() -> dict[int, str]:
    """
    Parse endpoints from env var:
    MINER_ENDPOINTS='0=http://127.0.0.1:8080,3=http://10.0.0.5:8090'
    """
    raw = os.getenv("MINER_ENDPOINTS", "").strip()
    if not raw:
        return {}
    mapping: dict[int, str] = {}
    for item in raw.split(","):
        if "=" not in item:
            continue
        uid_raw, endpoint = item.split("=", 1)
        try:
            mapping[int(uid_raw.strip())] = endpoint.strip().rstrip("/")
        except ValueError:
            logger.warning("Ignoring invalid MINER_ENDPOINTS entry: %s", item)
    return mapping


def _extract_endpoint_from_commitment(commitment: Any) -> str | None:
    """
    Accepts plain URL commitments or JSON payload commitments.
    Supported examples:
      - "https://miner.example.com"
      - {"endpoint": "https://miner.example.com"}
      - {"url": "https://miner.example.com"}
    """
    if commitment is None:
        return None
    if isinstance(commitment, bytes):
        commitment = commitment.decode("utf-8", errors="ignore")
    if isinstance(commitment, str):
        raw = commitment.strip()
        if raw.startswith("http://") or raw.startswith("https://"):
            return raw.rstrip("/")
        if raw.startswith("{") and raw.endswith("}"):
            try:
                as_json = json.loads(raw)
            except json.JSONDecodeError:
                return None
            return _extract_endpoint_from_commitment(as_json)
        return None
    if isinstance(commitment, dict):
        for key in ("endpoint", "url", "base_url"):
            value = commitment.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().rstrip("/")
    return None


def read_endpoints_from_commitments(
    subtensor: bt.Subtensor,
    netuid: int,
    metagraph: bt.Metagraph,
) -> dict[int, str]:
    """
    Reads miner commitments and extracts HTTP endpoints.
    """
    endpoints: dict[int, str] = {}
    try:
        all_commitments = subtensor.get_all_commitments(netuid)
        if isinstance(all_commitments, dict):
            for uid, payload in all_commitments.items():
                try:
                    parsed_uid = int(uid)
                except (TypeError, ValueError):
                    continue
                endpoint = _extract_endpoint_from_commitment(payload)
                if endpoint:
                    endpoints[parsed_uid] = endpoint
    except Exception as exc:
        logger.warning("Failed reading all commitments: %s", exc)

    if endpoints:
        return endpoints

    # Fallback for SDK/provider combos where only per-uid fetch works.
    for uid in range(int(metagraph.n)):
        try:
            payload = subtensor.get_commitment(netuid, uid)
            endpoint = _extract_endpoint_from_commitment(payload)
            if endpoint:
                endpoints[uid] = endpoint
        except Exception:
            continue
    return endpoints


def merge_endpoint_sources(
    chain_endpoints: dict[int, str],
    override_endpoints: dict[int, str],
) -> dict[int, str]:
    merged = dict(chain_endpoints)
    merged.update(override_endpoints)
    return merged


def create_epistula_headers(hotkey: Any, request_body: bytes) -> dict[str, str]:
    nonce_ns = str(time.time_ns())
    body_hash = hashlib.sha256(request_body).hexdigest()
    message = f"{nonce_ns}.{body_hash}".encode("utf-8")
    signature = hotkey.sign(message)
    if isinstance(signature, bytes):
        signature_hex = signature.hex()
    else:
        signature_hex = bytes(signature).hex()
    return {
        "X-Epistula-Timestamp": nonce_ns,
        "X-Epistula-Signature": signature_hex,
        "X-Epistula-Hotkey": hotkey.ss58_address,
    }


def _verify_signature_with_hotkey(
    hotkey_ss58: str,
    message: bytes,
    signature_hex: str,
) -> bool:
    try:
        from substrateinterface import Keypair  # type: ignore
    except Exception:
        return False
    try:
        keypair = Keypair(ss58_address=hotkey_ss58)
        return bool(keypair.verify(message, bytes.fromhex(signature_hex)))
    except Exception:
        return False


def verify_epistula_response(
    response_body: bytes,
    response_headers: dict[str, str],
    expected_hotkey: str,
) -> tuple[bool, str]:
    timestamp = response_headers.get("X-Epistula-Timestamp", "").strip()
    signature = response_headers.get("X-Epistula-Signature", "").strip()
    hotkey = response_headers.get("X-Epistula-Hotkey", "").strip()
    if not timestamp or not signature or not hotkey:
        return False, "missing Epistula response headers"
    if hotkey != expected_hotkey:
        return False, "response hotkey does not match metagraph hotkey"
    try:
        timestamp_ns = int(timestamp)
    except ValueError:
        return False, "invalid Epistula timestamp"
    now_ns = time.time_ns()
    skew_ns = abs(now_ns - timestamp_ns)
    if skew_ns > EPISTULA_MAX_SKEW_SECONDS * 1_000_000_000:
        return False, "Epistula timestamp outside allowed skew"
    body_hash = hashlib.sha256(response_body).hexdigest()
    message = f"{timestamp}.{body_hash}".encode("utf-8")
    if not _verify_signature_with_hotkey(hotkey, message, signature):
        return False, "invalid Epistula signature"
    return True, "verified"


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
    extra_headers: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str], bytes]:
    request_body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(
        url=url,
        data=request_body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        response_body = response.read()
        response_headers = {k: v for k, v in response.headers.items()}
        parsed = json.loads(response_body.decode("utf-8"))
        return parsed, response_headers, response_body


def validate_submission_constraints(proof_code: str) -> tuple[bool, str]:
    lines = proof_code.splitlines()
    if len(lines) > MAX_PROOF_LINES:
        return False, f"too many lines ({len(lines)} > {MAX_PROOF_LINES})"
    if len(proof_code) > MAX_RESPONSE_CHARS:
        return False, f"too many chars ({len(proof_code)} > {MAX_RESPONSE_CHARS})"
    import_lines = [ln for ln in lines if ln.strip().startswith("import ")]
    if len(import_lines) > MAX_IMPORT_LINES:
        return False, f"too many imports ({len(import_lines)} > {MAX_IMPORT_LINES})"
    banned = ("unsafe", "IO.", "open scoped", "set_option")
    for token in banned:
        if token in proof_code:
            return False, f"forbidden token detected ({token})"
    return True, "ok"


def validate_lean_proof(challenge: Challenge, proof_code: str) -> tuple[bool, str]:
    """
    Returns (is_valid, compiler_output). Validation is deterministic.
    """
    candidate = proof_code.strip()
    if "theorem lemma_task" in candidate:
        full_code = f"{challenge.lean_header}\n{candidate}\n"
    else:
        # Allow miners to return only tactic/script body; validator wraps it.
        indented = "\n".join(f"  {line}" for line in candidate.splitlines())
        full_code = f"{challenge.lean_header}\n{challenge.lean_goal}{indented}\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as tf:
        tf.write(full_code)
        lean_file = tf.name

    preexec_fn = None
    if hasattr(os, "setsid"):
        def _set_limits() -> None:
            try:
                import resource  # pylint: disable=import-outside-toplevel
                bytes_limit = LEAN_MEMORY_LIMIT_MB * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
            except Exception:
                return
        preexec_fn = _set_limits
    try:
        result = subprocess.run(
            ["lean", lean_file],
            capture_output=True,
            text=True,
            timeout=LEAN_TIMEOUT_SECONDS,
            check=False,
            preexec_fn=preexec_fn,
        )
        combined = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, combined.strip()
    except FileNotFoundError:
        return False, "Lean executable not found on validator host."
    except subprocess.TimeoutExpired:
        return False, f"Lean compile timed out after {LEAN_TIMEOUT_SECONDS}s."
    finally:
        try:
            os.remove(lean_file)
        except OSError:
            pass


def elegance_score(proof_code: str) -> float:
    """
    Higher is better. Penalizes long code and extra imports.
    """
    proof = proof_code.strip()
    if not proof:
        return 0.0

    char_count = min(len(proof), MAX_PROOF_CHARS_FOR_SCORING)
    import_count = sum(
        1 for line in proof.splitlines() if line.strip().startswith("import ")
    )
    length_component = max(0.0, 1.0 - (char_count / MAX_PROOF_CHARS_FOR_SCORING))
    dependency_penalty = min(import_count * 0.05, 0.5)
    return max(0.0, length_component - dependency_penalty)


def normalize_weights(scores_by_uid: dict[int, float]) -> tuple[list[int], list[float]]:
    if not scores_by_uid:
        return [], []
    uids = sorted(scores_by_uid.keys())
    raw = [max(scores_by_uid[uid], 0.0) for uid in uids]
    total = sum(raw)
    if total <= 0:
        uniform = 1.0 / len(uids)
        return uids, [uniform for _ in uids]
    return uids, [score / total for score in raw]


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"ema_scores": {}}
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        logger.warning("State file unreadable, starting fresh.")
        return {"ema_scores": {}}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def append_round_log(entry: dict[str, Any]) -> None:
    ROUND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ROUND_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def aggregate_weighted_instant_score(
    valid_flags: list[float],
    elegance_scores: list[float],
    difficulties: list[float],
    difficulty_weighting_enabled: bool,
) -> tuple[float, float, float]:
    if not valid_flags:
        return 0.0, 0.0, 0.0
    if difficulty_weighting_enabled:
        weights = [max(d, 0.0) for d in difficulties]
    else:
        weights = [1.0 for _ in valid_flags]
    denom = sum(weights)
    if denom <= 0:
        return 0.0, 0.0, 0.0
    weighted_valid = sum(v * w for v, w in zip(valid_flags, weights)) / denom
    weighted_elegance = sum(e * w for e, w in zip(elegance_scores, weights)) / denom
    return weighted_valid * weighted_elegance, weighted_valid, weighted_elegance


def score_uid(
    uid: int,
    endpoint: str,
    challenge: Challenge,
    validator_hotkey: Any,
    expected_miner_hotkey: str,
) -> tuple[float, float, str, str]:
    payload = {
        "challenge_id": challenge.challenge_id,
        "statement": challenge.statement,
        "lean_header": challenge.lean_header,
        "lean_goal": challenge.lean_goal,
        "max_response_chars": MAX_RESPONSE_CHARS,
    }
    try:
        headers: dict[str, str] | None = None
        if EPISTULA_SIGN_REQUESTS:
            headers = create_epistula_headers(
                hotkey=validator_hotkey,
                request_body=json.dumps(payload).encode("utf-8"),
            )
        response, response_headers, response_body = post_json(
            url=f"{endpoint}/solve",
            payload=payload,
            timeout_s=REQUEST_TIMEOUT_SECONDS,
            extra_headers=headers,
        )
        if EPISTULA_VERIFY_RESPONSES:
            verified, verify_reason = verify_epistula_response(
                response_body=response_body,
                response_headers=response_headers,
                expected_hotkey=expected_miner_hotkey,
            )
            if not verified:
                if EPISTULA_STRICT_VERIFY:
                    return 0.0, 0.0, f"epistula verify failed ({verify_reason})", ""
                logger.warning("uid=%s Epistula verify failed: %s", uid, verify_reason)
        proof_code = (response.get("proof_code") or "").strip()
        if not proof_code:
            return 0.0, 0.0, "empty response", ""
        constraints_ok, constraints_reason = validate_submission_constraints(proof_code)
        if not constraints_ok:
            return 0.0, 0.0, f"constraint failed ({constraints_reason})", proof_code
        is_valid, compiler_output = validate_lean_proof(challenge, proof_code)
        if not is_valid:
            short_output = compiler_output[:180].replace("\n", " ")
            return 0.0, 0.0, f"invalid proof ({short_output})", proof_code
        return 1.0, elegance_score(proof_code), "valid", proof_code
    except urllib.error.URLError as exc:
        return 0.0, 0.0, f"http error ({exc})", ""
    except json.JSONDecodeError:
        return 0.0, 0.0, "invalid json", ""
    except Exception as exc:
        return 0.0, 0.0, f"unexpected error ({exc})", ""


@click.command()
@click.option(
    "--network",
    default=lambda: os.getenv("NETWORK", "finney"),
    help="Network to connect to (finney, test, local)",
)
@click.option(
    "--netuid",
    type=int,
    default=lambda: int(os.getenv("NETUID", "1")),
    help="Subnet netuid",
)
@click.option(
    "--coldkey",
    default=lambda: os.getenv("WALLET_NAME", "default"),
    help="Wallet name",
)
@click.option(
    "--hotkey",
    default=lambda: os.getenv("HOTKEY_NAME", "default"),
    help="Hotkey name",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=lambda: os.getenv("LOG_LEVEL", "INFO"),
    help="Logging level",
)
@click.option(
    "--ema-alpha",
    type=float,
    default=lambda: float(os.getenv("EMA_ALPHA", str(DEFAULT_EMA_ALPHA))),
    help="EMA smoothing factor in [0,1].",
)
def main(
    network: str,
    netuid: int,
    coldkey: str,
    hotkey: str,
    log_level: str,
    ema_alpha: float,
) -> None:
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    logger.info("Starting Lemma validator on network=%s netuid=%s", network, netuid)
    ema_alpha = min(max(ema_alpha, 0.0), 1.0)
    _start_metrics_server()

    last_heartbeat = [time.time()]
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=heartbeat_monitor,
        args=(last_heartbeat, stop_event),
        daemon=True,
    )
    heartbeat_thread.start()

    try:
        wallet = Wallet(name=coldkey, hotkey=hotkey)
        subtensor = bt.Subtensor(network=network)
        metagraph = bt.Metagraph(netuid=netuid, network=network)
        metagraph.sync(subtensor=subtensor)

        my_hotkey = wallet.hotkey.ss58_address
        if my_hotkey not in metagraph.hotkeys:
            logger.error("Hotkey %s is not registered on netuid %s", my_hotkey, netuid)
            return
        my_uid = metagraph.hotkeys.index(my_hotkey)
        logger.info("Validator UID=%s", my_uid)

        tempo = subtensor.get_subnet_hyperparameters(netuid).tempo
        subnet_hparams = subtensor.get_subnet_hyperparameters(netuid)
        tempo = subnet_hparams.tempo
        weights_rate_limit = int(getattr(subnet_hparams, "weights_rate_limit", tempo))
        logger.info(
            "Subnet tempo=%s weights_rate_limit=%s",
            tempo,
            weights_rate_limit,
        )
        last_weight_block = 0
        last_commitment_refresh = -COMMITMENT_REFRESH_BLOCKS
        state = load_state()
        endpoint_overrides = parse_miner_endpoints()
        endpoint_map: dict[int, str] = {}

        while True:
            try:
                metagraph.sync(subtensor=subtensor)
                current_block = subtensor.get_current_block()
                last_heartbeat[0] = time.time()
                blocks_since_last = current_block - last_weight_block

                if current_block - last_commitment_refresh >= COMMITMENT_REFRESH_BLOCKS:
                    chain_endpoints = read_endpoints_from_commitments(
                        subtensor=subtensor,
                        netuid=netuid,
                        metagraph=metagraph,
                    )
                    endpoint_map = merge_endpoint_sources(
                        chain_endpoints=chain_endpoints,
                        override_endpoints=endpoint_overrides,
                    )
                    last_commitment_refresh = current_block
                    logger.info(
                        "Endpoint refresh: chain=%s overrides=%s merged=%s",
                        len(chain_endpoints),
                        len(endpoint_overrides),
                        len(endpoint_map),
                    )

                if blocks_since_last < weights_rate_limit:
                    time.sleep(BLOCK_TIME_SECONDS)
                    continue

                challenges = choose_challenges(
                    block_number=current_block,
                    netuid=netuid,
                    batch_size=BATCH_SIZE,
                )
                logger.info("Round at block=%s batch_size=%s", current_block, len(challenges))
                with METRICS_LOCK:
                    METRICS["last_round_block"] = current_block
                    METRICS["last_round_batch_size"] = len(challenges)

                round_scores: dict[int, float] = {}
                for uid in range(int(metagraph.n)):
                    if uid == my_uid:
                        continue
                    endpoint = endpoint_map.get(uid)
                    if not endpoint:
                        continue
                    valid_flags: list[float] = []
                    elegance_scores: list[float] = []
                    challenge_difficulties: list[float] = []
                    last_reason = "no_attempt"
                    for challenge in challenges:
                        valid_flag, elegance, reason, proof_code = score_uid(
                            uid=uid,
                            endpoint=endpoint,
                            challenge=challenge,
                            validator_hotkey=wallet.hotkey,
                            expected_miner_hotkey=metagraph.hotkeys[uid],
                        )
                        valid_flags.append(valid_flag)
                        elegance_scores.append(elegance)
                        challenge_difficulties.append(challenge.difficulty)
                        last_reason = reason
                        append_round_log(
                            {
                                "timestamp_ns": time.time_ns(),
                                "block": current_block,
                                "netuid": netuid,
                                "uid": uid,
                                "endpoint": endpoint,
                                "challenge_id": challenge.challenge_id,
                                "challenge_difficulty": challenge.difficulty,
                                "valid_flag": valid_flag,
                                "elegance_score": elegance,
                                "reason": reason,
                                "proof_code": proof_code,
                            }
                        )
                        with METRICS_LOCK:
                            METRICS["attempts_total"] += 1
                            if valid_flag > 0:
                                METRICS["valid_total"] += 1
                            else:
                                METRICS["invalid_total"] += 1
                            if "epistula verify failed" in reason:
                                METRICS["epistula_fail_total"] += 1
                            if "timed out" in reason:
                                METRICS["timeout_total"] += 1
                            if "http error" in reason:
                                METRICS["http_error_total"] += 1
                    instant_score, valid_rate, avg_elegance = aggregate_weighted_instant_score(
                        valid_flags=valid_flags,
                        elegance_scores=elegance_scores,
                        difficulties=challenge_difficulties,
                        difficulty_weighting_enabled=DIFFICULTY_WEIGHTING_ENABLED,
                    )
                    prior = float(state["ema_scores"].get(str(uid), 0.0))
                    ema = ema_alpha * instant_score + (1.0 - ema_alpha) * prior
                    state["ema_scores"][str(uid)] = ema
                    round_scores[uid] = ema
                    with METRICS_LOCK:
                        METRICS["per_uid"][str(uid)] = {
                            "instant": instant_score,
                            "ema": ema,
                            "valid_rate": valid_rate,
                            "avg_elegance": avg_elegance,
                        }
                    logger.info(
                        "uid=%s instant=%.4f valid_rate=%.3f elegance=%.3f ema=%.4f reason=%s",
                        uid,
                        instant_score,
                        valid_rate,
                        avg_elegance,
                        ema,
                        last_reason,
                    )

                if not round_scores:
                    logger.warning("No miner endpoints configured/reachable. Skipping.")
                    time.sleep(BLOCK_TIME_SECONDS)
                    continue

                uids, weights = normalize_weights(round_scores)
                total = math.fsum(weights)
                logger.info("Prepared %s weights (sum=%.6f)", len(uids), total)

                success = subtensor.set_weights(
                    wallet=wallet,
                    netuid=netuid,
                    uids=uids,
                    weights=weights,
                    wait_for_inclusion=True,
                    wait_for_finalization=False,
                )
                if success:
                    save_state(state)
                    last_weight_block = current_block
                    with METRICS_LOCK:
                        METRICS["weights_set_success_total"] += 1
                    logger.info("Successfully set weights at block %s", current_block)
                else:
                    with METRICS_LOCK:
                        METRICS["weights_set_fail_total"] += 1
                    logger.warning("set_weights returned false at block %s", current_block)

                time.sleep(BLOCK_TIME_SECONDS)
            except KeyboardInterrupt:
                logger.info("Validator stopped by user.")
                break
            except Exception as exc:
                logger.error("Validator loop error: %s", exc)
                time.sleep(BLOCK_TIME_SECONDS)
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=2)


if __name__ == "__main__":
    main()
