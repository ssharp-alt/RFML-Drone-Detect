"""
Node-locked license verification -- SHIPS INSIDE the serving image.

Design: the vendor holds an Ed25519 PRIVATE key offline and signs a small
JSON license payload (licensee, node fingerprint, optional expiry/features)
with license/generate_license.py (a separate, vendor-only tool that must
NEVER be copied into the served image -- see serving/Dockerfile). Only the
matching PUBLIC key ships inside the image, embedded below. Asymmetric
signing is the point: anyone with full access to the deployed container
(the licensee) can read this module and the public key, but cannot forge a
new valid license with them -- only the private key can produce a
signature this module will accept.

The "node" a license locks to is fingerprinted from /etc/machine-id (a
stable systemd-assigned identifier, standard for node-locking on Linux --
regenerated only on OS reinstall). In Kubernetes that file lives on the
HOST, not in the container by default, so the k3s InferenceService manifest
mounts it in read-only via hostPath (see serving/inferenceservice.yaml) --
this ties the license to a physical/VM node, not to a pod or container,
which is what "node-locked" is supposed to mean. That also means a license
does NOT follow a pod across a reschedule to a different node; re-issue a
license (or a multi-node one, see ALLOWED fingerprints) if that's needed.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

# Generated once via `generate_license.py keygen`; the private key that
# signed this is kept offline by the vendor and is NOT in this repo.
PUBLIC_KEY_HEX = "18ee3271d526e7e4a2d3c4cf0d49986f640d0948e1a02ea7a86974cea9ed6b01"

DEFAULT_LICENSE_PATH = os.environ.get("LICENSE_FILE", "/etc/drone-rf-license/license.json")
DEFAULT_MACHINE_ID_PATH = os.environ.get("LICENSE_MACHINE_ID_PATH", "/etc/machine-id")


class LicenseError(Exception):
    pass


def compute_node_fingerprint(machine_id_path: str) -> str:
    path = Path(machine_id_path)
    if not path.exists():
        raise LicenseError(
            f"Cannot compute node fingerprint: {machine_id_path} not found. "
            "In Kubernetes this must be mounted in read-only via hostPath "
            "(see serving/inferenceservice.yaml)."
        )
    raw = path.read_text().strip()
    if not raw:
        raise LicenseError(f"{machine_id_path} is empty")
    return hashlib.sha256(raw.encode()).hexdigest()


def _canonical_bytes(payload: dict) -> bytes:
    # sort_keys + no extra whitespace => byte-identical between signer and
    # verifier regardless of dict insertion order or formatting.
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def verify_or_raise(license_path: str | None = None, machine_id_path: str | None = None) -> dict:
    """Returns the verified license payload dict, or raises LicenseError
    with a human-readable reason. Call this once at predictor startup."""
    license_path = license_path or DEFAULT_LICENSE_PATH
    machine_id_path = machine_id_path or DEFAULT_MACHINE_ID_PATH

    if PUBLIC_KEY_HEX.startswith("REPLACE_"):
        raise LicenseError(
            "verify.py's PUBLIC_KEY_HEX was never set -- run "
            "`generate_license.py keygen` and paste the printed public key "
            "into license/verify.py before building the serving image."
        )

    path = Path(license_path)
    if not path.exists():
        raise LicenseError(
            f"License file not found at {license_path}. Mount it as a Secret "
            "(see serving/inferenceservice.yaml) or set LICENSE_FILE."
        )
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise LicenseError(f"License file at {license_path} is not valid JSON: {exc}") from exc

    signature_hex = doc.get("signature")
    if not signature_hex:
        raise LicenseError("License file has no 'signature' field")
    payload = {k: v for k, v in doc.items() if k != "signature"}

    public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(PUBLIC_KEY_HEX))
    try:
        public_key.verify(bytes.fromhex(signature_hex), _canonical_bytes(payload))
    except InvalidSignature as exc:
        raise LicenseError(
            "License signature is invalid -- the file was modified after "
            "signing, or was not signed with the matching private key."
        ) from exc
    except ValueError as exc:
        raise LicenseError(f"Malformed signature/payload: {exc}") from exc

    expires_at = payload.get("expires_at")
    if expires_at:
        expiry = datetime.fromisoformat(expires_at)
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expiry:
            raise LicenseError(f"License expired at {expires_at}")

    allowed = payload.get("node_fingerprints")
    if not allowed:
        raise LicenseError("License payload has no 'node_fingerprints' list")
    actual_fingerprint = compute_node_fingerprint(machine_id_path)
    if actual_fingerprint not in allowed:
        raise LicenseError(
            f"This node's fingerprint ({actual_fingerprint[:12]}...) is not in the "
            f"license's allowed list. This license is bound to a different node."
        )

    return payload


if __name__ == "__main__":
    import sys

    try:
        payload = verify_or_raise()
    except LicenseError as exc:
        print(f"LICENSE INVALID: {exc}", file=sys.stderr)
        sys.exit(1)
    print("License OK:", json.dumps(payload, indent=2))
