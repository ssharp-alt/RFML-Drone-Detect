"""
VENDOR-ONLY license tooling -- must NEVER be copied into the served image
(see serving/Dockerfile, which only copies license/verify.py). Keep the
private key file this produces somewhere safe and offline; anyone who has
it can mint valid licenses for any node.

Workflow:
  1. One-time setup:
       .venv/Scripts/python.exe -m license.generate_license keygen
     Paste the printed public key hex into license/verify.py's
     PUBLIC_KEY_HEX before building the serving image. Keep private_key.pem
     out of the image and out of source control (already gitignored).

  2. Get the target node's fingerprint. Either run this ON the node:
       .venv/Scripts/python.exe -m license.generate_license fingerprint
     (reads /etc/machine-id there), or have the customer send you their
     /etc/machine-id content and hash it yourself:
       .venv/Scripts/python.exe -m license.generate_license fingerprint --machine-id-value <contents>

  3. Issue a license:
       .venv/Scripts/python.exe -m license.generate_license issue \
         --private-key private_key.pem --licensee "Acme Drones Inc" \
         --node-fingerprint <hex-from-step-2> --expires 2027-01-01 \
         --out license.json
     Ship license.json to the customer; they load it as a k8s Secret (see
     serving/inferenceservice.yaml).

  4. Or issue a free trial (same mechanism, just short-lived and tagged):
       .venv/Scripts/python.exe -m license.generate_license issue \
         --private-key private_key.pem --licensee "Prospect Co (trial)" \
         --node-fingerprint <hex> --trial-days 14 --out trial_license.json
     Good for letting a prospective customer run their own integration
     testing against a real node-locked deployment before buying -- it
     just stops working when the trial expires, same as any other license.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def cmd_keygen(args: argparse.Namespace) -> None:
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    Path(args.out).write_bytes(private_bytes)
    public_hex = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw,
    ).hex()
    print(f"Wrote private key to {args.out} -- keep this OFFLINE and out of the image/repo.")
    print(f"\nPublic key (paste into license/verify.py's PUBLIC_KEY_HEX):\n{public_hex}")


def cmd_fingerprint(args: argparse.Namespace) -> None:
    if args.machine_id_value is not None:
        raw = args.machine_id_value.strip()
    else:
        raw = Path(args.machine_id_file).read_text().strip()
    print(hashlib.sha256(raw.encode()).hexdigest())


def cmd_issue(args: argparse.Namespace) -> None:
    if args.trial_days and args.expires:
        raise SystemExit("Pass either --trial-days or --expires, not both")

    private_key = serialization.load_pem_private_key(Path(args.private_key).read_bytes(), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise SystemExit("Private key file is not an Ed25519 key")

    tier = "trial" if args.trial_days else args.tier
    payload = {
        "licensee": args.licensee,
        "tier": tier,
        "issued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "node_fingerprints": args.node_fingerprint,  # list: supports multi-node licenses
        "features": args.feature or ["predict"],
    }
    if args.trial_days:
        payload["expires_at"] = (datetime.now(timezone.utc) + timedelta(days=args.trial_days)).isoformat(timespec="seconds")
    elif args.expires:
        payload["expires_at"] = datetime.strptime(args.expires, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ).isoformat(timespec="seconds")

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    signature_hex = private_key.sign(canonical).hex()

    doc = dict(payload)
    doc["signature"] = signature_hex
    Path(args.out).write_text(json.dumps(doc, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(doc, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    keygen_p = sub.add_parser("keygen", help="Generate a new Ed25519 vendor keypair")
    keygen_p.add_argument("--out", default="private_key.pem")
    keygen_p.set_defaults(func=cmd_keygen)

    fp_p = sub.add_parser("fingerprint", help="Compute a node fingerprint from /etc/machine-id")
    fp_p.add_argument("--machine-id-file", default="/etc/machine-id")
    fp_p.add_argument("--machine-id-value", default=None, help="Hash this literal value instead of reading a file")
    fp_p.set_defaults(func=cmd_fingerprint)

    issue_p = sub.add_parser("issue", help="Sign a new license file")
    issue_p.add_argument("--private-key", required=True)
    issue_p.add_argument("--licensee", required=True)
    issue_p.add_argument("--node-fingerprint", required=True, nargs="+", help="One or more node fingerprints (multi-node license)")
    issue_p.add_argument("--tier", choices=["trial", "commercial"], default="commercial")
    issue_p.add_argument("--expires", default=None, help="YYYY-MM-DD, omit for a perpetual license")
    issue_p.add_argument("--trial-days", type=int, default=None, help="Shortcut: sets tier=trial and expires N days from now")
    issue_p.add_argument("--feature", action="append", default=None)
    issue_p.add_argument("--out", default="license.json")
    issue_p.set_defaults(func=cmd_issue)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
