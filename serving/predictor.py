"""
KServe custom predictor for the drone-RF classifier.

Speaks KServe's V1 predict protocol:
    POST /v1/models/drone-rf-classifier:predict
    {"instances": [<instance>, ...]}

Each <instance> is EITHER:
  - a flat JSON list of interleaved floats [I0, Q0, I1, Q1, ...] (simple,
    good for curl/manual testing, ~4x larger over the wire than binary), or
  - {"iq_base64": "<base64 of a little-endian float32 buffer, interleaved
    I0,Q0,I1,Q1,...>"}  (compact -- the recommended format for real traffic)

The pipeline assumes each capture is sampled at config.ASSUMED_SAMPLE_RATE_HZ
-- a documented, overridable assumption (20MHz commercial drone-link channel
bandwidth x 1.2 margin = 24MHz by default; see src/config.py), NOT a
measured property of any real receiver. If your SDR/receiver uses a
different rate, either resample to config.ASSUMED_SAMPLE_RATE_HZ before
sending, or override ASSUMED_CHANNEL_BANDWIDTH_HZ / BANDWIDTH_MARGIN via
environment variables and retrain/re-export the model to match. Captures
shorter than the training window are zero-padded; longer ones are
truncated to the first config.IQ_WINDOW_SAMPLES samples.
"""
from __future__ import annotations

import base64
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from kserve import Model, ModelServer

from license.verify import LicenseError, verify_or_raise
from src import config
from src.transforms import compute_log_magnitude_spectrogram

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("drone-rf-predictor")

MODEL_DIR = Path(__file__).parent / "model"


class DroneRFModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.model = None
        self.class_names: List[str] = []
        self.load()

    def load(self) -> None:
        # Node-locked license check, before touching the model at all. A hard
        # sys.exit here (rather than just leaving self.ready=False) makes a
        # missing/invalid license loudly visible as CrashLoopBackOff in
        # `kubectl get pods`, instead of a silently-NotReady pod that's easy
        # to miss. See license/verify.py for what's actually being checked.
        try:
            license_info = verify_or_raise()
        except LicenseError as exc:
            log.error("LICENSE INVALID -- refusing to start: %s", exc)
            sys.exit(1)
        log.info(
            "License OK: licensee=%s tier=%s expires=%s",
            license_info.get("licensee"), license_info.get("tier", "commercial"),
            license_info.get("expires_at", "never"),
        )

        self.model = torch.jit.load(str(MODEL_DIR / "model_scripted.pt"), map_location="cpu")
        self.model.eval()
        self.class_names = json.loads((MODEL_DIR / "class_names.json").read_text())
        self.ready = True
        log.info("Loaded model, classes=%s", self.class_names)
        log.info(
            "Assumed sample rate: %.1f MHz (%.1f MHz channel BW x %.2f margin); "
            "window=%d samples (~%.1fms at that rate)",
            config.ASSUMED_SAMPLE_RATE_HZ / 1e6, config.ASSUMED_CHANNEL_BANDWIDTH_HZ / 1e6,
            config.BANDWIDTH_MARGIN, config.IQ_WINDOW_SAMPLES, config.IQ_WINDOW_DURATION_SEC * 1000,
        )

    @staticmethod
    def _decode_instance(instance) -> torch.Tensor:
        if isinstance(instance, dict) and "iq_base64" in instance:
            raw = base64.b64decode(instance["iq_base64"])
            flat = np.frombuffer(raw, dtype="<f4")
        else:
            flat = np.asarray(instance, dtype=np.float32)

        if flat.ndim != 1 or flat.size == 0 or flat.size % 2 != 0:
            raise ValueError("Each IQ instance must be a flat, even-length, interleaved [I,Q,I,Q,...] buffer")

        iq = flat.reshape(-1, 2).T.copy()  # (2, N): row 0 = I, row 1 = Q
        n = iq.shape[1]
        target_n = config.IQ_WINDOW_SAMPLES
        if n < target_n:
            iq = np.pad(iq, ((0, 0), (0, target_n - n)))
        elif n > target_n:
            iq = iq[:, :target_n]
        return torch.from_numpy(iq)

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        instances = payload["instances"]
        specs = torch.stack([
            compute_log_magnitude_spectrogram(self._decode_instance(inst)) for inst in instances
        ])
        with torch.no_grad():
            probs = torch.softmax(self.model(specs), dim=1)

        predictions = []
        for p in probs:
            p_list = p.tolist()
            top_idx = max(range(len(p_list)), key=lambda i: p_list[i])
            predictions.append({
                "class": self.class_names[top_idx],
                "confidence": p_list[top_idx],
                "probabilities": dict(zip(self.class_names, p_list)),
            })
        return {"predictions": predictions}


if __name__ == "__main__":
    model = DroneRFModel("drone-rf-classifier")
    ModelServer().start([model])
