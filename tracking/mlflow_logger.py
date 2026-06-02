import os
import sys
import json
import tempfile
from contextlib import contextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from config import (
    DAGSHUB_TOKEN,
    DAGSHUB_USER,
    DAGSHUB_REPO,
    MLFLOW_EXPERIMENT,
    EVAL_THRESHOLD,
    DEV_MODEL,
    PROD_MODEL,
)

_initialized = False


def _init_tracking() -> bool:
    """Point MLflow at DagsHub. Returns True if tracking is configured."""
    global _initialized
    if _initialized:
        return True

    if not (DAGSHUB_TOKEN and DAGSHUB_USER and DAGSHUB_REPO):
        print("[mlflow_logger] DAGSHUB_USER/REPO/TOKEN not set — tracking disabled")
        return False

    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    _initialized = True
    print(f"[mlflow_logger] tracking -> dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO} "
          f"(experiment={MLFLOW_EXPERIMENT})")
    return True


@contextmanager
def track_run(query: str):
    """
    Context manager that opens an MLflow run for one orchestrator query.
    Yields a `log` object with .log_state(state) which extracts metrics/params/artifacts.
    If tracking isn't configured, yields a no-op logger so callers don't need to branch.
    """
    if not _init_tracking():
        yield _NoOpLogger()
        return

    with mlflow.start_run(run_name=query[:60]):
        mlflow.log_param("query", query)
        mlflow.log_param("eval_threshold", EVAL_THRESHOLD)
        mlflow.log_param("dev_model", DEV_MODEL)
        mlflow.log_param("prod_model", PROD_MODEL)
        yield _Logger()


class _Logger:
    def log_state(self, state: dict) -> None:
        digest = state.get("digest") or {}
        eval_score = state.get("eval_score") or {}

        metrics = {
            "papers_found": len(state.get("papers") or []),
            "papers_shortlisted": len(state.get("critiqued") or []),
            "contradictions_found": len(state.get("contradictions") or []),
            "iterations": int(state.get("iterations") or 0),
            "citation_accuracy": float(eval_score.get("citation_accuracy", 0.0) or 0.0),
            "coverage": float(eval_score.get("coverage", 0.0) or 0.0),
            "coherence": float(eval_score.get("coherence", 0.0) or 0.0),
            "overall_score": float(eval_score.get("overall", 0.0) or 0.0),
            "passed": 1 if eval_score.get("passed") else 0,
        }
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            digest_path = os.path.join(tmp, "digest.json")
            with open(digest_path, "w") as f:
                json.dump(digest, f, indent=2)
            mlflow.log_artifact(digest_path)

        print(f"[mlflow_logger] logged metrics: {metrics}")


class _NoOpLogger:
    def log_state(self, _state: dict) -> None:
        return