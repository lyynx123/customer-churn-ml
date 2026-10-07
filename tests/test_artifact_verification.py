"""Tests for Phase 16 - Artifact Verification and CI Hardening."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_inference_artifacts import (
    VerificationError,
    verify_artifacts,
    verify_files_exist,
    verify_metadata_json,
    verify_pipeline_loads,
    verify_schema_json,
    verify_sha256_match,
)


@pytest.fixture
def artifacts_dir():
    return Path(__file__).resolve().parents[1] / "models"


@pytest.fixture
def temp_artifacts(artifacts_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for name in ["final_pipeline.joblib", "inference_schema.json", "inference_metadata.json"]:
            shutil.copy(artifacts_dir / name, tmpdir / name)
        yield tmpdir


def test_verify_passes_with_real_artifacts(artifacts_dir):
    result = verify_artifacts(artifacts_dir)
    assert result["all_passed"] is True


def test_verify_schema_structure(artifacts_dir):
    schema = verify_schema_json(artifacts_dir)
    assert schema["version"] == "1.0"
    assert len(schema["numeric_features"]) == 4
    assert len(schema["categorical_features"]) == 15
    assert len(schema["feature_order"]) == 19
    assert set(schema["forbidden_fields"]) == {"customerID", "Churn"}


def test_verify_metadata_threshold(artifacts_dir):
    metadata = verify_metadata_json(artifacts_dir)
    assert metadata["production_threshold"] == 0.28
    assert metadata["threshold_metadata"]["value"] == 0.28
    assert metadata["threshold_metadata"]["source_phase"] == "Phase 11"
    assert metadata["threshold_metadata"]["selection_method"] == "Business cost optimization on OOF probabilities"


def test_verify_pipeline_loads(artifacts_dir):
    pipeline = verify_pipeline_loads(artifacts_dir)
    assert pipeline is not None
    assert pipeline.named_steps["model"].n_estimators == 500
    assert pipeline.named_steps["model"].max_depth == 10


def test_missing_pipeline_fails(temp_artifacts):
    (temp_artifacts / "final_pipeline.joblib").unlink()
    with pytest.raises(VerificationError, match="Missing required artifact"):
        verify_files_exist(temp_artifacts)


def test_missing_schema_fails(temp_artifacts):
    (temp_artifacts / "inference_schema.json").unlink()
    with pytest.raises(VerificationError, match="Missing required artifact"):
        verify_files_exist(temp_artifacts)


def test_missing_metadata_fails(temp_artifacts):
    (temp_artifacts / "inference_metadata.json").unlink()
    with pytest.raises(VerificationError, match="Missing required artifact"):
        verify_files_exist(temp_artifacts)


def test_corrupted_pipeline_sha_mismatch_fails(temp_artifacts):
    metadata = verify_metadata_json(temp_artifacts)
    pipeline_path = temp_artifacts / "final_pipeline.joblib"
    with open(pipeline_path, "ab") as f:
        f.write(b"\x00")
    with pytest.raises(VerificationError, match="SHA256 mismatch"):
        verify_sha256_match(temp_artifacts, metadata)


def test_modified_threshold_fails(temp_artifacts):
    metadata_path = temp_artifacts / "inference_metadata.json"
    with open(metadata_path) as f:
        meta = json.load(f)
    meta["production_threshold"] = 0.50
    meta["threshold_metadata"]["value"] = 0.50
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
    with pytest.raises(VerificationError, match="production_threshold mismatch"):
        verify_metadata_json(temp_artifacts)


def test_invalid_sha_in_metadata_fails(temp_artifacts):
    metadata_path = temp_artifacts / "inference_metadata.json"
    with open(metadata_path) as f:
        meta = json.load(f)
    meta["artifact_integrity"]["pipeline_sha256"] = "0" * 64
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
    metadata = verify_metadata_json(temp_artifacts)
    with pytest.raises(VerificationError, match="SHA256 mismatch"):
        verify_sha256_match(temp_artifacts, metadata)


def test_malformed_schema_fails(temp_artifacts):
    (temp_artifacts / "inference_schema.json").write_text("not valid json {")
    with pytest.raises(VerificationError, match="not valid JSON"):
        verify_schema_json(temp_artifacts)


def test_malformed_metadata_fails(temp_artifacts):
    (temp_artifacts / "inference_metadata.json").write_text("not valid json {")
    with pytest.raises(VerificationError, match="not valid JSON"):
        verify_metadata_json(temp_artifacts)


def test_verify_artifacts_fails_on_corrupted_sha(temp_artifacts):
    metadata_path = temp_artifacts / "inference_metadata.json"
    with open(metadata_path) as f:
        meta = json.load(f)
    meta["artifact_integrity"]["pipeline_sha256"] = "0" * 64
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
    with pytest.raises(VerificationError, match="SHA256 mismatch"):
        verify_artifacts(temp_artifacts)


def test_schema_missing_required_key_fails(temp_artifacts):
    schema_path = temp_artifacts / "inference_schema.json"
    with open(schema_path) as f:
        schema = json.load(f)
    del schema["numeric_features"]
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    with pytest.raises(VerificationError, match="missing required key"):
        verify_schema_json(temp_artifacts)


def test_cli_smoke_test(artifacts_dir):
    project_root = Path(__file__).resolve().parents[1]
    fixture = project_root / "tests" / "fixtures" / "portable_smoke_input.csv"
    assert fixture.exists(), f"Smoke fixture not found: {fixture}"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_path = Path(f.name)
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "customer_churn.portable_predict",
                "--input",
                str(fixture),
                "--artifacts",
                str(artifacts_dir),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            check=False,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        content = output_path.read_text()
        assert "churn_prediction" in content
        assert "churn_probability" in content
        assert "Predictions saved" in result.stdout
        assert "Threshold: 0.28" in result.stdout
    finally:
        if output_path.exists():
            output_path.unlink()


def test_cli_rejects_invalid_input(artifacts_dir):
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("gender,SeniorCitizen\nFemale,0\n")
        input_path = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_path = Path(f.name)
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "customer_churn.portable_predict",
                "--input",
                str(input_path),
                "--artifacts",
                str(artifacts_dir),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            check=False,
        )
        assert result.returncode != 0
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def test_verify_script_cli_passes(artifacts_dir):
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/verify_inference_artifacts.py",
            "--artifacts-dir",
            str(artifacts_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        check=False,
    )
    assert result.returncode == 0, f"Verify script failed: {result.stderr}"
    assert "All artifact verification checks PASSED" in result.stdout


def test_verify_script_cli_fails_on_missing_artifact(temp_artifacts):
    (temp_artifacts / "final_pipeline.joblib").unlink()
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/verify_inference_artifacts.py",
            "--artifacts-dir",
            str(temp_artifacts),
        ],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        check=False,
    )
    assert result.returncode != 0
    assert "Verification FAILED" in result.stderr or "Missing required artifact" in result.stderr
