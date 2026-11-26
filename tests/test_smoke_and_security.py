"""Smoke, structural, monkey-patch order, and security sanitization tests."""

from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "portfolio_optimization.ipynb"


def test_clean_headless_import_without_secrets_or_side_effects():
    """Importing portfolio_optimization must not require keys, network, or GUI."""
    env = os.environ.copy()
    for var in ("ALPHA_VANTAGE_API_KEYS", "ALPHA_VANTAGE_API_KEY", "FRED_API_KEY", "GEMINI_API_KEY"):
        env.pop(var, None)
    env["PYTHONPATH"] = str(SRC_DIR)

    code = (
        "import sys, portfolio_optimization as po; "
        "assert 'ipywidgets' not in sys.modules; "
        "assert 'tkinter' not in sys.modules; "
        "assert po.state.resultado_analisis is None; "
        "assert po.state.prima_riesgo_mas_reciente is None; "
        "assert po.state.country_risk_premia == {}; "
        "assert po.state.activos_portfolio == []; "
        "assert po.get_alpha_vantage_api_keys() == []; "
        "assert po.get_fred_api_key() == ''; "
        "assert po.get_gemini_api_key() == ''; "
        "print('OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "OK"


def test_redefinitions_and_monkey_patches_match_notebook_order():
    """Verify Cell 13 analyze_portfolio override and Cell 16/17 monkey patches."""
    import portfolio_optimization as po

    # Cell 13 override adds shrinkage_method='ledoit_wolf'
    sig = inspect.signature(po.analyze_portfolio)
    assert "shrinkage_method" in sig.parameters
    assert sig.parameters["shrinkage_method"].default == "ledoit_wolf"

    # Cell 16 monkey patches on PortfolioPredictor
    assert po.PortfolioPredictor._train_xgb is po._train_xgb_robust
    assert po.PortfolioPredictor.train_short_term_model is po._train_short_term_model_robust
    assert po.PortfolioPredictor.train_long_term_model is po._train_long_term_model_robust

    # Cell 17 monkey patch on PortfolioProcessor
    assert po.PortfolioProcessor._process_macro is po._process_macro_with_lag
    proc = po.PortfolioProcessor(po.IngestedData())
    assert proc._process_macro("XLK").empty


def test_alpha_vantage_key_rotation_and_missing_key_error(monkeypatch):
    """Preserve exact round-robin order when configured; fail explicitly when missing."""
    import portfolio_optimization as po
    from portfolio_optimization import config, state

    monkeypatch.setattr(config, "ALPHA_VANTAGE_API_KEYS", [])
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEYS", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

    ingestor = po.DataIngestor(["AAPL"])
    with pytest.raises(RuntimeError, match="ALPHA_VANTAGE_API_KEYS"):
        ingestor._get_next_alpha_vantage_api_key()

    monkeypatch.setenv("ALPHA_VANTAGE_API_KEYS", "FAKE_K1, FAKE_K2, FAKE_K3")
    monkeypatch.setattr(state, "_indice_api_key", 0)
    seq = [ingestor._get_next_alpha_vantage_api_key() for _ in range(7)]
    assert seq == ["FAKE_K1", "FAKE_K2", "FAKE_K3", "FAKE_K1", "FAKE_K2", "FAKE_K3", "FAKE_K1"]


def test_sanitized_notebook_has_no_outputs_widgets_or_secrets():
    """Published notebook adapter must have zero outputs, execution counts, or widget states."""
    assert NOTEBOOK_PATH.exists()
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert "widgets" not in nb.get("metadata", {})
    assert "colab" not in nb.get("metadata", {})
    assert len(nb["cells"]) > 0

    all_sources = []
    for cell in nb["cells"]:
        assert not cell.get("attachments")
        if cell["cell_type"] == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
        all_sources.append("".join(cell.get("source", [])))

    joined = "\n".join(all_sources)
    assert "import portfolio_optimization" in joined
    assert "!pip install" not in joined


def test_repository_contains_no_hardcoded_secrets():
    """Scan all tracked/project files for real API key patterns or .env files."""
    assert not (REPO_ROOT / ".env").exists()
    assert (REPO_ROOT / ".env.example").exists()
    assert (REPO_ROOT / ".gitignore").exists()

    secret_regexes = [
        re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
        re.compile(r"ALPHA_VANTAGE_API_KEYS\s*=\s*\[\s*['\"][A-Z0-9]{10,}['\"]"),
        re.compile(r"FRED_API_KEY\s*=\s*['\"][0-9a-f]{32}['\"]"),
        re.compile(r"GEMINI_API_KEY\s*=\s*['\"]AIza"),
    ]

    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache", ".venv"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for rx in secret_regexes:
            assert rx.search(text) is None, f"Potential secret pattern matched in {path}"
