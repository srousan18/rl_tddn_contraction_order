"""Corpus validation with QCEC on fixture pairs and miters."""

import csv
import shutil
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures"


def _rows(out):
    return {r["instance"]: r for r in csv.DictReader(open(out / "results.csv"))}


def test_pairs_mode(tmp_path):
    pytest.importorskip("mqt.qcec")
    from tddn.verify.validate import main

    assert main(["pairs", "--root", str(FIX / "pairs"), "--out", str(tmp_path / "ok")]) == 0
    rows = _rows(tmp_path / "ok")
    assert {r["verdict"] for r in rows.values()} == {"equivalent"} and len(rows) == 2

    bad = tmp_path / "bad_root"
    shutil.copytree(FIX / "pairs" / "alg", bad / "alg")
    (bad / "compiled").mkdir()
    shutil.copy(FIX / "pairs" / "compiled_mutated" / "ghz_5q.qasm", bad / "compiled")
    assert main(["pairs", "--root", str(bad), "--out", str(tmp_path / "bad")]) == 1
    assert _rows(tmp_path / "bad")["ghz_5q"]["verdict"] == "not_equivalent"


def test_combined_mode(tmp_path):
    pytest.importorskip("mqt.qcec")
    from tddn.verify.validate import main

    files = [str(FIX / "combined_ghz_5q.qasm"), str(FIX / "nonequiv_ghz_5q.qasm")]
    assert main(["combined", "--out", str(tmp_path), *files]) == 1
    rows = _rows(tmp_path)
    assert rows["combined_ghz_5q"]["verdict"] == "equivalent"
    assert rows["nonequiv_ghz_5q"]["verdict"] == "not_equivalent"
