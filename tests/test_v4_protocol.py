from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_amendment_5_is_pre_registered():
    text = (ROOT / "docs/paper/PROTOCOL_V2.md").read_text()
    for needle in ("Amendment 5", "fold_loso_ostrava", "fold_loso_ropvl", "ema_decay 0.999",
                   "head warm-up", "≥ +0.010", "H − B", "I − B", "I − H", "test_locked"):
        assert needle in text, needle
    assert text.index("Amendment 4") < text.index("Amendment 5")
