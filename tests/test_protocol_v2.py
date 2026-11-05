import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_protocol_v2_names_the_fixed_decisions():
    text = (ROOT / "docs/paper/PROTOCOL_V2.md").read_text()
    for needle in ["Primary endpoint", "5-class map", "A − C", "Selection rule",
                   "Minimum detectable", "second opening", "retfound_green", "dinov2_s",
                   "layer_decay", "seeds 42 and 1337", "test_locked",
                   "retained to keep paper 1's counts comparable"]:
        assert needle in text, needle


def test_duplicate_exclusion_list_has_fourteen_in_pool_images():
    df = pd.read_csv(ROOT / "results/prepared/rop_staging/exclude_duplicates.csv")
    assert list(df.columns) == ["image_path", "md5", "group"]
    assert len(df) == 14 and df.group.nunique() == 7
    assert df.image_path.str.startswith("data/rop/").all()
    assert df.image_path.is_unique
