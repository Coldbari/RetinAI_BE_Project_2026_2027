"""Phase 0A acceptance tests — domain routing.

Guards the failure measured before routing existed: the ROP model (trained only on preterm
infants) fired on 41/45 adult eyes, and the adult-trained Glaucoma model called 16/29 preterm
infants glaucomatous. These tests assert that a routed build cannot repeat that, and that a
build with a broken routing table refuses to boot at all.

Run:  .venv/bin/python -m pytest tests/test_routing.py -v
"""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from webapp.inference import Registry, RegistryConfigError  # noqa: E402

REG = ROOT / "webapp" / "registry.yaml"


def _write(tmp_path, spec):
    p = tmp_path / "registry.yaml"
    p.write_text(yaml.safe_dump(spec))
    return str(p)


@pytest.fixture(scope="module")
def spec():
    return yaml.safe_load(REG.read_text())


# ── the routing table itself ─────────────────────────────────────────────────
def test_shipped_registry_declares_contexts(spec):
    assert spec.get("contexts"), "shipped registry.yaml has no routing table"
    # ROP-only deployment (2026-08-18 cleanup): the adult contexts left with the adult
    # models. Their absence IS the routing now - an adult context must not exist to be
    # mis-served. The multi-disease registry is parked in junk/webapp_multidisease/.
    assert set(spec["contexts"]) == {"infant"}


def test_rop_is_infant_only_and_never_in_adult_context(spec):
    rop = next(m for m in spec["models"] if m["disease"] == "ROP")
    assert rop["applies_to"] == ["infant"]
    for name, ctx in spec["contexts"].items():
        if "ROP" in ctx["runs"]:
            assert name == "infant", "ROP wired into non-infant context " + name


def test_adult_models_never_run_on_infants(spec):
    # Strongest form now available: no adult-trained model exists in this deployment.
    assert all(m["applies_to"] == ["infant"] for m in spec["models"])
    assert spec["contexts"]["infant"]["runs"] == ["ROP"]


# ── fail-fast boot validation (no weights needed) ────────────────────────────
def test_boot_fails_without_contexts(tmp_path, spec):
    bad = {"models": spec["models"]}
    with pytest.raises(RegistryConfigError, match="no 'contexts'"):
        Registry(_write(tmp_path, bad))


def test_boot_fails_on_unknown_disease_in_context(tmp_path, spec):
    bad = {"models": spec["models"],
           "contexts": {"adult": {"runs": ["DR", "Cataract"]}}}
    with pytest.raises(RegistryConfigError, match="unknown disease"):
        Registry(_write(tmp_path, bad))


def test_boot_fails_when_model_lacks_applies_to(tmp_path, spec):
    models = [dict(m) for m in spec["models"]]
    models[0].pop("applies_to", None)
    bad = {"models": models, "contexts": spec["contexts"]}
    with pytest.raises(RegistryConfigError, match="no 'applies_to'"):
        Registry(_write(tmp_path, bad))


def test_boot_fails_when_context_runs_a_model_outside_its_population(tmp_path, spec):
    """The regression that started all of this: ROP wired into an adult context.
    (The shipped registry has only ROP now, so the mis-wiring is built directly:
    an adult context running the infant-only model must fail boot.)"""
    bad = {"models": spec["models"],
           "contexts": {"adult": {"runs": ["ROP"]},
                        "infant": {"runs": ["ROP"]}}}
    with pytest.raises(RegistryConfigError, match="only applies_to"):
        Registry(_write(tmp_path, bad))


def test_shipped_registry_boots(spec):
    r = Registry(str(REG))
    assert set(r.contexts) == {"infant"}


# ── routing behaviour (weights required) ─────────────────────────────────────
@pytest.fixture(scope="module")
def registry():
    r = Registry(str(REG))
    if not any(m.loaded for m in r.models):
        pytest.skip("no model weights available")
    return r


from conftest import fundus_like  # noqa: E402


def _blank():
    """Must pass the Phase 0B gradability gate, which runs before routing."""
    return fundus_like()


def test_adult_context_is_rejected_outright(registry):
    """The 41/45 false-positive regression this file exists for is now prevented
    one layer earlier: there is no adult context to mis-serve."""
    with pytest.raises(ValueError, match="unknown patient context"):
        registry.analyze(_blank(), "adult")


def test_infant_context_never_returns_adult_findings(registry):
    res = registry.analyze(_blank(), "infant")
    assert not any(f["disease"] in ("DR", "Glaucoma") for f in res["findings"])


def test_findings_are_exactly_the_deployed_models(registry):
    """An omitted disease reads as 'nothing found'; a phantom one implies a
    model that does not exist. The ROP-only deployment must return exactly ROP."""
    res = registry.analyze(_blank(), "infant")
    assert {f["disease"] for f in res["findings"]} == {"ROP"}


def test_unknown_context_is_rejected(registry):
    with pytest.raises(ValueError, match="unknown patient context"):
        registry.analyze(_blank(), "toddler")


def test_no_default_context_exists(registry):
    """analyze() must not be callable without an explicit context."""
    with pytest.raises(TypeError):
        registry.analyze(_blank())
