import pytest
from pydantic import ValidationError

from qareena.config import (
    REPO_ROOT,
    Catalog,
    FieldKind,
    Settings,
    content_hash,
    load_catalog,
    load_policy,
    load_runtime,
)


@pytest.fixture
def runtime():
    return load_runtime(Settings(_env_file=None))


def test_repo_config_loads_and_cross_validates(runtime):
    assert runtime.policy.packet.auto_approve is False  # a person signs off every file
    assert set(runtime.policy.packet.required_documents) <= set(runtime.catalog.documents)
    fp = runtime.fingerprint
    assert len(fp["policy_sha256"]) == 64
    assert fp["policy_version"] == runtime.policy.policy_version


def test_policy_hash_ignores_formatting_but_not_content(runtime):
    p = runtime.policy
    assert content_hash(p) == content_hash(load_policy(REPO_ROOT / "config" / "policy.yaml"))
    changed = p.model_copy(update={"policy_version": p.policy_version + "-x"})
    assert content_hash(changed) != content_hash(p)


def test_owner_name_group_links_id_parts_and_full_names(runtime):
    group = dict.fromkeys(runtime.catalog.cross_groups()["owner_name"])
    kinds = {runtime.catalog.field(d, f).kind for d, f in group}
    assert kinds == {FieldKind.NAME_PART, FieldKind.FULL_NAME}


def test_critical_fields_cover_the_decision(runtime):
    crit = {
        (d, f) for d, doc in runtime.catalog.documents.items() for f, s in doc.fields.items() if s.critical
    }
    for needed in [
        ("iq_national_card_front", "national_id_number"),
        ("iq_national_card_back", "expiry_date"),
        ("chamber_of_commerce_id", "membership_number"),
        ("tax_card", "tax_id"),
    ]:
        assert needed in crit


def test_cloud_reference_is_refused_in_live_mode():
    with pytest.raises(ValidationError, match="only allowed in eval mode"):
        Settings(_env_file=None, mode="live", features={"cloud_reference": True})
    assert Settings(_env_file=None, mode="eval", features={"cloud_reference": True}).features.cloud_reference


def test_feature_flags_from_environment(monkeypatch):
    monkeypatch.setenv("QAREENA_FEATURES__EVIDENCE_LOOP", "false")
    assert Settings(_env_file=None).features.evidence_loop is False


def test_invalid_policy_is_rejected(tmp_path):
    bad = (
        (REPO_ROOT / "config" / "policy.yaml")
        .read_text(encoding="utf-8")
        .replace("critical: { alpha: 0.01", "critical: { alpha: 0.05")
    )
    path = tmp_path / "policy.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValidationError, match="critical alpha"):
        load_policy(path)


def test_catalog_rejects_inconsistent_fields():
    with pytest.raises(ValidationError, match="date_format"):
        Catalog.model_validate(
            {"documents": {"d": {"label_ar": "x", "fields": {"f": {"label_ar": "y", "kind": "date"}}}}}
        )
    with pytest.raises(ValidationError):
        Catalog.model_validate(
            {
                "documents": {
                    "d": {"label_ar": "x", "fields": {"f": {"label_ar": "y", "kind": "text", "surprise": 1}}}
                }
            }
        )


def test_catalog_file_loads():
    cat = load_catalog(REPO_ROOT / "config" / "documents.yaml")
    assert sum(len(d.fields) for d in cat.documents.values()) == 33
