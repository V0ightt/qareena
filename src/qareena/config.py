"""Configuration: runtime settings, compliance policy and the document schema.

Three kinds of switches, deliberately kept apart (see docs/DESIGN.md §17):

* ``Settings``  — operational and engineering knobs from the environment (``QAREENA_*``):
                  model names, paths, feature flags. Feature flags also drive ablations.
* ``Policy``    — compliance decisions in ``config/policy.yaml``. Identified by a content
                  hash that is stamped on every decision. Never toggled at runtime.
* ``Catalog``   — document types and their fields in ``config/documents.yaml``.

``load_runtime()`` returns all three plus a ``fingerprint`` for traces and reports.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- document catalog -----------------------------------------------------------------


class FieldKind(StrEnum):
    NAME_PART = "name_part"
    FULL_NAME = "full_name"
    TEXT = "text"
    DIGITS = "digits"
    CODE = "code"
    DATE = "date"
    ENUM = "enum"
    MRZ_LINE = "mrz_line"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FieldSpec(_Frozen):
    label_ar: str
    kind: FieldKind
    required: bool = False
    critical: bool = False
    pattern: str | None = None
    values: tuple[str, ...] | None = None
    date_format: str | None = None
    cross: str | None = None
    aliases: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> FieldSpec:
        if self.pattern is not None:
            re.compile(self.pattern)
        if (self.kind == FieldKind.ENUM) != (self.values is not None):
            raise ValueError("enum fields need 'values', and only enum fields may have them")
        if (self.kind == FieldKind.DATE) != (self.date_format is not None):
            raise ValueError("date fields need 'date_format', and only date fields may have it")
        return self


class DocumentSpec(_Frozen):
    label_ar: str
    fields: dict[str, FieldSpec]

    @field_validator("fields")
    @classmethod
    def _non_empty(cls, v: dict[str, FieldSpec]) -> dict[str, FieldSpec]:
        if not v:
            raise ValueError("a document needs at least one field")
        return v


class Catalog(_Frozen):
    documents: dict[str, DocumentSpec]

    def field(self, doc_type: str, name: str) -> FieldSpec:
        return self.documents[doc_type].fields[name]

    def printed_labels(self) -> frozenset[str]:
        """Every label string the catalog knows (Arabic labels and aliases), for label-echo checks."""
        return frozenset(
            label
            for doc in self.documents.values()
            for spec in doc.fields.values()
            for label in (spec.label_ar, *spec.aliases)
        )

    def cross_groups(self) -> dict[str, list[tuple[str, str]]]:
        """Map each cross-document group to its (doc_type, field) members, in schema order."""
        groups: dict[str, list[tuple[str, str]]] = {}
        for doc_type, doc in self.documents.items():
            for name, spec in doc.fields.items():
                if spec.cross:
                    groups.setdefault(spec.cross, []).append((doc_type, name))
        return groups


# --- compliance policy -----------------------------------------------------------------


class PacketPolicy(_Frozen):
    required_documents: tuple[str, ...]
    retakes_per_document: int = Field(ge=0, le=10)
    auto_approve: bool = False
    audit_sample_rate: float = Field(ge=0.0, le=1.0)


class ThresholdClass(_Frozen):
    alpha: float = Field(gt=0.0, lt=0.5)
    confidence: float = Field(gt=0.5, lt=1.0)
    min_accepted: int = Field(ge=1)


class ThresholdPolicy(_Frozen):
    grid: tuple[float, ...]
    critical: ThresholdClass
    other: ThresholdClass

    @field_validator("grid")
    @classmethod
    def _grid(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if not v or any(not 0.0 < t <= 1.0 for t in v):
            raise ValueError("grid values must be in (0, 1]")
        if list(v) != sorted(set(v), reverse=True):
            raise ValueError("grid must be strictly decreasing (strict -> loose)")
        return v

    @model_validator(mode="after")
    def _critical_is_stricter(self) -> ThresholdPolicy:
        if self.critical.alpha > self.other.alpha:
            raise ValueError("critical alpha must not be looser than the other-field alpha")
        return self


class NamePolicy(_Frozen):
    missing_laqab: Literal["note", "check"]
    missing_grandfather: Literal["note", "check"]
    sound_alike: Literal["check"]  # never auto-matched: different letters can be different people


class LoopPolicy(_Frozen):
    max_actions_per_field: int = Field(ge=0, le=5)
    packet_budget_seconds: int = Field(ge=10, le=600)


class RetentionPolicy(_Frozen):
    image_hours: int = Field(ge=1)


class ReaderPolicy(_Frozen):
    """How a document is shown to the VLM. It changes what the model reads, so it is certified
    policy, not a setting (``tools/spike/business_ab.py`` measured the effect of size)."""

    long_side_px: int = Field(ge=640, le=2048)  # long side of the image sent to the model
    long_side_px_by_document: dict[str, int] = Field(default_factory=dict)
    retry_long_side_px: tuple[int, ...] = Field(min_length=2)  # first that differs is used once

    @field_validator("long_side_px_by_document")
    @classmethod
    def _sizes_in_range(cls, v: dict[str, int]) -> dict[str, int]:
        if any(not 640 <= px <= 2048 for px in v.values()):
            raise ValueError("long sides must be within 640-2048 px")
        return v

    def long_side_for(self, doc_type: str) -> int:
        return self.long_side_px_by_document.get(doc_type, self.long_side_px)

    def retry_for(self, doc_type: str) -> int:
        first = self.long_side_for(doc_type)
        return next(px for px in self.retry_long_side_px if px != first)


class Policy(_Frozen):
    policy_version: str = Field(min_length=1)
    packet: PacketPolicy
    thresholds: ThresholdPolicy
    names: NamePolicy
    evidence_loop: LoopPolicy
    reader: ReaderPolicy
    retention: RetentionPolicy


# --- runtime settings ------------------------------------------------------------------


class Features(_Frozen):
    """Engineering switches. Each is recorded in every trace; ablations flip them."""

    evidence_loop: bool = True
    ocr_grounding: bool = True
    two_source_rule: bool = True
    pdf_intake: bool = True
    live_capture_guidance: bool = False
    cloud_reference: bool = False  # offline benchmark only; refused in live mode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QAREENA_", env_file=".env", env_nested_delimiter="__", extra="ignore"
    )

    mode: Literal["live", "eval"] = "live"
    ollama_url: str = "http://127.0.0.1:11434"
    vlm_model: str = "qwen3.5:4b"
    ocr_engine: Literal["rapidocr_ppocrv5_ar", "tesseract"] = "rapidocr_ppocrv5_ar"
    config_dir: Path = REPO_ROOT / "config"
    data_dir: Path = REPO_ROOT / "var"
    data_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    features: Features = Features()

    @model_validator(mode="after")
    def _no_cloud_in_live(self) -> Settings:
        if self.mode == "live" and self.features.cloud_reference:
            raise ValueError(
                "cloud_reference is only allowed in eval mode: no document leaves the machine live"
            )
        return self


# --- loading -----------------------------------------------------------------------------


def content_hash(model: BaseModel) -> str:
    """SHA-256 of the canonical JSON of a model. Formatting and comments do not change it."""
    canonical = json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_policy(path: Path) -> Policy:
    return Policy.model_validate(_read_yaml(path))


def load_catalog(path: Path) -> Catalog:
    return Catalog.model_validate(_read_yaml(path))


class Runtime(BaseModel):
    """Everything a pipeline run depends on, loaded and cross-validated once."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    settings: Settings
    policy: Policy
    catalog: Catalog

    @model_validator(mode="after")
    def _cross_validate(self) -> Runtime:
        missing = [d for d in self.policy.packet.required_documents if d not in self.catalog.documents]
        if missing:
            raise ValueError(f"policy requires documents not in the catalog: {missing}")
        return self

    @cached_property
    def policy_sha256(self) -> str:
        return content_hash(self.policy)

    @cached_property
    def catalog_sha256(self) -> str:
        return content_hash(self.catalog)

    @property
    def fingerprint(self) -> dict[str, Any]:
        """Stamped on every trace and evaluation report so any result can be reproduced."""
        return {
            "policy_version": self.policy.policy_version,
            "policy_sha256": self.policy_sha256,
            "catalog_sha256": self.catalog_sha256,
            "mode": self.settings.mode,
            "vlm_model": self.settings.vlm_model,
            "ocr_engine": self.settings.ocr_engine,
            "features": self.settings.features.model_dump(),
        }


def load_runtime(settings: Settings | None = None) -> Runtime:
    settings = settings or Settings()
    return Runtime(
        settings=settings,
        policy=load_policy(settings.config_dir / "policy.yaml"),
        catalog=load_catalog(settings.config_dir / "documents.yaml"),
    )
