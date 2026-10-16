"""Validate the reproducible IPN inputs used by the symbolic branch.

The symbolic pipeline has three contracts which are easy to violate when data
is copied between machines:

* the three split files must have the expected counts, labels, and (when
  recorded) SHA-256 digests;
* split membership must be subject-disjoint and the known ``1CM42_3`` subject
  must remain excluded when metadata is available; and
* every expected instance graph must be present and readable, alongside the
  complete set of 13 template graphs.

This module intentionally uses only the standard library.  It can be used as
both a command-line check and a small library in tests or data-preparation
scripts.  Metadata is deliberately parsed permissively because older exports
have used ``clip_counts``, ``split_counts``, ``hashes``, and per-split objects
for the same information.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "external_data" / "ipn_processed"

LABELS = (
    "B0A",
    "B0B",
    "G01",
    "G02",
    "G03",
    "G04",
    "G05",
    "G06",
    "G07",
    "G08",
    "G09",
    "G10",
    "G11",
)
LABEL_IDS = {name: index for index, name in enumerate(LABELS)}
EXPECTED_LABEL_IDS = frozenset(LABEL_IDS.values())

# Counts from the agreed IPN 13-class manifest (2,443 train, 589 validation,
# and 1,101 test clips).  A metadata file with explicit counts takes priority;
# callers can also pass a smaller expected mapping for fixture/smoke datasets.
DEFAULT_SPLIT_COUNTS = {"train": 2443, "val": 589, "test": 1101}
DEFAULT_TEMPLATE_COUNT = 13
EXPECTED_EXCLUDED_SUBJECT = "1CM42_3"

_SPLITS = ("train", "val", "test")
_SPLIT_ALIASES = {
    "train": frozenset(("train", "train.txt", "train_manifest", "train_manifest.csv")),
    "val": frozenset(("val", "val.txt", "val_manifest", "val_manifest.csv", "validation", "validation.txt")),
    "test": frozenset(("test", "test.txt", "test_manifest", "test_manifest.csv")),
}
_COUNT_KEYS = frozenset(("count", "counts", "clips", "clip_count", "samples", "sample_count", "rows", "lines", "n"))
_HASH_KEYS = frozenset(("hash", "sha256", "sha_256", "checksum", "digest"))
_METADATA_NAMES = (
    "split_metadata.json",
    "manifest_metadata.json",
    "input_metadata.json",
    "metadata.json",
    "split_subjects.json",
    "checksums.json",
    "hashes.json",
    "input_hashes.json",
)


class ValidationError(ValueError):
    """Raised by ``raise_if_failed`` when symbolic inputs are invalid."""


@dataclass
class ValidationReport:
    """Structured result returned by :func:`validate_symbolic_inputs`."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ValidationError("; ".join(self.errors))

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ok"] = self.ok
        return result


@dataclass(frozen=True)
class SplitEntry:
    source: str
    label_id: int
    line_number: int


def _canonical_split(value: Any) -> str | None:
    """Map common split/file spellings to ``train``, ``val`` or ``test``."""

    if value is None:
        return None
    text = str(value).strip().lower().replace("\\", "/")
    name = Path(text).name
    stem = Path(name).stem
    for split, aliases in _SPLIT_ALIASES.items():
        if text in aliases or name in aliases or stem in aliases:
            return split
    if stem in {"validation", "valid"}:
        return "val"
    return None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _coerce_int(value: Any) -> int | None:
    if _is_int(value):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    return None


def _normalise_hash(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in _HASH_KEYS:
            if key in value:
                return _normalise_hash(value[key])
        return None
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text if re.fullmatch(r"[0-9a-f]{64}", text) else None


def _read_json(path: Path) -> Any:
    raw = path.read_bytes()
    if not raw.strip():
        raise ValueError("file is empty")
    return json.loads(raw.decode("utf-8-sig"))


def _read_metadata(path: Path) -> Any:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    return _read_json(path)


def discover_metadata(data_root: Path, metadata_path: Path | None = None) -> Path | None:
    """Return an explicit or conventional metadata path, if one exists."""

    if metadata_path is not None:
        return metadata_path
    for name in _METADATA_NAMES:
        candidate = data_root / name
        if candidate.is_file():
            return candidate
    # The source annotation metadata is useful for the exclusion check when
    # the processed-data export did not carry its own metadata copy.
    source_metadata = data_root.parent / "ipn_full" / "annotations" / "metadata.csv"
    if source_metadata.is_file():
        return source_metadata
    return None


def read_split_file(path: Path) -> list[SplitEntry]:
    """Read a standard ``source 0 label_id`` split file.

    ``rsplit`` permits paths containing spaces.  A malformed line is raised
    here rather than silently excluded, because silently dropping one line
    would make a coverage check appear successful.
    """

    entries: list[SplitEntry] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"{path}: line {line_number} must contain source, placeholder, and label ID")
        source, _placeholder, label_text = parts
        label_id = _coerce_int(label_text)
        if label_id is None:
            raise ValueError(f"{path}: line {line_number} has a non-integer label ID {label_text!r}")
        entries.append(SplitEntry(source=source, label_id=label_id, line_number=line_number))
    return entries


def _iter_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _iter_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_mappings(child)


def _value_for_split(mapping: Mapping[str, Any], split: str) -> Any:
    for key, value in mapping.items():
        if _canonical_split(key) == split:
            return value
    return None


def _count_from_value(value: Any) -> int | None:
    direct = _coerce_int(value)
    if direct is not None:
        return direct
    if isinstance(value, Mapping):
        # Prefer explicit count fields.  Do not interpret video_counts as clip
        # counts; the latter are a distinct metadata field in split_subjects.
        for key in _COUNT_KEYS:
            if key in value:
                count = _coerce_int(value[key])
                if count is not None:
                    return count
    return None


def extract_split_counts(metadata: Any) -> dict[str, int]:
    """Extract expected split-line counts from common metadata layouts."""

    result: dict[str, int] = {}
    if isinstance(metadata, list):
        for row in metadata:
            if not isinstance(row, Mapping):
                continue
            split = _canonical_split(row.get("split") or row.get("Set") or row.get("set"))
            if split is None:
                continue
            count = _count_from_value(row.get("count") or row.get("clips") or row.get("samples"))
            if count is not None:
                result[split] = count
        return result

    if not isinstance(metadata, Mapping):
        return result

    # ``clip_counts`` is the output field of create_ipn_manifest.py.  Other
    # names cover hand-written manifests and previous exports.
    for key in ("clip_counts", "split_counts", "sample_counts", "line_counts", "counts"):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            for split in _SPLITS:
                count = _count_from_value(_value_for_split(value, split))
                if count is not None:
                    result[split] = count

    # Also support {"splits": {"train": {"count": ...}}} and direct
    # per-split objects.
    splits_value = metadata.get("splits")
    if isinstance(splits_value, Mapping):
        for split in _SPLITS:
            count = _count_from_value(_value_for_split(splits_value, split))
            if count is not None:
                result.setdefault(split, count)
    for split in _SPLITS:
        count = _count_from_value(_value_for_split(metadata, split))
        if count is not None:
            result.setdefault(split, count)
    return result


def extract_split_hashes(metadata: Any) -> dict[str, str]:
    """Extract SHA-256 digests for split files from metadata."""

    result: dict[str, str] = {}

    def add(split: str | None, value: Any) -> None:
        digest = _normalise_hash(value)
        if split and digest:
            result.setdefault(split, digest)

    def scan_mapping(mapping: Mapping[str, Any], *, hash_context: bool = False) -> None:
        for key, value in mapping.items():
            split = _canonical_split(key)
            if split:
                if isinstance(value, Mapping):
                    for hash_key in _HASH_KEYS:
                        if hash_key in value:
                            add(split, value[hash_key])
                elif hash_context or _normalise_hash(value) is not None:
                    # A flat input_hashes.json commonly maps "train.txt"
                    # directly to its digest.  The split-like key plus a
                    # valid 64-hex value is already unambiguous.
                    add(split, value)
            key_text = str(key).lower()
            child_hash_context = hash_context or key_text in {
                "hashes",
                "checksums",
                "sha256",
                "sha_256",
                "file_hashes",
                "split_hashes",
                "digests",
            }
            if isinstance(value, Mapping):
                scan_mapping(value, hash_context=child_hash_context)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Mapping):
                        split_item = _canonical_split(item.get("split") or item.get("path") or item.get("file") or item.get("name"))
                        for hash_key in _HASH_KEYS:
                            if hash_key in item:
                                add(split_item, item[hash_key])
                        scan_mapping(item, hash_context=child_hash_context)

    if isinstance(metadata, Mapping):
        scan_mapping(metadata)
    elif isinstance(metadata, list):
        for item in metadata:
            if isinstance(item, Mapping):
                split = _canonical_split(item.get("split") or item.get("path") or item.get("file") or item.get("name"))
                for hash_key in _HASH_KEYS:
                    if hash_key in item:
                        add(split, item[hash_key])
                scan_mapping(item)
    return result


def _subject_from_text(text: str) -> str | None:
    normalized = text.replace("\\", "/")
    stem = Path(normalized).stem

    # Original IPN video naming: 1CM42_3_R_#132.  Also accept exports which
    # removed the '#': 1CM42_3_R_132.
    match = re.search(r"(?P<subject>[^/]+_\d+)_([RL])_#?\d+(?:_|$)", stem)
    if match:
        return match.group("subject")

    # Processed keypoint/graph paths use a SubjectNN[_Name] directory.
    for part in normalized.split("/"):
        if re.fullmatch(r"Subject[\w-]+", part, flags=re.IGNORECASE):
            return part

    # A few exports retain the subject as a generic directory below the label
    # directory.  Use it only when the path is deep enough to distinguish it
    # from the split and label components.
    parts = [part for part in normalized.split("/") if part]
    for index, part in enumerate(parts[:-1]):
        if part.upper() in LABEL_IDS and index + 1 < len(parts) - 1:
            return parts[index + 1]
    return None


def subject_from_source(source: str) -> str | None:
    """Infer an IPN subject identifier from a split source path."""

    return _subject_from_text(source)


def _metadata_subjects(metadata: Any) -> set[str]:
    subjects: set[str] = set()

    def add_text(value: Any) -> None:
        if not isinstance(value, str):
            return
        if value.strip() == EXPECTED_EXCLUDED_SUBJECT:
            subjects.add(EXPECTED_EXCLUDED_SUBJECT)
        inferred = _subject_from_text(value)
        if inferred:
            subjects.add(inferred)

    for mapping in _iter_mappings(metadata):
        for key, value in mapping.items():
            key_text = str(key).lower()
            if any(token in key_text for token in ("subject", "video", "source", "exclude", "omit")):
                if isinstance(value, list):
                    for item in value:
                        add_text(item)
                else:
                    add_text(value)
    return subjects


def _metadata_exclusions(metadata: Any) -> set[str]:
    result: set[str] = set()
    for mapping in _iter_mappings(metadata):
        for key, value in mapping.items():
            if "exclud" not in str(key).lower() and "omit" not in str(key).lower():
                continue
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, str):
                    inferred = _subject_from_text(item) or item.strip()
                    if inferred:
                        result.add(inferred)
    return result


def _metadata_has_exclusion_field(metadata: Any) -> bool:
    """Whether metadata explicitly records an exclusion/omission list."""

    return any(
        "exclud" in str(key).lower() or "omit" in str(key).lower()
        for mapping in _iter_mappings(metadata)
        for key in mapping
    )


def _resolve_relative(source: str, root: Path) -> Path | None:
    source_path = Path(source)
    try:
        return source_path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        pass

    source_text = source.replace("\\", "/").strip()
    root_text = root.as_posix().rstrip("/")
    if source_text.startswith(root_text + "/"):
        return Path(source_text[len(root_text) + 1 :])

    # Relative paths often retain the keypoint-root directory name.
    root_name = root.name.replace("\\", "/")
    marker = f"/{root_name}/"
    if marker in f"/{source_text}":
        return Path(source_text.split(marker, 1)[1])
    return None


def instance_graph_candidates(source: str, keypoint_root: Path, graph_root: Path) -> list[Path]:
    """Return likely graph paths for a split-file source entry."""

    candidates: list[Path] = []
    relative = _resolve_relative(source, keypoint_root)
    if relative is not None:
        relative = relative.with_suffix(".json")
        candidates.append(graph_root / relative)
        parts = relative.parts
        if parts and parts[0].lower() in _SPLITS:
            candidates.append(graph_root / Path(*parts[1:]))

    source_path = Path(source.replace("\\", "/"))
    if source_path.suffix:
        source_path = source_path.with_suffix(".json")
    candidates.append(graph_root / source_path)
    if source_path.is_absolute():
        candidates.append(source_path)

    # Avoid duplicate checks while preserving deterministic order.
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.as_posix().lower()
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _load_graph(path: Path) -> Any:
    raw = path.read_bytes()
    if not raw.strip():
        raise ValueError("file is empty")
    value = json.loads(raw.decode("utf-8-sig"))
    if value in (None, {}, [], ""):
        raise ValueError("JSON value is empty")
    return value


def _load_instance_graph(path: Path) -> Mapping[str, Any]:
    value = _load_graph(path)
    if not isinstance(value, Mapping) or value.get("graph_type") != "instance_graph":
        raise ValueError("expected graph_type=instance_graph")
    nodes = value.get("nodes")
    edges = value.get("edges")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("instance graph nodes must be a nonempty list")
    if not isinstance(edges, list):
        raise ValueError("instance graph edges must be a list")
    return value


def _canonical_json(value: Any) -> str:
    """Return a deterministic representation for metadata comparisons."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_v2_instance_metadata(
    value: Mapping[str, Any],
    path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Validate the reproducibility metadata emitted by the V2 precompute.

    The source fingerprint is intentionally not part of the uniformity check:
    it is expected to differ for every clip.  The predicate and build config,
    however, must be present and identical across one graph cohort.
    """

    if _coerce_int(value.get("schema_version")) != 2:
        raise ValueError("missing or unsupported schema_version=2")

    predicate_config = value.get("predicate_config")
    if not isinstance(predicate_config, Mapping) or not predicate_config:
        raise ValueError("missing non-empty predicate_config")

    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("missing non-empty provenance")

    provenance_predicate_config = provenance.get("predicate_config")
    if not isinstance(provenance_predicate_config, Mapping) or not provenance_predicate_config:
        raise ValueError("missing non-empty provenance.predicate_config")
    if _canonical_json(predicate_config) != _canonical_json(provenance_predicate_config):
        raise ValueError("predicate_config disagrees with provenance.predicate_config")

    build_config = provenance.get("build_config")
    if not isinstance(build_config, Mapping) or not build_config:
        raise ValueError("missing non-empty provenance.build_config")
    if _coerce_int(build_config.get("schema_version")) != 2:
        raise ValueError("missing or unsupported provenance.build_config.schema_version=2")
    source_fingerprint = provenance.get("source_fingerprint")
    if not isinstance(source_fingerprint, Mapping) or not source_fingerprint:
        raise ValueError("missing non-empty provenance.source_fingerprint")
    build_predicate_config = build_config.get("predicate_config")
    if build_predicate_config is not None and _canonical_json(build_predicate_config) != _canonical_json(predicate_config):
        raise ValueError("build_config.predicate_config disagrees with predicate_config")

    return predicate_config, provenance, build_config


def _graph_records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if "label_id" in value:
            return [value]
        templates = value.get("templates")
        if isinstance(templates, list):
            return [item for item in templates if isinstance(item, Mapping)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def validate_split_inputs(
    split_files: Mapping[str, Path],
    report: ValidationReport,
    *,
    expected_counts: Mapping[str, int] | None,
    metadata: Any = None,
    metadata_path: Path | None = None,
    require_all_labels: bool = True,
) -> dict[str, list[SplitEntry]]:
    entries_by_split: dict[str, list[SplitEntry]] = {}
    all_ids: set[int] = set()
    subjects_by_split: dict[str, set[str]] = {}
    source_seen: dict[str, str] = {}

    metadata_counts = extract_split_counts(metadata) if metadata is not None else {}
    metadata_hashes = extract_split_hashes(metadata) if metadata is not None else {}
    report.stats["metadata_counts"] = metadata_counts
    report.stats["metadata_hashes"] = metadata_hashes

    for split in _SPLITS:
        path = split_files.get(split)
        if path is None:
            report.add_error(f"Missing {split} split path")
            continue
        if not path.is_file():
            report.add_error(f"Missing {split} split file: {path}")
            continue
        try:
            entries = read_split_file(path)
        except (OSError, UnicodeError, ValueError) as exc:
            report.add_error(str(exc))
            continue
        entries_by_split[split] = entries

        expected = metadata_counts.get(split)
        if expected is None and expected_counts is not None:
            expected = expected_counts.get(split)
        if expected is not None and len(entries) != expected:
            report.add_error(f"{split} split count mismatch: expected {expected}, found {len(entries)}")

        digest = metadata_hashes.get(split)
        if digest:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                report.add_error(f"{split} split SHA-256 mismatch: expected {digest}, found {actual}")

        ids = {entry.label_id for entry in entries}
        all_ids.update(ids)
        invalid_ids = sorted(ids - EXPECTED_LABEL_IDS)
        if invalid_ids:
            report.add_error(f"{split} contains label IDs outside 0..12: {invalid_ids}")

        split_subjects: set[str] = set()
        for entry in entries:
            if entry.source in source_seen:
                report.add_error(
                    f"Duplicate source across split files: {entry.source} "
                    f"({source_seen[entry.source]} and {split})"
                )
            source_seen[entry.source] = split
            subject = subject_from_source(entry.source)
            if subject is None:
                report.add_error(f"Cannot infer subject from {split} source: {entry.source}")
            else:
                split_subjects.add(subject)
            # Where the source path carries a known label directory, check the
            # explicit numeric ID against the canonical 13-class mapping.
            parts = entry.source.replace("\\", "/").split("/")
            path_labels = [part for part in parts if part in LABEL_IDS]
            if path_labels and LABEL_IDS[path_labels[-1]] != entry.label_id:
                report.add_error(
                    f"{split} label/path mismatch for {entry.source}: "
                    f"ID {entry.label_id} is not {path_labels[-1]} ({LABEL_IDS[path_labels[-1]]})"
                )
        subjects_by_split[split] = split_subjects

    if require_all_labels and all_ids != EXPECTED_LABEL_IDS:
        report.add_error(
            "Combined split files must contain every label ID 0..12; "
            f"missing {sorted(EXPECTED_LABEL_IDS - all_ids)}"
        )

    for left_index, left in enumerate(_SPLITS):
        for right in _SPLITS[left_index + 1 :]:
            overlap = sorted(subjects_by_split.get(left, set()) & subjects_by_split.get(right, set()))
            if overlap:
                report.add_error(f"Subject overlap between {left} and {right}: {overlap}")
    report.stats["split_counts"] = {split: len(entries_by_split.get(split, [])) for split in _SPLITS}
    report.stats["subjects"] = {split: sorted(subjects_by_split.get(split, set())) for split in _SPLITS}
    report.stats["label_ids"] = sorted(all_ids)
    return entries_by_split


def validate_templates(
    template_dir: Path,
    report: ValidationReport,
    *,
    expected_count: int = DEFAULT_TEMPLATE_COUNT,
    expected_ids: Iterable[int] | None = None,
) -> None:
    if not template_dir.is_dir():
        report.add_error(f"Missing template graph directory: {template_dir}")
        return

    graph_files = sorted(
        path for path in template_dir.glob("*.json") if path.name.lower() not in {"index.json", "manifest.json"}
    )
    records: list[tuple[Path, Mapping[str, Any]]] = []
    for path in graph_files:
        try:
            value = _load_graph(path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            report.add_error(f"Invalid template graph {path}: {exc}")
            continue
        for record in _graph_records(value):
            records.append((path, record))

    index_path = template_dir / "index.json"
    index_value: Any = None
    if index_path.is_file():
        try:
            index_value = _load_graph(index_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            report.add_error(f"Invalid template graph index {index_path}: {exc}")
        if not records:
            records.extend((index_path, record) for record in _graph_records(index_value))

    if not records:
        report.add_error(f"No template graph records found in {template_dir}")
        return
    if len(records) != expected_count:
        report.add_error(f"Template graph count mismatch: expected {expected_count}, found {len(records)}")

    ids: list[int] = []
    names_by_id: dict[int, str] = {}
    ids_by_name: dict[str, set[int]] = {}
    for path, record in records:
        # Entries copied from index.json are summaries and need not repeat the
        # full graph_type field.  Individual graph files must carry it.
        if path.name.lower() != "index.json" and record.get("graph_type") != "template_graph":
            report.add_error(f"Template graph {path} must have graph_type=template_graph")
        label_id = _coerce_int(record.get("label_id"))
        if label_id is None:
            report.add_error(f"Template graph {path} has no integer label_id")
            continue
        ids.append(label_id)
        if label_id not in EXPECTED_LABEL_IDS:
            report.add_error(f"Template graph {path} has label_id outside 0..12: {label_id}")
            continue
        label_name = record.get("label_name")
        if not isinstance(label_name, str) or not label_name.strip():
            report.add_error(
                f"Template graph {path} label_id {label_id} is missing label_name; "
                f"expected canonical name {LABELS[label_id]}"
            )
            continue
        label_name = label_name.strip()
        names_by_id.setdefault(label_id, label_name)
        ids_by_name.setdefault(label_name, set()).add(label_id)
        if label_name != LABELS[label_id]:
            report.add_error(
                f"Template graph {path} has non-canonical label_name {label_name!r} "
                f"for label_id {label_id}; expected {LABELS[label_id]!r}"
            )
    duplicates = sorted({label_id for label_id in ids if ids.count(label_id) > 1})
    if duplicates:
        report.add_error(f"Duplicate template graph label IDs: {duplicates}")
    wanted = set(expected_ids) if expected_ids is not None else EXPECTED_LABEL_IDS
    if set(ids) != wanted:
        missing_ids = sorted(wanted - set(ids))
        extra_ids = sorted(set(ids) - wanted)
        report.add_error(
            f"Template graph IDs mismatch: expected {sorted(wanted)}, found {sorted(set(ids))}; "
            f"missing {missing_ids}, extra {extra_ids}"
        )
    duplicate_names = sorted(name for name, name_ids in ids_by_name.items() if len(name_ids) > 1)
    if duplicate_names:
        report.add_error(f"Duplicate template graph label names: {duplicate_names}")

    # A valid cohort has exactly one canonical ID -> name pair for every
    # expected class.  This catches an index with the right IDs but stale or
    # swapped names, which numeric-ID-only checks cannot detect.
    missing_names = sorted(
        label_id for label_id in wanted if label_id not in names_by_id
    )
    if missing_names:
        report.add_error(
            "Template graph canonical label names missing for IDs: "
            f"{missing_names}"
        )

    if isinstance(index_value, Mapping):
        index_count = _coerce_int(index_value.get("template_count"))
        if index_count is not None and index_count != expected_count:
            report.add_error(f"Template index count mismatch: expected {expected_count}, found {index_count}")
        index_records = _graph_records(index_value)
        if index_records:
            index_ids: list[int] = []
            index_names_by_id: dict[int, str] = {}
            for item in index_records:
                index_id = _coerce_int(item.get("label_id"))
                if index_id is None:
                    report.add_error("Template index entry has no integer label_id")
                    continue
                index_ids.append(index_id)
                if index_id in EXPECTED_LABEL_IDS:
                    index_name = item.get("label_name")
                    if not isinstance(index_name, str) or not index_name.strip():
                        report.add_error(
                            f"Template index entry for label_id {index_id} is missing label_name; "
                            f"expected {LABELS[index_id]}"
                        )
                    else:
                        index_name = index_name.strip()
                        index_names_by_id.setdefault(index_id, index_name)
                        if index_name != LABELS[index_id]:
                            report.add_error(
                                f"Template index entry has non-canonical label_name {index_name!r} "
                                f"for label_id {index_id}; expected {LABELS[index_id]!r}"
                            )
            if len(index_ids) != len(set(index_ids)):
                report.add_error(
                    "Duplicate template index label IDs: "
                    f"{sorted({label_id for label_id in index_ids if index_ids.count(label_id) > 1})}"
                )
            if set(index_ids) != set(ids):
                report.add_error("Template index IDs do not match template graph IDs")
            for label_id, label_name in names_by_id.items():
                if index_names_by_id.get(label_id) != label_name:
                    report.add_error(
                        f"Template index label mapping disagrees for ID {label_id}: "
                        f"expected {label_name!r}, found {index_names_by_id.get(label_id)!r}"
                    )

    report.stats["template_count"] = len(records)
    report.stats["template_ids"] = sorted(ids)
    report.stats["template_label_map"] = {
        str(label_id): names_by_id[label_id]
        for label_id in sorted(names_by_id)
    }


def validate_instance_graphs(
    entries_by_split: Mapping[str, Sequence[SplitEntry]],
    keypoint_root: Path,
    graph_root: Path,
    report: ValidationReport,
    *,
    allow_legacy_graphs: bool = False,
) -> None:
    if not graph_root.is_dir():
        report.add_error(f"Missing instance graph directory: {graph_root}")
        return

    expected_total = sum(len(entries) for entries in entries_by_split.values())
    found_total = 0
    checked_paths: set[Path] = set()
    missing: list[str] = []
    invalid: list[str] = []
    predicate_configs: dict[str, str] = {}
    build_configs: dict[str, str] = {}
    provenance_configs: dict[str, str] = {}

    def inspect(path: Path) -> None:
        try:
            value = _load_instance_graph(path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            invalid.append(f"{path}: {exc}")
            return
        if allow_legacy_graphs:
            return
        try:
            predicate_config, provenance, build_config = _validate_v2_instance_metadata(value, path)
        except ValueError as exc:
            invalid.append(f"{path}: {exc}")
            return
        predicate_signature = _canonical_json(predicate_config)
        build_signature = _canonical_json(build_config)
        # source_fingerprint is per-clip by design; all other provenance
        # fields describe the graph-building recipe and must be uniform.
        provenance_core = {
            key: item
            for key, item in provenance.items()
            if key != "source_fingerprint"
        }
        provenance_signature = _canonical_json(provenance_core)
        predicate_configs.setdefault(predicate_signature, str(path))
        build_configs.setdefault(build_signature, str(path))
        provenance_configs.setdefault(provenance_signature, str(path))

    for split in _SPLITS:
        for entry in entries_by_split.get(split, ()):
            candidates = instance_graph_candidates(entry.source, keypoint_root, graph_root)
            graph_path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if graph_path is None:
                missing.append(f"{split}: {entry.source}")
                continue
            found_total += 1
            resolved_path = graph_path.resolve()
            checked_paths.add(resolved_path)
            inspect(graph_path)

    # Validate every JSON file, including stale/unreferenced files.  A corrupt
    # extra graph should not be silently shipped alongside a complete split.
    for path in sorted(graph_root.rglob("*.json")):
        if path.resolve() in checked_paths:
            continue
        inspect(path)

    if not allow_legacy_graphs:
        if len(predicate_configs) > 1:
            report.add_error(
                "Mixed predicate_config across instance graphs: "
                f"{sorted(predicate_configs.values())}"
            )
        if len(build_configs) > 1:
            report.add_error(
                "Mixed provenance.build_config across instance graphs: "
                f"{sorted(build_configs.values())}"
            )
        if len(provenance_configs) > 1:
            report.add_error(
                "Mixed provenance across instance graphs: "
                f"{sorted(provenance_configs.values())}"
            )

    if missing:
        report.add_error(f"Instance graph coverage is not 100%: missing {len(missing)} of {expected_total}: {missing[:5]}")
    if invalid:
        report.add_error(f"Invalid/nonempty instance graph JSON ({len(invalid)}): {invalid[:5]}")
    report.stats["instance_graph_expected"] = expected_total
    report.stats["instance_graph_found"] = found_total
    report.stats["instance_graph_coverage"] = (found_total / expected_total) if expected_total else 0.0
    report.stats["instance_graph_allow_legacy"] = allow_legacy_graphs
    report.stats["instance_graph_predicate_config_count"] = len(predicate_configs)
    report.stats["instance_graph_build_config_count"] = len(build_configs)
    if len(predicate_configs) == 1:
        predicate_signature = next(iter(predicate_configs))
        report.stats["instance_graph_predicate_config"] = json.loads(predicate_signature)
        report.stats["instance_graph_predicate_config_sha256"] = _json_sha256(
            report.stats["instance_graph_predicate_config"]
        )
    if len(build_configs) == 1:
        build_signature = next(iter(build_configs))
        report.stats["instance_graph_build_config"] = json.loads(build_signature)
        report.stats["instance_graph_build_config_sha256"] = _json_sha256(
            report.stats["instance_graph_build_config"]
        )


def _default_split_files(data_root: Path) -> dict[str, Path]:
    return {split: data_root / f"{split}.txt" for split in _SPLITS}


def validate_symbolic_inputs(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    *,
    split_files: Mapping[str, str | Path] | None = None,
    keypoint_root: str | Path | None = None,
    template_dir: str | Path | None = None,
    instance_graph_root: str | Path | None = None,
    metadata_path: str | Path | None = None,
    expected_split_counts: Mapping[str, int] | None = DEFAULT_SPLIT_COUNTS,
    expected_template_count: int = DEFAULT_TEMPLATE_COUNT,
    require_all_labels: bool = True,
    require_hashes: bool = False,
    expected_exclusion: str = EXPECTED_EXCLUDED_SUBJECT,
    allow_legacy_graphs: bool = False,
) -> ValidationReport:
    """Validate IPN split, template graph, and instance graph inputs.

    Metadata-provided counts and hashes override the corresponding defaults.
    Set ``expected_split_counts`` to ``None`` when validating a deliberately
    small fixture whose metadata does not carry counts.  The returned report is
    always populated; callers can inspect ``report.errors`` or ``report.ok``.
    """

    data_root = Path(data_root)
    split_paths = _default_split_files(data_root)
    if split_files:
        split_paths.update({split: Path(path) for split, path in split_files.items()})
    keypoint_root = Path(keypoint_root) if keypoint_root is not None else data_root / "keypoints"
    template_dir = Path(template_dir) if template_dir is not None else data_root / "template_graphs"
    instance_graph_root = Path(instance_graph_root) if instance_graph_root is not None else data_root / "instance_graphs"
    metadata = None
    resolved_metadata = discover_metadata(data_root, Path(metadata_path) if metadata_path is not None else None)
    report = ValidationReport()

    if resolved_metadata is not None:
        if not resolved_metadata.is_file():
            report.add_error(f"Missing metadata file: {resolved_metadata}")
        else:
            try:
                metadata = _read_metadata(resolved_metadata)
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                report.add_error(f"Cannot parse metadata {resolved_metadata}: {exc}")
    elif require_hashes:
        report.add_error("Hash metadata is required but no metadata file was found")
    else:
        report.add_warning("No metadata file found; split hashes and metadata counts were not checked")

    entries_by_split = validate_split_inputs(
        split_paths,
        report,
        expected_counts=expected_split_counts,
        metadata=metadata,
        metadata_path=resolved_metadata,
        require_all_labels=require_all_labels,
    )

    if metadata is not None:
        metadata_hashes = extract_split_hashes(metadata)
        if require_hashes and set(metadata_hashes) != set(_SPLITS):
            report.add_error(
                "Hash metadata must contain SHA-256 for train, val, and test; "
                f"found {sorted(metadata_hashes)}"
            )

        actual_subjects = {
            subject
            for split_entries in entries_by_split.values()
            for entry in split_entries
            if (subject := subject_from_source(entry.source)) is not None
        }
        metadata_subjects = _metadata_subjects(metadata)
        metadata_exclusions = _metadata_exclusions(metadata)
        if expected_exclusion and expected_exclusion in actual_subjects:
            report.add_error(f"Expected excluded subject is present in split inputs: {expected_exclusion}")
        # If metadata explicitly carries an exclusion list, ensure the known
        # exclusion is recorded there too; otherwise absence from the splits is
        # the meaningful invariant.
        if _metadata_has_exclusion_field(metadata) and expected_exclusion not in metadata_exclusions:
            report.add_error(
                f"Metadata exclusion list does not contain expected subject {expected_exclusion}: "
                f"{sorted(metadata_exclusions)}"
            )
        report.stats["metadata_subject_count"] = len(metadata_subjects)
        report.stats["metadata_exclusions"] = sorted(metadata_exclusions)
    elif expected_exclusion:
        report.stats["metadata_exclusion_check"] = "skipped (metadata unavailable)"

    validate_templates(template_dir, report, expected_count=expected_template_count)
    validate_instance_graphs(
        entries_by_split,
        keypoint_root,
        instance_graph_root,
        report,
        allow_legacy_graphs=allow_legacy_graphs,
    )
    report.stats["data_root"] = str(data_root)
    report.stats["metadata_path"] = str(resolved_metadata) if resolved_metadata else None
    return report


# Short aliases are useful for callers which prefer the command's verb.
validate_inputs = validate_symbolic_inputs
validate = validate_symbolic_inputs


def _parse_expected_counts(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected count must use SPLIT=COUNT, got {value!r}")
        split_text, count_text = value.split("=", 1)
        split = _canonical_split(split_text)
        count = _coerce_int(count_text)
        if split is None or count is None or count < 0:
            raise ValueError(f"Expected count must use SPLIT=nonnegative integer, got {value!r}")
        result[split] = count
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--keypoint-root", type=Path, help="Root used to relativize split source paths")
    parser.add_argument("--template-dir", type=Path)
    parser.add_argument("--instance-graph-root", type=Path)
    parser.add_argument("--metadata", "--metadata-path", dest="metadata_path", type=Path)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--val-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument(
        "--expected-count",
        action="append",
        default=[],
        metavar="SPLIT=COUNT",
        help="Override a split count (repeat for train/val/test); metadata counts take priority.",
    )
    parser.add_argument("--no-default-counts", action="store_true", help="Only enforce counts recorded in metadata")
    parser.add_argument("--expected-template-count", type=int, default=DEFAULT_TEMPLATE_COUNT)
    parser.add_argument("--allow-partial-labels", action="store_true", help="Do not require all IDs 0..12 to occur")
    parser.add_argument(
        "--allow-legacy-graphs",
        action="store_true",
        help="Allow pre-V2 instance graphs without predicate/provenance/build metadata",
    )
    parser.add_argument("--require-hashes", action="store_true", help="Require SHA-256 metadata for all three split files")
    parser.add_argument("--expected-exclusion", default=EXPECTED_EXCLUDED_SUBJECT)
    parser.add_argument("--json-report", type=Path, help="Write the structured report to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        overrides = _parse_expected_counts(args.expected_count)
    except ValueError as exc:
        parser.error(str(exc))
    expected_counts: Mapping[str, int] | None
    if args.no_default_counts:
        expected_counts = overrides or None
    else:
        expected_counts = dict(DEFAULT_SPLIT_COUNTS)
        expected_counts.update(overrides)

    split_files = {
        split: path
        for split, path in {
            "train": args.train_file,
            "val": args.val_file,
            "test": args.test_file,
        }.items()
        if path is not None
    }
    report = validate_symbolic_inputs(
        args.data_root,
        split_files=split_files or None,
        keypoint_root=args.keypoint_root,
        template_dir=args.template_dir,
        instance_graph_root=args.instance_graph_root,
        metadata_path=args.metadata_path,
        expected_split_counts=expected_counts,
        expected_template_count=args.expected_template_count,
        require_all_labels=not args.allow_partial_labels,
        require_hashes=args.require_hashes,
        expected_exclusion=args.expected_exclusion,
        allow_legacy_graphs=args.allow_legacy_graphs,
    )

    for warning in report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if report.ok:
        print("Symbolic input validation: PASS")
    else:
        print("Symbolic input validation: FAIL", file=sys.stderr)
        for error in report.errors:
            print(f"ERROR: {error}", file=sys.stderr)
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
