#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSVS = [
    REPO_ROOT / "chi_2026_schedule_export.csv",
    REPO_ROOT / "chi_2025_schedule_export.csv",
    REPO_ROOT / "chi_2024_schedule_export.csv",
    REPO_ROOT / "acl_2025_schedule_export.csv",
    REPO_ROOT / "acl_2024_schedule_export.csv",
    REPO_ROOT / "facct_2025_schedule_export.csv",
    REPO_ROOT / "facct_2024_schedule_export.csv",
]
OUT_PUBLIC_DIR = REPO_ROOT / "chi_relevance_client" / "public"
OUT_DATA_DIR = OUT_PUBLIC_DIR / "data"
OUT_MANIFEST_JSON = OUT_DATA_DIR / "schedule_manifest.json"
OUT_LEGACY_JSON = OUT_DATA_DIR / "schedule_index.json"
CONFERENCE_DATA_DIR = OUT_DATA_DIR / "conferences"
SEMANTIC_EMBEDDINGS_BASENAME = "schedule_semantic_embeddings_q4"

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "llm", "llms", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves",
    "acm", "accepted", "affiliation", "al", "arxiv", "author", "authors", "conference", "copyright", "date", "department", "doi",
    "edition", "et", "figure", "figures", "http", "https", "institute", "isbn", "issn", "journal", "org", "page", "pages",
    "preprint", "proceedings", "section", "table", "tables", "university", "volume", "vol", "workshop", "www",
    "staufer", "morehouse", "hartmann", "berendt",
}
TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall((text or "").lower())
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit()]


def combine_schedule_text(row: dict[str, str]) -> str:
    parts = [
        row.get("title", ""),
        row.get("abstract", ""),
        row.get("session_titles", ""),
        row.get("session_type", ""),
        row.get("content_type", ""),
    ]
    return " ".join(p for p in parts if p)


def parse_optional_int(value: str | None) -> int | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def compact(value: str | None) -> str:
    return (value or "").strip()


def conference_key(short_name: str, year: str) -> str:
    normalized_short_name = re.sub(r"[^a-z0-9]+", "-", short_name.lower()).strip("-")
    normalized_year = re.sub(r"[^0-9]+", "", year)
    if normalized_short_name and normalized_year:
        return f"{normalized_short_name}-{normalized_year}"
    return normalized_short_name or normalized_year or "unknown-conference"


def conference_label(short_name: str, year: str) -> str:
    if short_name and year:
        return f"{short_name} {year}"
    return short_name or year or "Unknown Conference"


def conference_data_url(conf_key: str, filename: str) -> str:
    return f"data/conferences/{conf_key}/{filename}"


def normalize_field(value: str | None) -> str:
    return compact(value or "CS").upper() or "CS"


def read_schedule_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    missing_required: list[Path] = []

    for input_csv in INPUT_CSVS:
        if not input_csv.exists():
            if input_csv.name == "chi_2026_schedule_export.csv":
                missing_required.append(input_csv)
            continue
        with input_csv.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))

    if missing_required:
        missing = ", ".join(str(path) for path in missing_required)
        raise FileNotFoundError(f"Required schedule CSV not found: {missing}")
    if not rows:
        raise ValueError("No schedule rows found in configured CSV inputs.")
    return rows


def main() -> int:
    rows = read_schedule_rows()

    packed_rows_by_key: dict[str, list[dict[str, object]]] = {}
    conferences_by_key: dict[str, dict[str, object]] = {}

    for row in rows:
        counter = Counter(tokenize(combine_schedule_text(row)))
        short_name = compact(row.get("conference_short_name"))
        year = compact(row.get("conference_year"))
        time_zone = compact(row.get("conference_timezone"))
        field = normalize_field(row.get("conference_field"))
        conf_key = conference_key(short_name, year)
        if conf_key not in conferences_by_key:
            conferences_by_key[conf_key] = {
                "key": conf_key,
                "id": compact(row.get("conference_id")),
                "short_name": short_name,
                "year": year,
                "field": field,
                "timezone": time_zone,
                "label": conference_label(short_name, year),
                "row_count": 0,
                "first_start_unix_ms": None,
                "last_end_unix_ms": None,
                "index_url": conference_data_url(conf_key, "schedule_index.json"),
                "semantic_embeddings_meta_url": conference_data_url(conf_key, f"{SEMANTIC_EMBEDDINGS_BASENAME}.json"),
                "semantic_embeddings_bin_url": conference_data_url(conf_key, f"{SEMANTIC_EMBEDDINGS_BASENAME}.bin"),
            }
        elif time_zone and not conferences_by_key[conf_key].get("timezone"):
            conferences_by_key[conf_key]["timezone"] = time_zone

        packed_row = {
            "conference_key": conf_key,
            "conference_id": compact(row.get("conference_id")),
            "conference_short_name": short_name,
            "conference_year": year,
            "conference_label": conference_label(short_name, year),
            "conference_timezone": time_zone,
            "conference_field": field,
            "item_kind": compact(row.get("item_kind")),
            "content_type": compact(row.get("content_type")),
            "title": row.get("title", ""),
            "authors": row.get("authors", ""),
            "abstract": row.get("abstract", ""),
            "room": row.get("room", ""),
            "building": row.get("building", ""),
            "day": row.get("day", ""),
            "session_type": row.get("session_type", ""),
            "start_date_unix_ms": parse_optional_int(row.get("start_date_unix_ms")),
            "end_date_unix_ms": parse_optional_int(row.get("end_date_unix_ms")),
            "tokens": [[term, count] for term, count in counter.items()],
        }

        conference = conferences_by_key[conf_key]
        conference["row_count"] = int(conference.get("row_count") or 0) + 1
        start_ms = packed_row["start_date_unix_ms"]
        end_ms = packed_row["end_date_unix_ms"] or start_ms
        if start_ms is not None and (
            conference.get("first_start_unix_ms") is None or start_ms < int(conference["first_start_unix_ms"])
        ):
            conference["first_start_unix_ms"] = start_ms
        if end_ms is not None and (
            conference.get("last_end_unix_ms") is None or end_ms > int(conference["last_end_unix_ms"])
        ):
            conference["last_end_unix_ms"] = end_ms

        packed_rows_by_key.setdefault(conf_key, []).append(packed_row)

    conferences = sorted(
        conferences_by_key.values(),
        key=lambda item: (str(item.get("year", "")), str(item.get("short_name", ""))),
        reverse=True,
    )
    total_rows = sum(len(rows_for_conference) for rows_for_conference in packed_rows_by_key.values())

    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for conference in conferences:
        conf_key = str(conference["key"])
        rows_for_conference = [
            {**row, "row_index": idx}
            for idx, row in enumerate(packed_rows_by_key.get(conf_key, []))
        ]
        schedule_doc_freq: Counter[str] = Counter()
        for row in rows_for_conference:
            schedule_doc_freq.update(term for term, _count in row.get("tokens", []))

        payload = {
            "version": 4,
            "shard_key": conf_key,
            "row_count": len(rows_for_conference),
            "conferences": [conference],
            "rows": rows_for_conference,
            "schedule_doc_freq": dict(schedule_doc_freq),
            "semantic_embeddings_meta_url": conference["semantic_embeddings_meta_url"],
            "semantic_embeddings_bin_url": conference["semantic_embeddings_bin_url"],
        }

        out_json = OUT_PUBLIC_DIR / str(conference["index_url"])
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        size_mb = out_json.stat().st_size / (1024 * 1024)
        print(f"Wrote {out_json} ({size_mb:.2f} MB, {len(rows_for_conference)} rows)")

    manifest = {
        "version": 4,
        "sharded": True,
        "row_count": total_rows,
        "conference_count": len(conferences),
        "conferences": conferences,
    }

    with OUT_MANIFEST_JSON.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, separators=(",", ":"))

    # Keep the historical URL lightweight for older tooling and quick inspection.
    with OUT_LEGACY_JSON.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, separators=(",", ":"))

    manifest_kb = OUT_MANIFEST_JSON.stat().st_size / 1024
    print(f"Wrote {OUT_MANIFEST_JSON} ({manifest_kb:.1f} KB, {total_rows} rows across {len(conferences)} conferences)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
