#!/usr/bin/env python3
"""Filter NDC extraction JSONs and export the selection to Excel.

Reads every per-document JSON in an output folder (default: all_docs_out/),
flattens document + passage fields into rows, and applies filters on the
fields the team cares about:

    --country        country name or ISO-3 code (repeatable)
    --commitment     commitment type: Restoration/Conservation/Agriculture/Both
    --relevance      relevance confidence: High/Medium/Low
    --has-target     only passages with a real target value (not NA)
    --target-contains  substring inside target_value (e.g. "2030")
    --search         substring across original text, translation and notes

Example:
    python3 export_extractions.py --country COL --country Brazil \
        --commitment Restoration --relevance High --has-target \
        --excel out/brazil_colombia_restoration.xlsx

Country names in the LLM output are free text ("Republic of Türkiye",
"European Union and its Member States", "République Centrafricaine", ...),
so they are normalised to ISO-3166-1 alpha-3 codes (iban.com list; the EU is
kept as the special pseudo-code "EU"). Filters accept either names or codes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from ndc_processor import schema

APP_DIR = Path(__file__).resolve().parent

# --- country name -> ISO-3 -------------------------------------------------
# Covers every distinct value found in the LLM output (free-text country names,
# official long forms, accented/French spellings) plus the plain iban.com list.
EU_ALIASES = {
    "European Union",
    "European Union and its Member States",
    "European Union (joint NDC of the EU and its Member States)",
    "European Union (on behalf of the European Union and its Member States)",
}

COUNTRY_ISO3 = {
    "afghanistan": "AFG", "albania": "ALB", "algeria": "DZA", "andorra": "AND",
    "angola": "AGO", "antigua and barbuda": "ATG", "argentina": "ARG",
    "armenia": "ARM", "australia": "AUS", "austria": "AUT", "azerbaijan": "AZE",
    "bahamas": "BHS", "bahrain": "BHR", "bangladesh": "BGD", "barbados": "BRB",
    "belarus": "BLR", "belgium": "BEL", "belize": "BLZ", "benin": "BEN",
    "bhutan": "BTN", "bolivia": "BOL", "bosnia and herzegovina": "BIH",
    "botswana": "BWA", "brazil": "BRA", "brunei": "BRN",
    "brunei darussalam": "BRN", "bulgaria": "BGR", "burkina faso": "BFA",
    "burundi": "BDI", "cabo verde": "CPV", "cambodia": "KHM",
    "cameroon": "CMR", "canada": "CAN", "chad": "TCD", "chile": "CHL",
    "china": "CHN", "colombia": "COL", "comoros": "COM", "congo": "COG",
    "cook islands": "COK", "costa rica": "CRI", "cuba": "CUB",
    "ivory coast": "CIV", "cote d'ivoire": "CIV", "croatia": "HRV",
    "cyprus": "CYP", "czechia": "CZE", "czech republic": "CZE",
    "denmark": "DNK", "djibouti": "DJI", "dominica": "DMA",
    "dominican republic": "DOM", "ecuador": "ECU", "egypt": "EGY",
    "el salvador": "SLV", "equatorial guinea": "GNQ", "eritrea": "ERI",
    "eswatini": "SWZ", "ethiopia": "ETH", "fiji": "FJI", "finland": "FIN",
    "france": "FRA", "gabon": "GAB", "georgia": "GEO", "germany": "DEU",
    "ghana": "GHA", "greece": "GRC", "grenada": "GRD", "guatemala": "GTM",
    "guinea": "GIN", "guinea-bissau": "GNB", "guyana": "GUY", "haiti": "HTI",
    "honduras": "HND", "hungary": "HUN", "iceland": "ISL", "india": "IND",
    "indonesia": "IDN", "iran": "IRN", "iraq": "IRQ", "ireland": "IRL",
    "israel": "ISR", "italy": "ITA", "jamaica": "JAM", "japan": "JPN",
    "jordan": "JOR", "kazakhstan": "KAZ", "kenya": "KEN", "kiribati": "KIR",
    "korea": "KOR", "kosovo": "XKX", "kuwait": "KWT", "kyrgyzstan": "KGZ",
    "laos": "LAO", "latvia": "LVA", "lebanon": "LBN", "lesotho": "LSO",
    "liberia": "LBR", "libya": "LBY", "liechtenstein": "LIE", "lithuania": "LTU",
    "luxembourg": "LUX", "madagascar": "MDG", "malawi": "MWI",
    "malaysia": "MYS", "maldives": "MDV", "mali": "MLI", "malta": "MLT",
    "marshall islands": "MHL", "mauritania": "MRT", "mauritius": "MUS",
    "mexico": "MEX", "micronesia": "FSM", "moldova": "MDA", "monaco": "MCO",
    "mongolia": "MNG", "montenegro": "MNE", "morocco": "MAR", "mozambique": "MOZ",
    "myanmar": "MMR", "namibia": "NAM", "nauru": "NRU", "nepal": "NPL",
    "netherlands": "NLD", "new zealand": "NZL", "nicaragua": "NIC",
    "niger": "NER", "nigeria": "NGA", "niue": "NIU", "north macedonia": "MKD",
    "norway": "NOR", "oman": "OMN", "pakistan": "PAK", "palau": "PLW",
    "panama": "PAN", "papua new guinea": "PNG", "paraguay": "PRY", "peru": "PER",
    "philippines": "PHL", "poland": "POL", "portugal": "PRT",
    "qatar": "QAT", "romania": "ROU", "russia": "RUS", "rwanda": "RWA",
    "saint kitts and nevis": "KNA", "saint lucia": "LCA",
    "saint vincent and the grenadines": "VCT", "samoa": "WSM",
    "san marino": "SMR", "sao tome and principe": "STP",
    "saudi arabia": "SAU", "senegal": "SEN", "serbia": "SRB",
    "seychelles": "SYC", "sierra leone": "SLE", "singapore": "SGP",
    "slovakia": "SVK", "slovenia": "SVN", "solomon islands": "SLB",
    "somalia": "SOM", "south africa": "ZAF", "south sudan": "SSD",
    "spain": "ESP", "sri lanka": "LKA", "sudan": "SDN",
    "suriname": "SUR", "sweden": "SWE", "switzerland": "CHE",
    "syria": "SYR", "tajikistan": "TJK", "tanzania": "TZA",
    "thailand": "THA", "timor-leste": "TLS", "togo": "TGO", "tonga": "TON",
    "trinidad and tobago": "TTO", "tunisia": "TUN", "turkmenistan": "TKM",
    "turkey": "TUR", "turkiye": "TUR", "tuvalu": "TUV", "uganda": "UGA",
    "ukraine": "UKR", "united arab emirates": "ARE",
    "united kingdom": "GBR", "united states": "USA", "uruguay": "URY",
    "uzbekistan": "UZB", "vanuatu": "VUT", "vatican city": "VAT",
    "venezuela": "VEN", "vietnam": "VNM", "yemen": "YEM", "zambia": "ZMB",
    "zimbabwe": "ZWE", "palestine": "PSE",
}

# Official long forms and accented / non-English spellings as they appear in
# the LLM output -> plain canonical name (lowercase).
LONG_FORM_ALIASES = {
    "commonwealth of dominica": "dominica",
    "democratic people's republic of korea": "korea",
    "democratic republic of the congo": "congo",  # DRC -> kept as "congo" below
    "islamic republic of afghanistan": "afghanistan",
    "kingdom of bahrain": "bahrain",
    "kingdom of bhutan": "bhutan",
    "kingdom of eswatini": "eswatini",
    "kingdom of saudi arabia": "saudi arabia",
    "lao people's democratic republic": "laos",
    "republic of azerbaijan": "azerbaijan",
    "republic of kazakhstan": "kazakhstan",
    "republic of korea": "korea",
    "republic of mauritius": "mauritius",
    "republic of moldova": "moldova",
    "republic of nauru": "nauru",
    "republic of north macedonia": "north macedonia",
    "republic of palau": "palau",
    "republic of rwanda": "rwanda",
    "republic of serbia": "serbia",
    "republic of suriname": "suriname",
    "republic of turkiye": "turkiye",
    "republic of the congo": "congo",
    "republic of the marshall islands": "marshall islands",
    "russian federation": "russia",
    "saint kitts and nevis": "saint kitts and nevis",
    "socialist republic of viet nam": "vietnam",
    "state of palestine": "palestine",
    "sultanate of oman": "oman",
    "syrian arab republic": "syria",
    "the bahamas": "bahamas",
    "the gambia": "gambia",
    "the republic of suriname": "suriname",
    "the republic of the union of myanmar": "myanmar",
    "united kingdom of great britain and northern ireland": "united kingdom",
    "united republic of tanzania": "tanzania",
    "vatican city state": "vatican city",
    # accented / French spellings
    "république centrafricaine": "central african republic",
    "république de cote d'ivoire": "cote d'ivoire",
    "république du congo": "congo",
    "république démocratique du congo": "congo",  # DRC
    "república dominicana": "dominican republic",
    "panamá": "panama",
    "são tomé e príncipe": "sao tome and principe",
    "sao tomé and príncipe": "sao tome and principe",
    "bénin": "benin",
    "gambia": "gambia",
    "central african republic": "CAF",
    # fix entries above that collide with plain ISO3 lookups
}
# entries whose plain name is itself an ISO3 code map directly
LONG_FORM_ALIASES["central african republic"] = "CAF"

# DRC vs Republic of the Congo: both map to "congo" (COG) in the table above.
# DRC is COD. Resolve the ambiguous long forms explicitly.
DRC_ALIASES = {
    "democratic republic of the congo",
    "république démocratique du congo",
}

_ISO3_RE = re.compile(r"^[A-Z]{3}$")


def country_to_iso3(name: str) -> str:
    """Map a free-text country name to an ISO-3 code ('' when unmapped)."""
    if not name or name.strip().upper() == "NA":
        return ""
    s = " ".join(name.split()).lower()
    if s in EU_ALIASES or s in {a.lower() for a in EU_ALIASES}:
        return "EU"
    if s in {a.lower() for a in DRC_ALIASES}:
        return "COD"
    s = LONG_FORM_ALIASES.get(s, s)
    # direct ISO-3 value (alias maps that already store a code, e.g. CAF)
    if _ISO3_RE.match(s.upper()):
        return s.upper()
    return COUNTRY_ISO3.get(s, "")


def iso3_to_names(code: str) -> set[str]:
    """Inverse lookup for filtering: code -> set of canonical lowercase names."""
    code = code.upper()
    if code == "EU":
        return EU_ALIASES
    inverse: dict[str, set[str]] = {}
    for name in COUNTRY_ISO3:
        if COUNTRY_ISO3[name] == code:
            inverse.setdefault(code, set()).add(name)
    for long_name, target in LONG_FORM_ALIASES.items():
        iso = target.upper() if _ISO3_RE.match(str(target).upper()) else COUNTRY_ISO3.get(str(target).lower(), "")
        if iso == code:
            inverse.setdefault(code, set()).add(long_name)
    return set(inverse.get(code, {code.lower()}))


# Stray values a model occasionally emits instead of a single enum entry.
_ENUM_ECHO = {
    "restoration | conservation | agriculture | both",
    "high | medium | low",
}


def _disaggregate_eu(source_file: str, fallback: str) -> str:
    """EU-series documents are labelled 'European Union (and its Member States)'
    by the LLM even when the file is a member-state submission. The member is
    encoded in the file name (e.g. ``..._191023_France.md``), so recover it.
    Non-EU documents and EU-level files (no member suffix) are returned as-is.
    """
    if fallback not in EU_ALIASES:
        return fallback
    stem = Path(source_file).stem
    for tok in reversed(stem.split("_")):
        tok = tok.strip()
        if not tok or tok.upper() == "EU":
            continue
        name = LONG_FORM_ALIASES.get(tok.lower(), tok.lower())
        iso = COUNTRY_ISO3.get(name)
        if iso and iso != "EU":
            return " ".join(w.capitalize() for w in name.split())
    return fallback


def load_rows(folder: Path) -> pd.DataFrame:
    rows = []
    n_files = 0
    for f in sorted(folder.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"!! skipping {f.name}: {e}", file=sys.stderr)
            continue
        n_files += 1
        doc = d.get("document") or {}
        source_file = d.get("source_file", f.stem)
        country = _disaggregate_eu(source_file, str(doc.get("country", "NA")))
        base = {
            "source_file": source_file,
            "provider": d.get("provider", "NA"),
            "model": d.get("model", "NA"),
            "iso3": country_to_iso3(country),
            "doc_country": country,
        }
        base.update({f"doc_{k}": doc.get(k, "NA") for k in schema.DOCUMENT_FIELDS})
        base.pop("doc_country", None)
        base["doc_country"] = country
        for p in d.get("passages") or []:
            row = dict(base)
            row.update({f"passage_{k}": p.get(k, "NA") for k in schema.PASSAGE_FIELDS})
            for k in ("commitment_type", "relevance_confidence"):
                if row[f"passage_{k}"].lower() in _ENUM_ECHO:
                    row[f"passage_{k}"] = "NA"
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # canonical column order: iso3 first, then schema.EXCEL_COLUMNS
    cols = ["iso3"] + [c for c in schema.EXCEL_COLUMNS]
    return df[[c for c in cols if c in df.columns]]


def apply_filters(
    df: pd.DataFrame,
    countries: list[str],
    commitments: list[str],
    relevances: list[str],
    has_target: bool,
    target_contains: str,
    search: str,
) -> pd.DataFrame:
    out = df
    if countries:
        keep = pd.Series(False, index=df.index)
        for c in countries:
            c = c.strip()
            if not c:
                continue
            if _ISO3_RE.match(c.upper()):
                keep |= df["iso3"].str.upper().eq(c.upper())
            else:
                keep |= df["doc_country"].str.lower().eq(c.lower())
        out = out[keep]
    if commitments:
        cl = {c.lower() for c in commitments}
        out = out[out["passage_commitment_type"].str.lower().isin(cl)]
    if relevances:
        rl = {r.lower() for r in relevances}
        out = out[out["passage_relevance_confidence"].str.lower().isin(rl)]
    if has_target:
        out = out[~out["passage_target_value"].str.upper().isin(("NA", ""))]
    if target_contains:
        out = out[out["passage_target_value"].str.contains(target_contains, case=False, na=False)]
    if search:
        hay = (
            out["passage_original_text"] + " " + out["passage_english_translation"] + " " + out["passage_notes"]
        )
        out = out[hay.str.contains(re.escape(search), case=False, na=False)]
    return out


_ILLEGAL_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Strip characters that are illegal in xlsx cells (openpyxl rejects them).

    Occurs when an LLM emits control chars inside a passage (e.g. a garbled
    non-English sentence in a note).
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(
                lambda v: _ILLEGAL_XML.sub("", v) if isinstance(v, str) else v
            )
    return out


def to_excel(df: pd.DataFrame, path, criteria: str) -> None:
    headers = dict(schema.EXCEL_HEADERS)
    headers["iso3"] = "ISO3"

    rows = [
        ("rows exported", len(df)),
        ("documents", df["source_file"].nunique() if len(df) else 0),
        ("countries", df["iso3"].nunique() if len(df) else 0),
        ("filter criteria", criteria or "(none — full export)"),
        ("", ""),
        ("commitment type", ""),
    ]
    if len(df):
        rows += [(k, int(v)) for k, v in df["passage_commitment_type"].value_counts().items()]
        rows.append(("", ""))
        rows.append(("relevance confidence", ""))
        rows += [(k, int(v)) for k, v in df["passage_relevance_confidence"].value_counts().items()]
    summary = pd.DataFrame(rows, columns=["item", "value"])

    rename = {c: headers.get(c, c) for c in df.columns} if len(df) else {}
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        if len(df):
            df.rename(columns=rename).pipe(_sanitize_for_excel).to_excel(
                writer, sheet_name="extractions", index=False
            )
        ws = writer.book["summary"]
        ws.column_dimensions["A"].width = 30
        ws.column_dimensions["B"].width = 70
        if len(df):
            ws2 = writer.book["extractions"]
            ws2.freeze_panes = "A2"
            ws2.auto_filter.ref = ws2.dimensions
    if not str(path).startswith(("<", "b")):
        print(f"Exported {len(df)} rows -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(APP_DIR / "all_docs_out"), help="folder with per-document JSONs")
    ap.add_argument("--country", action="append", default=[], help="country name or ISO-3 code (repeatable)")
    ap.add_argument("--commitment", action="append", default=[], help="commitment type (repeatable)")
    ap.add_argument("--relevance", action="append", default=[], help="High/Medium/Low (repeatable)")
    ap.add_argument("--has-target", action="store_true", help="only passages with a real target value")
    ap.add_argument("--target-contains", help="substring inside target value")
    ap.add_argument("--search", help="free-text search across passage text/translation/notes")
    ap.add_argument("--excel", default=None, help="output .xlsx path (default: extractions_<stamp>.xlsx next to --out-dir)")
    args = ap.parse_args(argv)

    folder = Path(args.out_dir)
    df = load_rows(folder)
    if df.empty:
        print("No JSON files found.", file=sys.stderr)
        return 2

    before = len(df)
    out = apply_filters(
        df, args.country, args.commitment, args.relevance,
        args.has_target, args.target_contains, args.search,
    )
    criteria = ", ".join(
        [f"country={args.country}"] if args.country else []
        + [f"commitment={args.commitment}"] if args.commitment else []
        + [f"relevance={args.relevance}"] if args.relevance else []
        + (["has_target"] if args.has_target else [])
        + (f"target~'{args.target_contains}'" if args.target_contains else "")
        + (f"search~'{args.search}'" if args.search else "")
    )
    print(f"loaded {before} rows from {folder}  ->  {len(out)} after filters")
    if len(out):
        cols = ["iso3", "doc_country", "source_file", "passage_commitment_type",
                "passage_relevance_confidence", "passage_target_value", "passage_target_unit"]
        print(out[cols].head(10).to_string(index=False))

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    xlsx = Path(args.excel) if args.excel else folder.parent / f"extractions_{stamp}.xlsx"
    to_excel(out, xlsx, criteria)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
