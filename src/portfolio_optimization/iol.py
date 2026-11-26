"""Public IOL (InvertirOnline) fixed-income quote acquisition and parsing.

Reimplements ONLY the public Argentine sovereign bond ("Bonos") and corporate
bond ("Obligaciones Negociables" / ONs) market panel quote extraction and
AL30/AL30D implicit MEP conversion referenced from the user's .gs inputs.

Intentionally excluded:
- Google Sheets / SpreadsheetApp / LockService / cell formatting / triggers
- UVA, dólar blue, GD30/GD30C CCL, equities, CEDEARs, and private IDs
- Historical price synthesis (IOL panels provide point-in-time snapshots only)
"""

from __future__ import annotations

import html as html_lib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from .errors import IOLQuoteError
from .schemas import InstrumentConfig, IOLConfig, IOLQuoteResult

_AR_TZ = ZoneInfo("America/Argentina/Buenos_Aires")

IOL_DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
}

_MISSING_TOKENS = frozenset({"", "-", "--", "—", "–", "n/a", "na", "s/d", "null", "none"})

# Exact normalized header names (Pass 1) and unambiguous substring aliases (Pass 2)
_HEADER_EXACT_MATCHERS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("symbol", ("simbolo", "especie", "ticker")),
    ("last_price", ("ultimo operado", "ultimo", "ultimo precio")),
    ("last_close", ("ultimo cierre", "cierre anterior", "cierre")),
    ("market_time", ("ultimo horario", "ultima actualizacion", "fecha y hora", "hora", "fecha", "horario")),
    ("daily_change_pct", ("variacion diaria", "variacion", "var. diaria", "var diaria")),
    ("bid_qty", ("cantidad compra", "cant. compra", "cant compra")),
    ("bid_price", ("precio compra", "compra")),
    ("ask_price", ("precio venta", "venta")),
    ("ask_qty", ("cantidad venta", "cant. venta", "cant venta")),
    ("high", ("maximo",)),
    ("low", ("minimo",)),
    ("volume_amount", ("monto operado", "monto")),
    ("tir", ("t. i. r.", "t.i.r.", "tir")),
    ("duration_md", ("duracion (md)", "duracion", "duration")),
)

_HEADER_ALIAS_MATCHERS: Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    # (key, required_substrings, forbidden_substrings)
    ("symbol", ("simbolo",), ()),
    ("market_time", ("horario", "hora", "fecha", "actualizacion"), ()),
    ("last_close", ("ultimo cierre", "cierre anterior"), ()),
    ("last_price", ("ultimo operado", "ultimo precio", "ultimo"), ("cierre", "hora", "horario", "fecha", "actualizacion", "var", "monto", "cant")),
    ("daily_change_pct", ("variacion diaria", "variacion"), ()),
    ("bid_qty", ("cantidad compra", "cant. compra", "cant compra"), ()),
    ("bid_price", ("precio compra", "compra"), ("cant",)),
    ("ask_qty", ("cantidad venta", "cant. venta", "cant venta"), ()),
    ("ask_price", ("precio venta", "venta"), ("cant",)),
    ("high", ("maximo",), ()),
    ("low", ("minimo",), ()),
    ("volume_amount", ("monto operado", "monto"), ()),
    ("tir", ("t. i. r.", "t.i.r.", "tir"), ()),
    ("duration_md", ("duracion", "duration"), ()),
)


@dataclass(frozen=True)
class IOLPanelRow:
    """Normalized row parsed from an IOL fixed-income HTML table."""

    ticker: str
    name: str
    last_price: Optional[float]
    daily_change_pct: Optional[float] = None
    bid_qty: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    ask_qty: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    last_close: Optional[float] = None
    volume_amount: Optional[float] = None
    tir: Optional[float] = None
    duration_md: Optional[float] = None
    market_timestamp: Optional[str] = None
    quote_unit: Optional[str] = None
    raw_symbol_cell: str = ""


@dataclass(frozen=True)
class IOLPanelSnapshot:
    """In-memory snapshot of a parsed IOL fixed-income panel (never persisted)."""

    panel_source: str  # "iol_bonos" | "iol_ons"
    url: str
    observed_at: str  # UTC ISO timestamp of HTTP fetch
    rows_by_ticker: Dict[str, IOLPanelRow]
    column_headers: Tuple[str, ...]
    warnings: Tuple[str, ...] = field(default_factory=tuple)


def normalize_text(text: str) -> str:
    """Strip diacritics, collapse whitespace, and lowercase for header matching."""
    cleaned = html_lib.unescape(str(text or ""))
    decomposed = unicodedata.normalize("NFD", cleaned)
    no_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", no_marks).strip().lower()


def clean_cell_text(raw: str) -> str:
    """Decode HTML entities and normalize whitespace inside a table cell."""
    text = html_lib.unescape(str(raw or ""))
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_argentine_number(raw: Any, *, locale_mode: str = "es-AR") -> Optional[float]:
    """Parse Argentine/IOL numeric strings into finite floats without decimal ambiguity.

    Rules:
    - Blank, ``"-"``, ``"—"``, ``"–"``, ``"N/A"``, or invalid strings return ``None`` (never ``0.0``).
    - Currency markers (``US$``, ``U$S``, ``ARS``, ``$``, ``%``) and spaces are stripped.
    - Mixed separators resolve by rightmost separator (``1.234,56`` -> ``1234.56``;
      ``1,234.56`` -> ``1234.56``) with strict thousands-group validation.
    - Single dot with leading zero (``0.125``) or non-3 fractional digits (``100.70``, ``62.5``)
      is ALWAYS parsed as a decimal point (fixing the old .gs parser bug).
    - Single dot matching Argentine thousands (``^[+-]?[1-9]\\d{0,2}\\.\\d{3}$``, e.g. ``154.000``)
      is parsed as thousands when ``locale_mode == "es-AR"``, and decimal when ``locale_mode == "en-US"``.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if math.isfinite(val) else None

    s = clean_cell_text(str(raw))
    if s.lower() in _MISSING_TOKENS:
        return None

    # Remove currency / percent prefixes and suffixes
    s = re.sub(r"(?i)(?:u\$s|us\$|usd|ars|\$|%)", "", s)
    s = re.sub(r"\s+", "", s)
    if not s or s.lower() in _MISSING_TOKENS:
        return None

    # Validate allowed characters only
    if not re.fullmatch(r"[+-]?[\d.,]+", s):
        return None

    sign = ""
    if s[0] in "+-":
        sign = s[0]
        body = s[1:]
    else:
        body = s

    if not body or body[0] in ".," or body[-1] in ".,":
        return None

    has_comma = "," in body
    has_dot = "." in body

    if has_comma and has_dot:
        last_comma = body.rfind(",")
        last_dot = body.rfind(".")
        if last_comma > last_dot:
            # Argentine format: 1.234.567,89
            int_part, dec_part = body.split(",", 1)
            if "," in dec_part or "." in dec_part:
                return None
            if not re.fullmatch(r"\d{1,3}(?:\.\d{3})+", int_part):
                return None
            normalized = f"{sign}{int_part.replace('.', '')}.{dec_part}"
        else:
            # Anglo format: 1,234,567.89
            int_part, dec_part = body.split(".", 1)
            if "." in dec_part or "," in dec_part:
                return None
            if not re.fullmatch(r"\d{1,3}(?:,\d{3})+", int_part):
                return None
            normalized = f"{sign}{int_part.replace(',', '')}.{dec_part}"
    elif has_comma:
        if body.count(",") == 1:
            # Argentine decimal comma: 100,70 or 5,25
            normalized = f"{sign}{body.replace(',', '.')}"
        else:
            if re.fullmatch(r"\d{1,3}(?:,\d{3})+", body):
                normalized = f"{sign}{body.replace(',', '')}"
            else:
                return None
    elif has_dot:
        dot_count = body.count(".")
        if dot_count > 1:
            if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", body):
                normalized = f"{sign}{body.replace('.', '')}"
            else:
                return None
        else:
            int_part, frac_part = body.split(".")
            if (
                locale_mode == "es-AR"
                and len(frac_part) == 3
                and re.fullmatch(r"[1-9]\d{0,2}", int_part)
            ):
                # Argentine thousands grouping (e.g., 154.000 -> 154000),
                # while preserving 0.125 -> 0.125 and 100.70 -> 100.70
                normalized = f"{sign}{int_part}{frac_part}"
            else:
                normalized = f"{sign}{int_part}.{frac_part}"
    else:
        normalized = f"{sign}{body}"

    try:
        val = float(normalized)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def parse_argentine_percentage(raw: Any, *, locale_mode: str = "es-AR") -> Optional[float]:
    """Parse Argentine percentage string (e.g. ``'5,25%'`` -> ``0.0525``)."""
    num = parse_argentine_number(raw, locale_mode=locale_mode)
    return None if num is None else num / 100.0


def extract_ticker_and_name(raw_symbol_text: str) -> Tuple[str, str]:
    """Extract canonical ticker symbol and trailing instrument description from IOL cell."""
    cleaned = clean_cell_text(raw_symbol_text)
    if not cleaned:
        return "", ""
    tokens = cleaned.split()
    ticker = ""
    for tok in tokens:
        cand = re.sub(r"^[^A-Za-z0-9.]+|[^A-Za-z0-9.]+$", "", tok)
        if re.fullmatch(r"[A-Z0-9][A-Z0-9.]{1,11}", cand):
            ticker = cand
            break
    if not ticker:
        return "", cleaned
    pos = cleaned.find(ticker)
    name = cleaned[pos + len(ticker):].strip() if pos >= 0 else ""
    return ticker.upper(), name


def _map_header_columns(header_cells: List[str]) -> Dict[str, int]:
    """Map semantic column keys by exact normalized header text first, then unambiguous aliases.

    Raises ``IOLQuoteError`` on ambiguous required-column collisions so non-price columns
    (e.g. ``"Último Horario"``) can never hijack ``last_price``.
    """
    norm_headers = [normalize_text(h) for h in header_cells]
    mapping: Dict[str, int] = {}
    used_cols: set[int] = set()

    # Pass 1: exact normalized header matches (strictest priority)
    for key, exact_names in _HEADER_EXACT_MATCHERS:
        exact_matches = [idx for idx, htext in enumerate(norm_headers) if htext in exact_names and idx not in used_cols]
        if len(exact_matches) == 1:
            mapping[key] = exact_matches[0]
            used_cols.add(exact_matches[0])
        elif len(exact_matches) > 1:
            if key in ("symbol", "last_price"):
                raise IOLQuoteError(
                    f"Ambiguous IOL table header collision for required column '{key}': "
                    f"{[header_cells[i] for i in exact_matches]}.",
                    details={"ambiguous_key": key, "matching_headers": [header_cells[i] for i in exact_matches]},
                )
            mapping[key] = exact_matches[0]
            used_cols.add(exact_matches[0])

    # Pass 2: unambiguous substring aliases on remaining unmapped keys & columns
    for key, aliases, forbidden in _HEADER_ALIAS_MATCHERS:
        if key in mapping:
            continue
        alias_matches = [
            idx
            for idx, htext in enumerate(norm_headers)
            if idx not in used_cols
            and htext
            and any(alias in htext for alias in aliases)
            and not any(bad in htext for bad in forbidden)
        ]
        if len(alias_matches) == 1:
            mapping[key] = alias_matches[0]
            used_cols.add(alias_matches[0])
        elif len(alias_matches) > 1:
            if key in ("symbol", "last_price"):
                raise IOLQuoteError(
                    f"Ambiguous IOL table header collision for required column '{key}': "
                    f"{[header_cells[i] for i in alias_matches]}.",
                    details={"ambiguous_key": key, "matching_headers": [header_cells[i] for i in alias_matches]},
                )
            mapping[key] = alias_matches[0]
            used_cols.add(alias_matches[0])

    return mapping


def _is_fixed_income_iol_header(header_cells: List[str]) -> bool:
    joined = " | ".join(normalize_text(c) for c in header_cells)
    has_base = ("simbolo" in joined) and ("ultimo" in joined) and ("variacion" in joined)
    has_rf = ("duracion" in joined or "duration" in joined) and (
        "t. i. r." in joined or "t.i.r." in joined or "tir" in joined
    )
    return has_base and has_rf


def parse_iol_fixed_income_table(
    html_text: str,
    *,
    panel_source: str,
    url: str = "",
    observed_at: Optional[str] = None,
    min_valid_rows: int = 1,
) -> IOLPanelSnapshot:
    """Parse an IOL Bonos or ONs HTML page using BeautifulSoup and header validation.

    Raises ``IOLQuoteError`` on missing fixed-income table, unexpected headers,
    or insufficient valid quote rows. Never invents data.
    """
    if not html_text or not str(html_text).strip():
        raise IOLQuoteError(
            f"Empty HTML response received from IOL panel '{panel_source}'.",
            details={"panel_source": panel_source, "url": url},
        )

    soup = BeautifulSoup(html_text, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        raise IOLQuoteError(
            f"No HTML <table> elements found in IOL panel '{panel_source}'. Page structure may have changed.",
            details={"panel_source": panel_source, "url": url},
        )

    candidates: List[Tuple[int, List[str], Dict[str, int], List[List[str]]]] = []

    for table in tables:
        tr_elements = table.find_all("tr")
        if not tr_elements:
            continue

        extracted_rows: List[List[str]] = []
        for tr in tr_elements:
            cells = tr.find_all(["th", "td"])
            if not cells:
                continue
            row_texts = [clean_cell_text(cell.get_text(separator=" ", strip=True)) for cell in cells]
            if any(row_texts):
                extracted_rows.append(row_texts)

        if len(extracted_rows) < 2:
            continue

        header_row = extracted_rows[0]
        if not _is_fixed_income_iol_header(header_row):
            continue

        col_map = _map_header_columns(header_row)
        if "symbol" not in col_map or "last_price" not in col_map:
            continue

        candidates.append((len(extracted_rows) - 1, header_row, col_map, extracted_rows[1:]))

    if not candidates:
        raise IOLQuoteError(
            f"Expected IOL fixed-income table (headers: Símbolo, Último, Variación, TIR, Duración) not found in '{panel_source}'.",
            details={"panel_source": panel_source, "url": url, "tables_inspected": len(tables)},
        )

    # Choose the matching fixed-income table with the most data rows (matching Actualizar Datos.gs)
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, header_row, col_map, data_rows = candidates[0]

    warnings: List[str] = []
    non_empty_headers = [h for h in header_row if h.strip()]
    if len(non_empty_headers) != 13:
        warnings.append(
            f"IOL panel '{panel_source}' returned {len(non_empty_headers)} non-empty columns instead of the standard 13; columns were mapped by header name."
        )

    rows_by_ticker: Dict[str, IOLPanelRow] = {}

    def _cell(row: List[str], key: str) -> Optional[str]:
        idx = col_map.get(key)
        if idx is None or idx >= len(row):
            return None
        return row[idx]

    for row in data_rows:
        raw_sym = _cell(row, "symbol")
        if not raw_sym:
            continue
        ticker, name = extract_ticker_and_name(raw_sym)
        if not ticker or len(ticker) > 15:
            continue

        market_ts_raw = _cell(row, "market_time")
        market_ts = market_ts_raw if market_ts_raw and market_ts_raw.lower() not in _MISSING_TOKENS else None

        parsed_row = IOLPanelRow(
            ticker=ticker,
            name=name,
            last_price=parse_argentine_number(_cell(row, "last_price")),
            daily_change_pct=parse_argentine_percentage(_cell(row, "daily_change_pct")),
            bid_qty=parse_argentine_number(_cell(row, "bid_qty")),
            bid_price=parse_argentine_number(_cell(row, "bid_price")),
            ask_price=parse_argentine_number(_cell(row, "ask_price")),
            ask_qty=parse_argentine_number(_cell(row, "ask_qty")),
            high=parse_argentine_number(_cell(row, "high")),
            low=parse_argentine_number(_cell(row, "low")),
            last_close=parse_argentine_number(_cell(row, "last_close")),
            volume_amount=parse_argentine_number(_cell(row, "volume_amount")),
            tir=parse_argentine_percentage(_cell(row, "tir")),
            duration_md=parse_argentine_number(_cell(row, "duration_md")),
            market_timestamp=market_ts,
            raw_symbol_cell=raw_sym,
        )

        if ticker in rows_by_ticker:
            warnings.append(f"Duplicate symbol '{ticker}' encountered in IOL panel '{panel_source}'; keeping latest valid row.")
            # Prefer a row with a valid positive last_price over a missing one
            if parsed_row.last_price is not None and parsed_row.last_price > 0:
                rows_by_ticker[ticker] = parsed_row
        else:
            rows_by_ticker[ticker] = parsed_row

    valid_price_rows = sum(
        1 for r in rows_by_ticker.values() if r.last_price is not None and math.isfinite(r.last_price) and r.last_price > 0
    )
    if len(rows_by_ticker) < min_valid_rows or valid_price_rows == 0:
        raise IOLQuoteError(
            f"IOL panel '{panel_source}' contains insufficient valid rows ({valid_price_rows} positive quotes out of {len(rows_by_ticker)} rows).",
            details={"panel_source": panel_source, "url": url, "rows_parsed": len(rows_by_ticker), "valid_quotes": valid_price_rows},
        )

    obs_ts = observed_at or datetime.now(timezone.utc).isoformat()
    return IOLPanelSnapshot(
        panel_source=panel_source,
        url=url,
        observed_at=obs_ts,
        rows_by_ticker=rows_by_ticker,
        column_headers=tuple(header_row),
        warnings=tuple(warnings),
    )


def fetch_iol_panel(
    panel_source: str,
    *,
    iol_config: Optional[IOLConfig] = None,
    session: Optional[requests.Session] = None,
) -> IOLPanelSnapshot:
    """Fetch and parse a public IOL fixed-income panel (``iol_bonos`` or ``iol_ons``).

    Enforces strict TLS verification (``verify=True``), explicit timeout,
    HTTP 2xx validation, and explicit errors with no access-control evasion.
    """
    cfg = iol_config or IOLConfig()
    if panel_source == "iol_bonos":
        url = cfg.bonos_url
    elif panel_source == "iol_ons":
        url = cfg.ons_url
    else:
        raise IOLQuoteError(f"Unsupported IOL panel_source '{panel_source}'. Expected 'iol_bonos' or 'iol_ons'.")

    owns_session = session is None
    http = requests.Session() if owns_session else session
    try:
        try:
            response = http.get(
                url,
                headers=IOL_DEFAULT_HEADERS,
                timeout=cfg.timeout_seconds,
                allow_redirects=True,
                verify=True,
            )
        except requests.RequestException as exc:
            raise IOLQuoteError(
                f"Network/HTTP error querying IOL panel '{panel_source}': {type(exc).__name__}",
                details={"panel_source": panel_source, "url": url, "error_type": type(exc).__name__},
            ) from exc

        if response.status_code < 200 or response.status_code >= 300:
            raise IOLQuoteError(
                f"IOL panel '{panel_source}' returned HTTP {response.status_code}.",
                details={"panel_source": panel_source, "url": url, "status_code": response.status_code},
            )

        observed_at = datetime.now(timezone.utc).isoformat()
        # Respect UTF-8 decoding as in Actualizar Datos.gs
        html_text = response.content.decode("utf-8", errors="replace") if isinstance(response.content, (bytes, bytearray)) else str(response.text)
    finally:
        if owns_session:
            http.close()

    return parse_iol_fixed_income_table(
        html_text,
        panel_source=panel_source,
        url=url,
        observed_at=observed_at,
        min_valid_rows=cfg.min_valid_rows,
    )


def _valid_positive(val: Optional[float]) -> bool:
    return val is not None and isinstance(val, (int, float)) and math.isfinite(val) and float(val) > 0.0


def _parse_iol_market_timestamp(raw_ts: Optional[str]) -> Optional[datetime]:
    """Parse an IOL market timestamp into a UTC timezone-aware ``datetime`` (or ``None`` if absent).

    Supports ISO-8601 (with or without tz) and Argentine formats (``dd/mm/yyyy HH:MM[:SS]``,
    ``dd-mm-yyyy HH:MM[:SS]``, ``dd/mm/yyyy``). Naive timestamps are interpreted in
    ``America/Argentina/Buenos_Aires``. Raises ``IOLQuoteError`` on malformed strings.
    """
    if raw_ts is None:
        return None
    text = clean_cell_text(str(raw_ts))
    if not text or text.lower() in _MISSING_TOKENS:
        return None

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_AR_TZ)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    for fmt in (
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%d/%m/%Y",
        "%d-%m-%Y",
    ):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=_AR_TZ)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue

    raise IOLQuoteError(
        f"Unparseable IOL market_timestamp '{text}'; cannot verify quote contemporaneity.",
        details={"raw_market_timestamp": text},
    )


def _normalize_mep_leg_to_per_100_vn(
    ticker: str,
    price: float,
    quote_unit: Optional[str],
) -> Tuple[float, str]:
    """Normalize a verified AL30/AL30D leg price to per-100-original-nominal when unit metadata is explicit."""
    if quote_unit is None:
        return price, "panel_default_per_100_original_nominal"
    qu = str(quote_unit).strip().lower()
    if qu == "unknown":
        raise IOLQuoteError(
            f"Quote unit for '{ticker}' is 'unknown'; refusing implicit MEP without verified nominal base.",
            asset_id=ticker,
            details={"ticker": ticker, "quote_unit": quote_unit},
        )
    if qu == "per_100_original_nominal":
        return price, "per_100_original_nominal"
    if qu in ("per_1_original_nominal", "per_unit_1_vn"):
        return price * 100.0, "normalized_from_per_1_original_nominal"
    raise IOLQuoteError(
        f"Unsupported or incompatible quote_unit '{quote_unit}' on '{ticker}' for implicit MEP.",
        asset_id=ticker,
        details={"ticker": ticker, "quote_unit": quote_unit},
    )


def compute_implicit_mep_al30(
    bonos_snapshot: IOLPanelSnapshot,
    *,
    reference_observed_at: Optional[str] = None,
    max_time_skew_seconds: Optional[float] = None,
    max_market_time_diff_seconds: Optional[float] = None,
    max_quote_age_seconds: Optional[float] = None,
    allow_missing_market_timestamp: Optional[bool] = None,
    mep_plausible_min: Optional[float] = None,
    mep_plausible_max: Optional[float] = None,
    iol_config: Optional[IOLConfig] = None,
) -> Dict[str, Any]:
    """Compute implicit ARS/USD MEP rate strictly from AL30 / AL30D in the Bonos panel.

    Validates exact symbol match, issuer identity compatibility, positive finite prices,
    HTTP snapshot contemporaneity, real ``market_timestamp`` contemporaneity/age when
    exposed (with explicit conservative policy when absent), verified nominal units,
    and configurable plausibility bounds. Never invents timestamps or falls back to
    non-AL30 FX sources.
    """
    cfg = iol_config or IOLConfig()
    eff_http_skew = cfg.max_mep_time_skew_seconds if max_time_skew_seconds is None else float(max_time_skew_seconds)
    eff_mkt_diff = cfg.max_mep_market_time_diff_seconds if max_market_time_diff_seconds is None else float(max_market_time_diff_seconds)
    eff_max_age = cfg.max_mep_quote_age_seconds if max_quote_age_seconds is None else float(max_quote_age_seconds)
    eff_allow_missing_ts = (
        cfg.allow_mep_without_market_timestamp
        if allow_missing_market_timestamp is None
        else bool(allow_missing_market_timestamp)
    )
    eff_min_mep = cfg.mep_plausible_min if mep_plausible_min is None else float(mep_plausible_min)
    eff_max_mep = cfg.mep_plausible_max if mep_plausible_max is None else float(mep_plausible_max)

    al30 = bonos_snapshot.rows_by_ticker.get("AL30")
    al30d = bonos_snapshot.rows_by_ticker.get("AL30D")

    if al30 is None or not _valid_positive(al30.last_price):
        raise IOLQuoteError(
            "Cannot compute implicit MEP: valid positive last price for 'AL30' not found in IOL Bonos panel.",
            asset_id="AL30",
            details={"al30_present": al30 is not None, "al30_last": None if al30 is None else al30.last_price},
        )
    if al30d is None or not _valid_positive(al30d.last_price):
        raise IOLQuoteError(
            "Cannot compute implicit MEP: valid positive last price for 'AL30D' not found in IOL Bonos panel.",
            asset_id="AL30D",
            details={"al30d_present": al30d is not None, "al30d_last": None if al30d is None else al30d.last_price},
        )

    if not _names_look_compatible(al30.name, al30d.name):
        raise IOLQuoteError(
            f"AL30 ('{al30.name}') and AL30D ('{al30d.name}') appear to belong to different instruments; refusing implicit MEP.",
            details={"al30_name": al30.name, "al30d_name": al30d.name},
        )

    t_bonos_obs: Optional[datetime] = None
    try:
        t_bonos_obs = datetime.fromisoformat(bonos_snapshot.observed_at.replace("Z", "+00:00"))
        if t_bonos_obs.tzinfo is None:
            t_bonos_obs = t_bonos_obs.replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise IOLQuoteError("Invalid snapshot observed_at timestamp format while validating MEP.") from exc

    if reference_observed_at:
        try:
            t_ref = datetime.fromisoformat(reference_observed_at.replace("Z", "+00:00"))
            if t_ref.tzinfo is None:
                t_ref = t_ref.replace(tzinfo=timezone.utc)
            skew = abs((t_bonos_obs - t_ref).total_seconds())
            if skew > eff_http_skew:
                raise IOLQuoteError(
                    f"AL30/AL30D MEP snapshot is not contemporaneous with target quote (fetch skew {skew:.1f}s > {eff_http_skew:.1f}s).",
                    details={"skew_seconds": skew, "max_skew_seconds": eff_http_skew},
                )
        except ValueError as exc:
            raise IOLQuoteError("Invalid observation timestamp format while validating MEP contemporaneity.") from exc

    # Market timestamp validation (real trade/update timestamps distinct from HTTP observed_at)
    ts_al30 = _parse_iol_market_timestamp(al30.market_timestamp)
    ts_al30d = _parse_iol_market_timestamp(al30d.market_timestamp)
    unit_warnings: List[str] = []
    simultaneity_verified = False
    market_time_diff_seconds: Optional[float] = None

    if ts_al30 is not None and ts_al30d is not None:
        market_time_diff_seconds = abs((ts_al30 - ts_al30d).total_seconds())
        if market_time_diff_seconds > eff_mkt_diff:
            raise IOLQuoteError(
                f"AL30 ({al30.market_timestamp}) and AL30D ({al30d.market_timestamp}) market timestamps diverge by "
                f"{market_time_diff_seconds:.1f}s (> allowed {eff_mkt_diff:.1f}s); refusing non-contemporaneous implicit MEP.",
                details={
                    "al30_market_timestamp": al30.market_timestamp,
                    "al30d_market_timestamp": al30d.market_timestamp,
                    "market_time_diff_seconds": market_time_diff_seconds,
                    "max_market_time_diff_seconds": eff_mkt_diff,
                },
            )
        if eff_max_age is not None:
            for leg_name, leg_ts in (("AL30", ts_al30), ("AL30D", ts_al30d)):
                age = (t_bonos_obs - leg_ts).total_seconds()
                if age > eff_max_age:
                    raise IOLQuoteError(
                        f"IOL quote '{leg_name}' market timestamp ({leg_ts.isoformat()}) is stale by {age:.1f}s (> allowed {eff_max_age:.1f}s).",
                        asset_id=leg_name,
                        details={"age_seconds": age, "max_quote_age_seconds": eff_max_age},
                    )
        simultaneity_verified = True
    elif (ts_al30 is None) != (ts_al30d is None):
        raise IOLQuoteError(
            f"Asymmetric market timestamps for AL30 ({al30.market_timestamp}) and AL30D ({al30d.market_timestamp}); "
            "cannot verify contemporaneity of implicit MEP.",
            details={
                "al30_market_timestamp": al30.market_timestamp,
                "al30d_market_timestamp": al30d.market_timestamp,
            },
        )
    else:
        if not eff_allow_missing_ts:
            raise IOLQuoteError(
                "AL30 and AL30D lack market trade timestamps in the IOL panel; simultaneity cannot be verified. "
                "Set IOLConfig(allow_mep_without_market_timestamp=True) to explicitly accept unverified simultaneity risk.",
                details={
                    "simultaneity_verified": False,
                    "al30_market_timestamp": None,
                    "al30d_market_timestamp": None,
                },
            )
        unit_warnings.append(
            "AL30 and AL30D market timestamps are absent in the IOL panel; MEP simultaneity is NOT verified "
            "(explicitly accepted via allow_mep_without_market_timestamp=True; observed_at reflects fetch time only)."
        )

    # Nominal base / unit verification and normalization (only when explicitly verified)
    raw_al30_px = float(al30.last_price)
    raw_al30d_px = float(al30d.last_price)
    if (al30.quote_unit is None) != (al30d.quote_unit is None):
        raise IOLQuoteError(
            f"Only one leg declares quote_unit (AL30={al30.quote_unit}, AL30D={al30d.quote_unit}); "
            "refusing implicit MEP without comparable verified nominal bases.",
            details={"al30_quote_unit": al30.quote_unit, "al30d_quote_unit": al30d.quote_unit},
        )

    al30_px, al30_unit_note = _normalize_mep_leg_to_per_100_vn("AL30", raw_al30_px, al30.quote_unit)
    al30d_px, al30d_unit_note = _normalize_mep_leg_to_per_100_vn("AL30D", raw_al30d_px, al30d.quote_unit)
    if al30_unit_note != al30d_unit_note and al30.quote_unit is not None and al30d.quote_unit is not None:
        unit_warnings.append(
            f"Normalized AL30 ({al30.quote_unit}) and AL30D ({al30d.quote_unit}) to common per-100-original-nominal basis using verified metadata."
        )

    mep_rate = al30_px / al30d_px
    if not math.isfinite(mep_rate) or mep_rate <= 0.0:
        raise IOLQuoteError(
            f"Computed implicit MEP rate (AL30={al30_px} / AL30D={al30d_px}) is not finite and positive.",
            details={"al30": al30_px, "al30d": al30d_px},
        )

    if al30_px < al30d_px:
        raise IOLQuoteError(
            f"AL30 ({al30_px}) < AL30D ({al30d_px}) suggests non-comparable quote units or corrupted feed; refusing implicit MEP.",
            details={"al30": al30_px, "al30d": al30d_px, "mep_rate": mep_rate},
        )

    # Guard against unverified 100x nominal base mismatch (e.g., per-100 VN vs per-1 VN) and implausible ratios
    if al30.quote_unit is None and al30d.quote_unit is None:
        if (al30d_px < 5.0 and al30_px > 5000.0) or (al30_px < 5000.0 and al30d_px >= 20.0 and mep_rate < eff_min_mep):
            raise IOLQuoteError(
                f"AL30 ({al30_px}) vs AL30D ({al30d_px}) produces MEP={mep_rate:.4f}, indicating non-comparable nominal units "
                "(e.g. per-100 VN vs per-1 VN) without verified unit metadata; refusing implicit MEP.",
                details={"al30": al30_px, "al30d": al30d_px, "mep_rate": mep_rate},
            )

    if mep_rate < eff_min_mep or mep_rate > eff_max_mep:
        raise IOLQuoteError(
            f"Computed implicit MEP ({mep_rate:.4f} ARS/USD from AL30={al30_px} / AL30D={al30d_px}) falls outside "
            f"configured plausible bounds [{eff_min_mep:g}, {eff_max_mep:g}], suggesting nominal-unit mismatch or feed corruption.",
            details={
                "al30": al30_px,
                "al30d": al30d_px,
                "mep_rate": mep_rate,
                "mep_plausible_min": eff_min_mep,
                "mep_plausible_max": eff_max_mep,
            },
        )

    return {
        "al30_price_ars": al30_px,
        "al30d_price_usd": al30d_px,
        "mep_rate_ars_per_usd": mep_rate,
        "observed_at": bonos_snapshot.observed_at,
        "al30_market_timestamp": al30.market_timestamp,
        "al30d_market_timestamp": al30d.market_timestamp,
        "simultaneity_verified": simultaneity_verified,
        "market_time_diff_seconds": market_time_diff_seconds,
        "al30_unit_basis": al30_unit_note,
        "al30d_unit_basis": al30d_unit_note,
        "warnings": unit_warnings,
    }


def _names_look_compatible(name_a: str, name_b: str) -> bool:
    """Conservative compatibility check when both IOL rows expose issuer/instrument names."""
    if not name_a or not name_b:
        return True
    na = normalize_text(name_a)
    nb = normalize_text(name_b)
    stop = {"d", "c", "usd", "ars", "u$s", "us$", "bono", "on", "clase", "ley", "arg", "ny", "sa", "corp", "emisor"}
    toks_a = {t for t in re.findall(r"[a-z0-9]+", na) if len(t) >= 2 and t not in stop}
    toks_b = {t for t in re.findall(r"[a-z0-9]+", nb) if len(t) >= 2 and t not in stop}
    if not toks_a or not toks_b:
        return True
    return len(toks_a.intersection(toks_b)) >= 1


def _resolve_species_candidates(
    requested_ticker: str,
    panel_snapshot: IOLPanelSnapshot,
    explicit_species_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Optional[str]]:
    """Determine exact candidate symbols for D, C, and ARS with issuer compatibility across all branches."""
    tkr = requested_ticker.strip().upper()
    rows = panel_snapshot.rows_by_ticker

    if explicit_species_map:
        return {
            "D": explicit_species_map.get("D"),
            "C": explicit_species_map.get("C"),
            "ARS": explicit_species_map.get("ARS"),
        }

    def _compat(ref_row: Optional[IOLPanelRow], cand_sym: Optional[str]) -> Optional[str]:
        if cand_sym is None or cand_sym not in rows:
            return None
        if ref_row is not None and not _names_look_compatible(ref_row.name, rows[cand_sym].name):
            return None
        return cand_sym

    if len(tkr) >= 4 and tkr[-1] in ("D", "C"):
        stem = tkr[:-1]
        req_row = rows.get(tkr)
        ars_candidate: Optional[str] = None
        d_candidate: Optional[str] = None
        c_candidate: Optional[str] = None

        if stem in rows:
            ref = req_row or rows[stem]
            ars_candidate = _compat(ref, stem)
            d_candidate = _compat(ref, f"{stem}D")
            c_candidate = _compat(ref, f"{stem}C")
            # Preserve the explicitly requested symbol itself as its declared species if present
            if req_row is not None:
                if tkr.endswith("D"):
                    d_candidate = tkr
                else:
                    c_candidate = tkr
        elif panel_snapshot.panel_source == "iol_ons" and len(tkr) == 5 and f"{stem}O" in rows:
            ref = req_row or rows[f"{stem}O"]
            ars_candidate = _compat(ref, f"{stem}O")
            d_candidate = _compat(ref, f"{stem}D")
            c_candidate = _compat(ref, f"{stem}C")
            if req_row is not None:
                if tkr.endswith("D"):
                    d_candidate = tkr
                else:
                    c_candidate = tkr
        elif tkr in rows:
            if re.fullmatch(r"[A-Z]{2,4}\d{1,2}[DC]", tkr) or (panel_snapshot.panel_source == "iol_ons" and len(tkr) == 5):
                if tkr.endswith("D"):
                    d_candidate = tkr
                    c_candidate = _compat(req_row, f"{stem}C")
                else:
                    c_candidate = tkr
                    d_candidate = _compat(req_row, f"{stem}D")
            else:
                raise IOLQuoteError(
                    f"Ambiguous ticker '{tkr}' ending in '{tkr[-1]}': cannot confirm whether it is a USD species or an ARS root symbol. Provide explicit 'iol_species_map'.",
                    asset_id=tkr,
                )
        else:
            raise IOLQuoteError(
                f"Ticker '{tkr}' not found in IOL panel '{panel_snapshot.panel_source}'. Provide explicit 'iol_species_map' if species symbols differ.",
                asset_id=tkr,
            )

        return {"D": d_candidate, "C": c_candidate, "ARS": ars_candidate}

    if panel_snapshot.panel_source == "iol_ons" and len(tkr) == 5 and tkr.endswith("O"):
        stem = tkr[:-1]
        base_row = rows.get(tkr)
        d_sym = _compat(base_row, f"{stem}D")
        c_sym = _compat(base_row, f"{stem}C")
        return {"D": d_sym, "C": c_sym, "ARS": tkr if tkr in rows else None}

    base_row = rows.get(tkr)
    d_sym = _compat(base_row, f"{tkr}D")
    c_sym = _compat(base_row, f"{tkr}C")
    return {"D": d_sym, "C": c_sym, "ARS": tkr if tkr in rows else None}


def _collect_quote_quality_warnings(
    row: IOLPanelRow,
    iol_config: IOLConfig,
) -> List[str]:
    warnings: List[str] = []
    if row.market_timestamp is None:
        warnings.append(
            "Source panel did not expose a market trade timestamp; 'observed_at' reflects fetch time only."
        )
    if row.volume_amount is None or row.volume_amount < iol_config.low_liquidity_amount_threshold:
        warnings.append(
            f"Low or unreported liquidity for '{row.ticker}' (monto operado={row.volume_amount})."
        )
    if _valid_positive(row.bid_price) and _valid_positive(row.ask_price):
        mid = 0.5 * (float(row.bid_price) + float(row.ask_price))
        if mid > 0:
            spread_bps = (float(row.ask_price) - float(row.bid_price)) / mid * 10000.0
            if spread_bps > iol_config.wide_spread_bps_threshold:
                warnings.append(
                    f"Wide bid-ask spread for '{row.ticker}' ({spread_bps:.0f} bps)."
                )
    return warnings


def resolve_iol_bond_quote_usd(
    *,
    instrument_id: str,
    ticker: str,
    panel_snapshot: IOLPanelSnapshot,
    bonos_snapshot: Optional[IOLPanelSnapshot] = None,
    explicit_species_map: Optional[Dict[str, str]] = None,
    quote_unit: str = "per_100_original_nominal",
    iol_config: Optional[IOLConfig] = None,
) -> IOLQuoteResult:
    """Resolve an Argentine bond or ON quote in USD following strict priority:

    1. Species ``D`` (USD local / MEP) with finite positive ``last_price``.
    2. Species ``C`` (USD cable / CCL) with finite positive ``last_price``,
       explicitly tagged as ``USD_CABLE_C`` (never silently equated 1:1 to USD local).
    3. ``ARS`` last price divided by implicit MEP ``AL30 / AL30D`` from the Bonos panel.

    Raises ``IOLQuoteError`` if no valid USD quote can be obtained. Never invents prices.
    """
    cfg = iol_config or IOLConfig()
    iid = instrument_id.strip().upper()
    tkr = ticker.strip().upper()

    candidates = _resolve_species_candidates(tkr, panel_snapshot, explicit_species_map)
    rows = panel_snapshot.rows_by_ticker

    d_sym = candidates.get("D")
    if d_sym and d_sym in rows and _valid_positive(rows[d_sym].last_price):
        row = rows[d_sym]
        warns = list(panel_snapshot.warnings) + _collect_quote_quality_warnings(row, cfg)
        return IOLQuoteResult(
            instrument_id=iid,
            requested_ticker=tkr,
            species_used=row.ticker,
            panel_source=panel_snapshot.panel_source,
            original_currency="USD",
            final_currency="USD",
            usd_settlement_type="USD_LOCAL_D",
            conversion_method="D",
            price_original=float(row.last_price),
            price_usd=float(row.last_price),
            quote_unit=quote_unit,
            observed_at=panel_snapshot.observed_at,
            market_timestamp=row.market_timestamp,
            instrument_name=row.name,
            bid=row.bid_price,
            ask=row.ask_price,
            last_close=row.last_close,
            daily_change_pct=row.daily_change_pct,
            volume_amount=row.volume_amount,
            published_tir=row.tir,
            published_md=row.duration_md,
            mep_reference=None,
            warnings=warns,
        )

    c_sym = candidates.get("C")
    if c_sym and c_sym in rows and _valid_positive(rows[c_sym].last_price):
        row = rows[c_sym]
        warns = list(panel_snapshot.warnings) + _collect_quote_quality_warnings(row, cfg)
        warns.append(
            f"Species D unavailable for '{iid}'; using USD cable species '{row.ticker}' (C). "
            "USD cable (C) is economically distinct from local USD (D) and is not assumed 1:1 equivalent."
        )
        return IOLQuoteResult(
            instrument_id=iid,
            requested_ticker=tkr,
            species_used=row.ticker,
            panel_source=panel_snapshot.panel_source,
            original_currency="USD_CABLE",
            final_currency="USD",
            usd_settlement_type="USD_CABLE_C",
            conversion_method="C",
            price_original=float(row.last_price),
            price_usd=float(row.last_price),
            quote_unit=quote_unit,
            observed_at=panel_snapshot.observed_at,
            market_timestamp=row.market_timestamp,
            instrument_name=row.name,
            bid=row.bid_price,
            ask=row.ask_price,
            last_close=row.last_close,
            daily_change_pct=row.daily_change_pct,
            volume_amount=row.volume_amount,
            published_tir=row.tir,
            published_md=row.duration_md,
            mep_reference=None,
            warnings=warns,
        )

    ars_sym = candidates.get("ARS")
    if ars_sym and ars_sym in rows and _valid_positive(rows[ars_sym].last_price):
        row = rows[ars_sym]
        ref_bonos = bonos_snapshot if bonos_snapshot is not None else (
            panel_snapshot if panel_snapshot.panel_source == "iol_bonos" else None
        )
        if ref_bonos is None:
            raise IOLQuoteError(
                f"Only ARS quote '{row.ticker}' exists for '{iid}', but IOL Bonos panel is unavailable to compute AL30/AL30D MEP.",
                asset_id=iid,
                details={"species_candidates": candidates},
            )
        mep = compute_implicit_mep_al30(
            ref_bonos,
            reference_observed_at=panel_snapshot.observed_at,
            iol_config=cfg,
        )
        price_ars = float(row.last_price)
        mep_rate = float(mep["mep_rate_ars_per_usd"])
        price_usd = price_ars / mep_rate
        if not math.isfinite(price_usd) or price_usd <= 0.0:
            raise IOLQuoteError(
                f"Invalid USD equivalent computed for '{iid}' from ARS={price_ars} / MEP={mep_rate}.",
                asset_id=iid,
            )

        warns = list(panel_snapshot.warnings) + list(mep.get("warnings", [])) + _collect_quote_quality_warnings(row, cfg)
        warns.append(
            f"Converted ARS species '{row.ticker}' ({price_ars:.4f} ARS) to USD equivalent ({price_usd:.4f} USD) "
            f"using implicit AL30/AL30D MEP={mep_rate:.4f}. Point-in-time snapshot only (never applied to history)."
        )
        return IOLQuoteResult(
            instrument_id=iid,
            requested_ticker=tkr,
            species_used=row.ticker,
            panel_source=panel_snapshot.panel_source,
            original_currency="ARS",
            final_currency="USD",
            usd_settlement_type="USD_MEP_IMPLIED",
            conversion_method="ARS/MEP",
            price_original=price_ars,
            price_usd=price_usd,
            quote_unit=quote_unit,
            observed_at=panel_snapshot.observed_at,
            market_timestamp=row.market_timestamp,
            instrument_name=row.name,
            bid=row.bid_price,
            ask=row.ask_price,
            last_close=row.last_close,
            daily_change_pct=row.daily_change_pct,
            volume_amount=row.volume_amount,
            published_tir=row.tir,
            published_md=row.duration_md,
            mep_reference=mep,
            warnings=warns,
        )

    raise IOLQuoteError(
        f"No valid positive quote found in IOL panel '{panel_snapshot.panel_source}' for '{iid}' "
        f"(checked D={d_sym}, C={c_sym}, ARS={ars_sym}).",
        asset_id=iid,
        details={"panel_source": panel_snapshot.panel_source, "candidates": candidates},
    )


def fetch_iol_quotes_for_instruments(
    instruments: Iterable[InstrumentConfig],
    *,
    iol_config: Optional[IOLConfig] = None,
    session: Optional[requests.Session] = None,
    preloaded_snapshots: Optional[Dict[str, IOLPanelSnapshot]] = None,
) -> Dict[str, IOLQuoteResult]:
    """Fetch public IOL USD quotes for all instruments whose ``quote_source`` is IOL.

    Only fetches the needed panel(s) per request (in-memory, zero persistence).
    Automatically fetches ``iol_bonos`` if an ON needs ARS/MEP fallback via AL30/AL30D.
    """
    cfg = iol_config or IOLConfig()
    iol_instruments = [
        inst for inst in instruments if inst.quote_source in ("iol_bonos", "iol_ons")
    ]
    if not iol_instruments:
        return {}

    snapshots: Dict[str, IOLPanelSnapshot] = dict(preloaded_snapshots or {})
    needed_panels = {inst.quote_source for inst in iol_instruments}

    for panel in sorted(needed_panels):
        if panel not in snapshots:
            snapshots[panel] = fetch_iol_panel(panel, iol_config=cfg, session=session)

    results: Dict[str, IOLQuoteResult] = {}
    for inst in iol_instruments:
        panel_snap = snapshots[inst.quote_source]
        quote_unit = (
            inst.fixed_income_terms.price_convention
            if inst.fixed_income_terms is not None
            else "per_100_original_nominal"
        )
        candidates = _resolve_species_candidates(inst.ticker, panel_snap, inst.iol_species_map)
        needs_mep_bonos = (
            inst.quote_source == "iol_ons"
            and not (candidates.get("D") and candidates["D"] in panel_snap.rows_by_ticker and _valid_positive(panel_snap.rows_by_ticker[candidates["D"]].last_price))
            and not (candidates.get("C") and candidates["C"] in panel_snap.rows_by_ticker and _valid_positive(panel_snap.rows_by_ticker[candidates["C"]].last_price))
            and "iol_bonos" not in snapshots
        )
        if needs_mep_bonos:
            snapshots["iol_bonos"] = fetch_iol_panel("iol_bonos", iol_config=cfg, session=session)

        results[inst.instrument_id] = resolve_iol_bond_quote_usd(
            instrument_id=inst.instrument_id,
            ticker=inst.ticker,
            panel_snapshot=panel_snap,
            bonos_snapshot=snapshots.get("iol_bonos"),
            explicit_species_map=inst.iol_species_map,
            quote_unit=quote_unit,
            iol_config=cfg,
        )

    return results
