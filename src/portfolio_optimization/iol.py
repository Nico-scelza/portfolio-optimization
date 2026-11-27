"""Public IOL (InvertirOnline) fixed-income quote and historical price acquisition.

Supports:
1. Public Argentine sovereign bond ("Bonos") and corporate bond ("Obligaciones
   Negociables" / ONs) market panel point-in-time snapshot extraction and
   AL30/AL30D implicit MEP conversion.
2. Public historical OHLCV retrieval from IOL's ``/Titulo/DatosHistoricos``
   endpoint (``GET`` metadata/``IdTitulo`` discovery followed by segmented
   ``POST /Titulo/DatosHistoricos`` range queries), strictly normalized as
   unadjusted market prices (price return only, never fabricated as total return).

Intentionally excluded:
- Google Sheets / SpreadsheetApp / LockService / cell formatting / triggers
- UVA, dólar blue, GD30/GD30C CCL, private IDs, browser automation, anti-bot bypass
- Historical price synthesis from snapshots, coupon schedules, or spot MEP rates
"""

from __future__ import annotations

import calendar
import html as html_lib
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlsplit
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .errors import ConfigurationError, FXConversionError, IOLHistoryError, IOLQuoteError
from .schemas import (
    VALID_IOL_HISTORICAL_FREQUENCIES,
    VALID_IOL_HISTORICAL_PRICE_FIELDS,
    InstrumentConfig,
    IOLConfig,
    IOLQuoteResult,
    _validate_https_url,
)

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
            if not re.fullmatch(r"[1-9]\d{0,2}(?:\.\d{3})+", int_part):
                return None
            normalized = f"{sign}{int_part.replace('.', '')}.{dec_part}"
        else:
            # Anglo format: 1,234,567.89
            int_part, dec_part = body.split(".", 1)
            if "." in dec_part or "," in dec_part:
                return None
            if not re.fullmatch(r"[1-9]\d{0,2}(?:,\d{3})+", int_part):
                return None
            normalized = f"{sign}{int_part.replace(',', '')}.{dec_part}"
    elif has_comma:
        if body.count(",") == 1:
            int_part, frac_part = body.split(",")
            if (
                locale_mode == "en-US"
                and len(frac_part) == 3
                and re.fullmatch(r"[1-9]\d{0,2}", int_part)
            ):
                # Anglo thousands grouping (e.g., 75,000 -> 75000),
                # while preserving 0,125 -> 0.125 and 100,70 -> 100.70
                normalized = f"{sign}{int_part}{frac_part}"
            else:
                # Argentine decimal comma: 100,70 or 5,25
                normalized = f"{sign}{int_part}.{frac_part}"
        else:
            if re.fullmatch(r"[1-9]\d{0,2}(?:,\d{3})+", body):
                normalized = f"{sign}{body.replace(',', '')}"
            else:
                return None
    elif has_dot:
        dot_count = body.count(".")
        if dot_count > 1:
            if re.fullmatch(r"[1-9]\d{0,2}(?:\.\d{3})+", body):
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


_IOL_REDIRECT_HTTP_CODES = frozenset({301, 302, 303, 307, 308})
_IOL_MAX_REDIRECT_HOPS = 5


def _assert_https_url_or_raise(
    raw_url: str,
    *,
    error_cls: type[IOLQuoteError],
    asset_id: Optional[str],
    context_label: str,
    code: Optional[str] = None,
) -> str:
    """Validate that a target or redirect URL uses HTTPS without embedded credentials."""
    try:
        return _validate_https_url(raw_url, context_label)
    except ConfigurationError as exc:
        kwargs: Dict[str, Any] = {
            "asset_id": asset_id,
            "details": {"url": raw_url, "context": context_label},
        }
        if code is not None:
            kwargs["code"] = code
        raise error_cls(
            f"Refusing insecure or invalid non-HTTPS URL '{raw_url}' during {context_label}.",
            **kwargs,
        ) from exc


def _send_https_only_request(
    http: requests.Session,
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    timeout: float,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    error_cls: type[IOLQuoteError] = IOLHistoryError,
    asset_id: Optional[str] = None,
    context_label: str = "IOL request",
    insecure_code: Optional[str] = "iol_insecure_redirect",
) -> requests.Response:
    """Send an HTTP request with ``allow_redirects=False`` and pre-hop HTTPS validation.

    Validates every redirect ``Location`` BEFORE issuing the next hop so no plaintext HTTP
    hop (including ``HTTPS -> HTTP -> HTTPS`` chains) can transmit cookies or requests.
    Also inspects ``response.history`` and final ``response.url`` for mocked responses.
    """
    current_url = _assert_https_url_or_raise(
        url, error_cls=error_cls, asset_id=asset_id, context_label=context_label, code=insecure_code
    )
    current_method = method.upper()
    current_params = params
    current_data = data

    for hop in range(_IOL_MAX_REDIRECT_HOPS + 1):
        if current_method == "GET":
            response = http.get(
                current_url,
                params=current_params,
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
                verify=True,
            )
        elif current_method == "POST":
            response = http.post(
                current_url,
                data=current_data,
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
                verify=True,
            )
        else:
            raise ConfigurationError(f"Unsupported HTTP method '{current_method}'.")

        # Validate any simulated redirect history or final URL on mocked responses
        hist_entries = getattr(response, "history", None)
        if isinstance(hist_entries, (list, tuple)):
            for h_resp in hist_entries:
                h_url = getattr(h_resp, "url", None)
                if isinstance(h_url, str) and h_url.strip():
                    _assert_https_url_or_raise(
                        h_url.strip(),
                        error_cls=error_cls,
                        asset_id=asset_id,
                        context_label=f"{context_label} redirect history",
                        code=insecure_code,
                    )

        final_url = getattr(response, "url", None)
        if isinstance(final_url, str) and final_url.strip():
            _assert_https_url_or_raise(
                final_url.strip(),
                error_cls=error_cls,
                asset_id=asset_id,
                context_label=f"{context_label} response URL",
                code=insecure_code,
            )

        status = int(response.status_code)
        if status in _IOL_REDIRECT_HTTP_CODES:
            resp_headers = getattr(response, "headers", None)
            location = (resp_headers.get("Location") or resp_headers.get("location")) if resp_headers else None
            if not location or not isinstance(location, str) or not location.strip():
                return response
            next_url = urljoin(current_url, location.strip())
            close_fn = getattr(response, "close", None)
            if callable(close_fn):
                close_fn()
            _assert_https_url_or_raise(
                next_url,
                error_cls=error_cls,
                asset_id=asset_id,
                context_label=f"{context_label} redirect hop {hop + 1}",
                code=insecure_code,
            )
            if hop >= _IOL_MAX_REDIRECT_HOPS:
                kwargs: Dict[str, Any] = {
                    "asset_id": asset_id,
                    "details": {"url": url, "last_redirect": next_url, "max_redirects": _IOL_MAX_REDIRECT_HOPS},
                }
                if error_cls is IOLHistoryError:
                    kwargs["code"] = "iol_http_error"
                raise error_cls(
                    f"Exceeded maximum redirects ({_IOL_MAX_REDIRECT_HOPS}) during {context_label}.",
                    **kwargs,
                )
            current_url = next_url
            current_params = None
            if status in (301, 302, 303) and current_method == "POST":
                current_method = "GET"
                current_data = None
            continue

        return response

    return response  # pragma: no cover


def fetch_iol_panel(
    panel_source: str,
    *,
    iol_config: Optional[IOLConfig] = None,
    session: Optional[requests.Session] = None,
) -> IOLPanelSnapshot:
    """Fetch and parse a public IOL fixed-income panel (``iol_bonos`` or ``iol_ons``).

    Enforces strict TLS verification (``verify=True``), pre-hop HTTPS redirect validation,
    explicit timeout, HTTP 2xx validation, and explicit errors with no access-control evasion.
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
            response = _send_https_only_request(
                http,
                "GET",
                url,
                headers=IOL_DEFAULT_HEADERS,
                timeout=cfg.timeout_seconds,
                error_cls=IOLQuoteError,
                asset_id=None,
                context_label=f"IOL panel '{panel_source}'",
                insecure_code=None,
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
        query_ticker = inst.iol_symbol or inst.ticker
        effective_species_map = inst.iol_species_map
        if inst.iol_symbol is not None:
            if inst.iol_species_map:
                matched_species = [
                    k for k, v in inst.iol_species_map.items() if v == inst.iol_symbol
                ]
                if not matched_species:
                    raise IOLQuoteError(
                        f"Instrument '{inst.instrument_id}' sets iol_symbol='{inst.iol_symbol}' which does not appear in "
                        f"explicit iol_species_map={inst.iol_species_map}. Align iol_symbol with iol_species_map.",
                        asset_id=inst.instrument_id,
                    )
                if len(matched_species) > 1:
                    raise IOLQuoteError(
                        f"Instrument '{inst.instrument_id}' maps iol_symbol='{inst.iol_symbol}' to multiple species {matched_species}.",
                        asset_id=inst.instrument_id,
                    )
                effective_species_map = {matched_species[0]: inst.iol_symbol}
            elif (
                len(inst.iol_symbol) >= 4
                and inst.iol_symbol[-1] in ("D", "C")
                and (
                    re.fullmatch(r"[A-Z]{2,4}\d{1,2}[DC]", inst.iol_symbol)
                    or inst.iol_symbol[:-1] in panel_snap.rows_by_ticker
                    or f"{inst.iol_symbol[:-1]}O" in panel_snap.rows_by_ticker
                    or (panel_snap.panel_source == "iol_ons" and len(inst.iol_symbol) == 5)
                )
            ):
                effective_species_map = {inst.iol_symbol[-1]: inst.iol_symbol}

        candidates = _resolve_species_candidates(query_ticker, panel_snap, effective_species_map)
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
            ticker=query_ticker,
            panel_snapshot=panel_snap,
            bonos_snapshot=snapshots.get("iol_bonos"),
            explicit_species_map=effective_species_map,
            quote_unit=quote_unit,
            iol_config=cfg,
        )

    return results


# =============================================================================
# Public IOL historical price acquisition (/Titulo/DatosHistoricos)
# =============================================================================

_IOL_HISTORICAL_FREQ_IDS: Dict[str, str] = {
    "daily": "1",
    "weekly": "2",
    "monthly": "3",
}

_IOL_HIST_HEADER_EXACT: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("trade_date", ("fecha cotizacion", "fecha")),
    ("open", ("apertura",)),
    ("high", ("maximo",)),
    ("low", ("minimo",)),
    ("close", ("cierre",)),
    ("adjusted_close", ("cierre ajustado", "cierre aj", "cierre aj.")),
    ("volume_amount", ("volumen monto", "monto operado", "monto")),
    ("volume_nominal", ("volumen nominal", "nominal")),
)

_IOL_HIST_HEADER_ALIAS: Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    ("trade_date", ("fecha",), ()),
    ("open", ("apertura",), ()),
    ("high", ("maximo",), ()),
    ("low", ("minimo",), ()),
    ("adjusted_close", ("cierre ajustado", "ajustado"), ()),
    ("close", ("cierre",), ("ajust",)),
    ("volume_amount", ("monto",), ()),
    ("volume_nominal", ("nominal", "volumen"), ("monto",)),
)

_IOL_TRANSIENT_HTTP_CODES = frozenset({429, 500, 502, 503, 504})
_IOL_ACCESS_DENIED_HTTP_CODES = frozenset({401, 403})
_IOL_PERIOD_RE = re.compile(r"^\s*(\d+)\s*(d|wk|w|mo|m|y)\s*$", re.IGNORECASE)
_BCBA_CLOSE_HOUR_ART = 17
_IOL_MAX_RETRY_DELAY_SECONDS = 10.0
_IOL_SUSPECT_ROW_CAP = 500
_IOL_MAX_TRUNCATION_SPLITS = 6


@dataclass(frozen=True)
class IOLHistoricalBar:
    """Single normalized historical observation parsed from IOL DatosHistoricos."""

    trade_date: date
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    adjusted_close: Optional[float] = None
    volume_amount: Optional[float] = None
    volume_nominal: Optional[float] = None


@dataclass
class IOLHistoricalSeries:
    """Normalized historical daily/periodic market-price series from IOL.

    Financial contract:
    - Empirical verification against IOL ``DatosHistoricos`` on Argentine sovereign
      bonds (e.g. ``AL30D``, ``BPA8D``) and corporate ONs (e.g. ``AEC3D``, ``YMCHD``)
      shows ``Cierre ajustado`` equals unadjusted market ``Cierre`` across coupon
      and principal amortization ex-dates.
    - Accordingly, ``series_type`` is ``"market_price"`` and ``is_total_return`` is
      ``False``. Do not treat returned series as total return without combining them
      with verified historical paid cashflows.
    """

    symbol: str
    market: str
    id_titulo: str
    requested_start_date: date
    requested_end_date: date
    effective_start_date: Optional[date]
    effective_end_date: Optional[date]
    frequency: str
    price_field: str
    series_type: str
    currency: Optional[str]
    usd_settlement_type: Optional[str]
    quote_unit: str
    observed_at: str
    source_url: str
    bars: pd.DataFrame
    price_series: pd.Series
    raw_row_count: int = 0
    valid_row_count: int = 0
    zero_or_no_trade_rows_dropped: int = 0
    zero_volume_no_trade_rows_dropped: int = 0
    duplicate_dates_resolved: int = 0
    out_of_range_rows_dropped: int = 0
    intraday_rows_dropped: int = 0
    segments_fetched: int = 1
    truncation_splits_performed: int = 0
    adjusted_close_differs_count: int = 0
    partial_coverage: bool = False
    is_total_return: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable provenance and coverage metadata (without full bar rows)."""
        return {
            "provider": "iol_datos_historicos",
            "symbol": self.symbol,
            "market": self.market,
            "id_titulo": self.id_titulo,
            "requested_start_date": self.requested_start_date.isoformat(),
            "requested_end_date": self.requested_end_date.isoformat(),
            "effective_start_date": None if self.effective_start_date is None else self.effective_start_date.isoformat(),
            "effective_end_date": None if self.effective_end_date is None else self.effective_end_date.isoformat(),
            "frequency": self.frequency,
            "price_field": self.price_field,
            "series_type": self.series_type,
            "is_total_return": self.is_total_return,
            "currency": self.currency,
            "usd_settlement_type": self.usd_settlement_type,
            "quote_unit": self.quote_unit,
            "observed_at": self.observed_at,
            "source_url": self.source_url,
            "raw_row_count": self.raw_row_count,
            "valid_row_count": self.valid_row_count,
            "zero_or_no_trade_rows_dropped": self.zero_or_no_trade_rows_dropped,
            "zero_volume_no_trade_rows_dropped": self.zero_volume_no_trade_rows_dropped,
            "duplicate_dates_resolved": self.duplicate_dates_resolved,
            "out_of_range_rows_dropped": self.out_of_range_rows_dropped,
            "intraday_rows_dropped": self.intraday_rows_dropped,
            "segments_fetched": self.segments_fetched,
            "truncation_splits_performed": self.truncation_splits_performed,
            "adjusted_close_differs_count": self.adjusted_close_differs_count,
            "partial_coverage": self.partial_coverage,
            "warnings": list(self.warnings),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable metadata and price observations."""
        out = self.to_metadata_dict()
        out["prices"] = {
            pd.Timestamp(idx).date().isoformat(): float(val)
            for idx, val in self.price_series.items()
        }
        return out


IOLHistoryCache = Dict[Tuple[Any, ...], IOLHistoricalSeries]


def _subtract_calendar_years(ref_date: date, years: int) -> date:
    target_year = ref_date.year - int(years)
    max_day = calendar.monthrange(target_year, ref_date.month)[1]
    return date(target_year, ref_date.month, min(ref_date.day, max_day))


def _subtract_calendar_months(ref_date: date, months: int) -> date:
    total_months = (ref_date.year * 12 + (ref_date.month - 1)) - int(months)
    target_year, zero_month = divmod(total_months, 12)
    target_month = zero_month + 1
    max_day = calendar.monthrange(target_year, target_month)[1]
    return date(target_year, target_month, min(ref_date.day, max_day))


def resolve_period_date_range(
    period: str,
    *,
    as_of_date: Optional[date] = None,
    now_fn: Optional[Callable[[Any], datetime]] = None,
) -> Tuple[date, date]:
    """Convert a portfolio lookback ``period`` string (e.g. ``'10y'``, ``'6mo'``) to ``(start_date, end_date)``."""
    raw = str(period or "").strip().lower()
    if not raw:
        raise ConfigurationError("Historical lookback 'period' must be a non-empty string.")

    if as_of_date is not None:
        end_dt = as_of_date
    else:
        clock = now_fn or datetime.now
        end_dt = clock(_AR_TZ).date()

    if raw == "ytd":
        return date(end_dt.year, 1, 1), end_dt
    if raw == "max":
        return date(2000, 1, 1), end_dt

    match = _IOL_PERIOD_RE.fullmatch(raw)
    if not match:
        raise ConfigurationError(
            f"Unsupported historical lookback period '{period}'. "
            "Expected formats like '30d', '12wk', '6mo', '1y', '5y', '10y', 'ytd', or 'max'."
        )
    qty = int(match.group(1))
    unit = match.group(2).lower()
    if qty <= 0:
        raise ConfigurationError(f"Historical lookback period '{period}' must be positive.")

    if unit == "d":
        start_dt = end_dt - timedelta(days=qty)
    elif unit in ("wk", "w"):
        start_dt = end_dt - timedelta(weeks=qty)
    elif unit in ("mo", "m"):
        start_dt = _subtract_calendar_months(end_dt, qty)
    elif unit == "y":
        start_dt = _subtract_calendar_years(end_dt, qty)
    else:  # pragma: no cover - regex-constrained
        raise ConfigurationError(f"Unsupported period unit '{unit}' in '{period}'.")

    return start_dt, end_dt


def build_iol_historical_segments(
    start_date: date,
    end_date: date,
    *,
    chunk_days: int = 1825,
) -> List[Tuple[date, date]]:
    """Split an inclusive ``[start_date, end_date]`` range into deterministic contiguous segments."""
    if start_date > end_date:
        raise ConfigurationError(
            f"Historical start_date ({start_date.isoformat()}) cannot be after end_date ({end_date.isoformat()})."
        )
    if chunk_days < 1:
        raise ConfigurationError("'chunk_days' must be >= 1.")

    segments: List[Tuple[date, date]] = []
    cursor = start_date
    span = timedelta(days=int(chunk_days) - 1)
    while cursor <= end_date:
        seg_end = min(cursor + span, end_date)
        segments.append((cursor, seg_end))
        cursor = seg_end + timedelta(days=1)
    return segments


def _map_historical_header_columns(header_cells: List[str]) -> Dict[str, int]:
    """Map IOL historical OHLCV table columns with strict exact-match priority."""
    norm_headers = [normalize_text(h) for h in header_cells]
    mapping: Dict[str, int] = {}
    used_cols: set[int] = set()

    for key, exact_names in _IOL_HIST_HEADER_EXACT:
        matches = [idx for idx, h in enumerate(norm_headers) if h in exact_names and idx not in used_cols]
        if len(matches) == 1:
            mapping[key] = matches[0]
            used_cols.add(matches[0])
        elif len(matches) > 1:
            if key in ("trade_date", "close", "adjusted_close"):
                raise IOLHistoryError(
                    f"Ambiguous IOL historical table header collision for '{key}': {[header_cells[i] for i in matches]}.",
                    code="iol_schema_changed",
                    details={"ambiguous_key": key, "matching_headers": [header_cells[i] for i in matches]},
                )
            mapping[key] = matches[0]
            used_cols.add(matches[0])

    for key, aliases, forbidden in _IOL_HIST_HEADER_ALIAS:
        if key in mapping:
            continue
        matches = [
            idx
            for idx, h in enumerate(norm_headers)
            if idx not in used_cols
            and h
            and any(a in h for a in aliases)
            and not any(bad in h for bad in forbidden)
        ]
        if len(matches) == 1:
            mapping[key] = matches[0]
            used_cols.add(matches[0])
        elif len(matches) > 1:
            if key in ("trade_date", "close", "adjusted_close"):
                raise IOLHistoryError(
                    f"Ambiguous IOL historical table header collision for '{key}': {[header_cells[i] for i in matches]}.",
                    code="iol_schema_changed",
                    details={"ambiguous_key": key, "matching_headers": [header_cells[i] for i in matches]},
                )
            mapping[key] = matches[0]
            used_cols.add(matches[0])

    return mapping


def _detect_historical_number_locale(numeric_cells: Sequence[str], *, symbol: str = "") -> str:
    """Detect whether an IOL DatosHistoricos table formats numbers in ``en-US`` or ``es-AR``.

    Live IOL ``DatosHistoricos`` renders ``en-US`` numbers (e.g. ``99.70``, ``40,558.76``)
    even when ``Accept-Language: es-AR`` is sent. Auto-detects ``es-AR`` cleanly if a
    response uses Argentine separators (``82.150,00`` / ``99,70``) and fails closed on
    contradictory mixed formats within the same table.
    """
    en_us_hits = 0
    es_ar_hits = 0

    for raw in numeric_cells:
        txt = clean_cell_text(raw)
        if not txt or txt.lower() in _MISSING_TOKENS:
            continue
        s = re.sub(r"(?i)(?:u\$s|us\$|usd|ars|\$|%)", "", txt)
        s = re.sub(r"\s+", "", s)
        if not s or not re.fullmatch(r"[+-]?[\d.,]+", s):
            continue
        body = s[1:] if s[0] in "+-" else s
        if "," in body and "." in body:
            if body.rfind(".") > body.rfind(","):
                en_us_hits += 1
            else:
                es_ar_hits += 1
        elif "." in body:
            if body.count(".") > 1:
                if re.fullmatch(r"[1-9]\d{0,2}(?:\.\d{3})+", body):
                    es_ar_hits += 1
            else:
                int_part, frac = body.split(".")
                if len(frac) != 3 or int_part == "0" or len(int_part) > 3:
                    en_us_hits += 1
        elif "," in body:
            if body.count(",") > 1:
                if re.fullmatch(r"[1-9]\d{0,2}(?:,\d{3})+", body):
                    en_us_hits += 1
            else:
                int_part, frac = body.split(",")
                if len(frac) != 3 or int_part == "0" or len(int_part) > 3:
                    es_ar_hits += 1

    if en_us_hits > 0 and es_ar_hits > 0:
        raise IOLHistoryError(
            f"Contradictory mixed numeric locales detected in IOL historical table for '{symbol or 'unknown'}' "
            f"(en-US signals={en_us_hits}, es-AR signals={es_ar_hits}).",
            asset_id=symbol or None,
            code="iol_schema_changed",
            details={"en_us_hits": en_us_hits, "es_ar_hits": es_ar_hits},
        )
    if es_ar_hits > 0:
        return "es-AR"
    return "en-US"


def _parse_historical_dates(
    raw_dates: Sequence[str],
    *,
    symbol: str = "",
    default_order: str = "mdy",
    expected_range: Optional[Tuple[date, date]] = None,
) -> List[date]:
    """Parse all date cells of an IOL historical table with consistent table-wide format validation.

    Live IOL ``DatosHistoricos`` outputs ``MM/DD/YYYY`` (e.g. ``11/26/2026``). Supports
    ``DD/MM/YYYY`` and ISO ``YYYY-MM-DD`` deterministically, uses ``expected_range``
    (when provided) and table locale to disambiguate short windows where all days <= 12,
    and rejects contradictory date orders inside the same table.
    """
    if not raw_dates:
        return []

    slash_triples: List[Optional[Tuple[int, int, int]]] = []
    iso_parsed: List[Optional[date]] = []

    for raw in raw_dates:
        txt = clean_cell_text(raw)
        if not txt:
            raise IOLHistoryError(
                f"Empty trade date encountered in IOL historical table for '{symbol or 'unknown'}'.",
                asset_id=symbol or None,
                code="iol_schema_changed",
            )
        m_iso = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:\s+.*)?", txt)
        if m_iso:
            y, mo, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
            try:
                iso_parsed.append(date(y, mo, d))
            except ValueError as exc:
                raise IOLHistoryError(
                    f"Invalid ISO date '{txt}' in IOL historical table for '{symbol or 'unknown'}'.",
                    asset_id=symbol or None,
                    code="iol_schema_changed",
                ) from exc
            slash_triples.append(None)
            continue

        m_slash = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?:\s+.*)?", txt)
        if m_slash:
            a, b, y = int(m_slash.group(1)), int(m_slash.group(2)), int(m_slash.group(3))
            slash_triples.append((a, b, y))
            iso_parsed.append(None)
            continue

        raise IOLHistoryError(
            f"Unrecognized date format '{txt}' in IOL historical table for '{symbol or 'unknown'}'.",
            asset_id=symbol or None,
            code="iol_schema_changed",
            details={"raw_date": txt},
        )

    valid_triples = [t for t in slash_triples if t is not None]
    has_iso = any(dt is not None for dt in iso_parsed)
    if valid_triples and has_iso:
        raise IOLHistoryError(
            f"Contradictory mixed ISO and slash date formats in IOL historical table for '{symbol or 'unknown'}'.",
            asset_id=symbol or None,
            code="iol_schema_changed",
        )

    order = default_order
    if valid_triples:
        first_gt12 = any(a > 12 for a, _, _ in valid_triples)
        second_gt12 = any(b > 12 for _, b, _ in valid_triples)
        if first_gt12 and second_gt12:
            raise IOLHistoryError(
                f"Contradictory date formats (mixed MM/DD/YYYY and DD/MM/YYYY) in IOL historical table for '{symbol or 'unknown'}'.",
                asset_id=symbol or None,
                code="iol_schema_changed",
            )
        if first_gt12:
            order = "dmy"
        elif second_gt12:
            order = "mdy"
        else:
            def _eval_order(cand_order: str) -> Optional[Tuple[int, bool]]:
                cand_dates: List[date] = []
                in_range = 0
                for a, b, y in valid_triples:
                    try:
                        cand_dt = date(y, b, a) if cand_order == "dmy" else date(y, a, b)
                    except ValueError:
                        return None
                    cand_dates.append(cand_dt)
                    if expected_range is not None and expected_range[0] <= cand_dt <= expected_range[1]:
                        in_range += 1
                is_mono = (
                    all(x >= y for x, y in zip(cand_dates, cand_dates[1:]))
                    or all(x <= y for x, y in zip(cand_dates, cand_dates[1:]))
                )
                return in_range, is_mono

            mdy_eval = _eval_order("mdy")
            dmy_eval = _eval_order("dmy")
            if mdy_eval is not None and dmy_eval is None:
                order = "mdy"
            elif dmy_eval is not None and mdy_eval is None:
                order = "dmy"
            elif mdy_eval is not None and dmy_eval is not None:
                if expected_range is not None and mdy_eval[0] != dmy_eval[0]:
                    order = "mdy" if mdy_eval[0] > dmy_eval[0] else "dmy"
                elif len(valid_triples) >= 2 and mdy_eval[1] != dmy_eval[1]:
                    order = "mdy" if mdy_eval[1] else "dmy"

    parsed: List[date] = []
    for raw, iso_dt, triple in zip(raw_dates, iso_parsed, slash_triples):
        if iso_dt is not None:
            parsed.append(iso_dt)
            continue
        assert triple is not None
        a, b, y = triple
        try:
            dt = date(y, b, a) if order == "dmy" else date(y, a, b)
        except ValueError as exc:
            raise IOLHistoryError(
                f"Invalid calendar date '{raw}' under inferred format '{order}' for '{symbol or 'unknown'}'.",
                asset_id=symbol or None,
                code="iol_schema_changed",
                details={"raw_date": raw, "inferred_order": order},
            ) from exc
        parsed.append(dt)

    return parsed


def parse_iol_historical_page_metadata(
    html_text: str,
    *,
    requested_symbol: str,
    market: str = "BCBA",
    url: str = "",
) -> Dict[str, Any]:
    """Extract ``IdTitulo``, ``Simbolo``, and page structure signals from IOL DatosHistoricos GET HTML."""
    sym = requested_symbol.strip().upper()
    if not html_text or not str(html_text).strip():
        raise IOLHistoryError(
            f"Empty HTML response received from IOL DatosHistoricos for '{sym}'.",
            asset_id=sym,
            code="iol_schema_changed",
            details={"symbol": sym, "market": market, "url": url},
        )

    soup = BeautifulSoup(html_text, "html.parser")
    page_text_lower = normalize_text(soup.get_text(" ", strip=True)[:4000])

    id_input = soup.find("input", attrs={"name": "IdTitulo"})
    sim_input = soup.find("input", attrs={"name": "Simbolo"})
    table = soup.find("table", id="tbcotizaciones")

    id_titulo = clean_cell_text(id_input.get("value", "")) if id_input is not None else ""
    page_symbol = clean_cell_text(sim_input.get("value", "")).upper() if sim_input is not None else ""

    if not id_titulo:
        if any(marker in page_text_lower for marker in ("access denied", "cloudflare", "attention required", "iniciar sesion", "captcha")):
            if table is None and soup.find("h2", class_="colorIOL") is None:
                raise IOLHistoryError(
                    f"Access denied or challenge page returned by IOL DatosHistoricos for '{sym}'.",
                    asset_id=sym,
                    code="iol_access_denied",
                    details={"symbol": sym, "market": market, "url": url},
                )
        h2 = soup.find("h2", class_="colorIOL")
        h2_text = clean_cell_text(h2.get_text(" ", strip=True)) if h2 is not None else ""
        if h2 is not None or soup.find("input", attrs={"name": "DesdeHasta"}) is not None:
            raise IOLHistoryError(
                f"Symbol '{sym}' was not found in IOL DatosHistoricos for market '{market}'.",
                asset_id=sym,
                code="iol_symbol_not_found",
                details={"symbol": sym, "market": market, "url": url, "heading": h2_text},
            )
        raise IOLHistoryError(
            f"Unexpected HTML layout on IOL DatosHistoricos for '{sym}': missing 'IdTitulo' form field.",
            asset_id=sym,
            code="iol_schema_changed",
            details={"symbol": sym, "market": market, "url": url},
        )

    if not re.fullmatch(r"\d+", id_titulo):
        raise IOLHistoryError(
            f"Invalid non-numeric IdTitulo '{id_titulo}' returned by IOL for '{sym}'.",
            asset_id=sym,
            code="iol_schema_changed",
            details={"symbol": sym, "market": market, "id_titulo": id_titulo},
        )

    if page_symbol and page_symbol != sym:
        raise IOLHistoryError(
            f"IOL DatosHistoricos resolved symbol '{page_symbol}' instead of requested '{sym}'; refusing mismatched series.",
            asset_id=sym,
            code="iol_symbol_mismatch",
            details={"requested_symbol": sym, "resolved_symbol": page_symbol, "id_titulo": id_titulo},
        )

    if table is None:
        raise IOLHistoryError(
            f"IOL DatosHistoricos page for '{sym}' (IdTitulo={id_titulo}) is missing expected table '#tbcotizaciones'.",
            asset_id=sym,
            code="iol_schema_changed",
            details={"symbol": sym, "market": market, "id_titulo": id_titulo, "url": url},
        )

    return {
        "symbol": page_symbol or sym,
        "market": market,
        "id_titulo": id_titulo,
        "has_table": True,
    }


def parse_iol_historical_table(
    html_text: str,
    *,
    symbol: str = "",
    allow_empty_rows: bool = True,
    require_adjusted_close: bool = False,
    expected_range: Optional[Tuple[date, date]] = None,
) -> Tuple[List[IOLHistoricalBar], Tuple[str, ...]]:
    """Parse an IOL DatosHistoricos HTML table (GET page or POST AJAX fragment) into normalized bars.

    Validates table presence and required headers (``Fecha Cotización`` and ``Cierre``),
    detects table-wide date and number formatting, and preserves both ``close`` and
    ``adjusted_close`` for downstream inspection. Never fabricates rows.
    """
    sym = symbol.strip().upper()
    if not html_text or not str(html_text).strip():
        raise IOLHistoryError(
            f"Empty HTML payload received from IOL DatosHistoricos for '{sym or 'unknown'}'.",
            asset_id=sym or None,
            code="iol_schema_changed",
        )

    soup = BeautifulSoup(html_text, "html.parser")
    tables = []
    primary = soup.find("table", id="tbcotizaciones")
    if primary is not None:
        tables.append(primary)
    for tbl in soup.find_all("table"):
        if tbl is not primary:
            tables.append(tbl)

    if not tables:
        page_lower = normalize_text(soup.get_text(" ", strip=True)[:2000])
        if any(m in page_lower for m in ("access denied", "cloudflare", "attention required", "captcha")):
            raise IOLHistoryError(
                f"Access denied or challenge page returned while parsing IOL historical data for '{sym or 'unknown'}'.",
                asset_id=sym or None,
                code="iol_access_denied",
            )
        raise IOLHistoryError(
            f"No HTML <table> found in IOL DatosHistoricos response for '{sym or 'unknown'}'.",
            asset_id=sym or None,
            code="iol_schema_changed",
        )

    chosen_header: Optional[List[str]] = None
    chosen_col_map: Optional[Dict[str, int]] = None
    chosen_data_rows: List[List[str]] = []

    for tbl in tables:
        tr_elems = tbl.find_all("tr")
        if not tr_elems:
            continue
        rows_text: List[List[str]] = []
        for tr in tr_elems:
            cells = tr.find_all(["th", "td"])
            if not cells:
                continue
            vals = [clean_cell_text(c.get_text(separator=" ", strip=True)) for c in cells]
            if any(vals):
                rows_text.append(vals)
        if not rows_text:
            continue
        hdr = rows_text[0]
        col_map = _map_historical_header_columns(hdr)
        if "trade_date" in col_map and "close" in col_map:
            chosen_header = hdr
            chosen_col_map = col_map
            chosen_data_rows = rows_text[1:]
            break

    if chosen_header is None or chosen_col_map is None:
        raise IOLHistoryError(
            f"Expected IOL historical table headers (Fecha Cotización, Cierre) not found for '{sym or 'unknown'}'.",
            asset_id=sym or None,
            code="iol_schema_changed",
            details={"tables_inspected": len(tables)},
        )

    if require_adjusted_close and "adjusted_close" not in chosen_col_map:
        raise IOLHistoryError(
            f"Requested price_field='adjusted_close' but column 'Cierre ajustado' is absent from IOL historical table for '{sym or 'unknown'}'.",
            asset_id=sym or None,
            code="iol_schema_changed",
            details={"headers": chosen_header},
        )

    if not chosen_data_rows:
        if not allow_empty_rows:
            raise IOLHistoryError(
                f"IOL DatosHistoricos returned zero rows for '{sym or 'unknown'}'.",
                asset_id=sym or None,
                code="iol_empty_history",
            )
        return [], tuple(chosen_header)

    def _cell(row: List[str], key: str) -> Optional[str]:
        idx = chosen_col_map.get(key)
        if idx is None or idx >= len(row):
            return None
        return row[idx]

    # Collect valid data rows (skipping DataTables empty placeholder messages like "Ningún dato disponible")
    filtered_rows: List[List[str]] = []
    for row in chosen_data_rows:
        raw_dt = _cell(row, "trade_date")
        if not raw_dt:
            continue
        norm_dt = normalize_text(raw_dt)
        if "ningun dato" in norm_dt or "no data" in norm_dt or "sin datos" in norm_dt:
            continue
        filtered_rows.append(row)

    if not filtered_rows:
        if not allow_empty_rows:
            raise IOLHistoryError(
                f"IOL DatosHistoricos returned zero trade rows for '{sym or 'unknown'}'.",
                asset_id=sym or None,
                code="iol_empty_history",
            )
        return [], tuple(chosen_header)

    numeric_samples: List[str] = []
    for r in filtered_rows:
        for k in ("open", "high", "low", "close", "adjusted_close", "volume_amount", "volume_nominal"):
            val = _cell(r, k)
            if val is not None:
                numeric_samples.append(val)
    locale_mode = _detect_historical_number_locale(numeric_samples, symbol=sym)

    raw_date_strs = [_cell(r, "trade_date") or "" for r in filtered_rows]
    default_date_order = "dmy" if locale_mode == "es-AR" else "mdy"
    parsed_dates = _parse_historical_dates(
        raw_date_strs,
        symbol=sym,
        default_order=default_date_order,
        expected_range=expected_range,
    )

    def _parse_strict_cell(
        raw_val: Optional[str],
        *,
        field_name: str,
        trade_dt: date,
    ) -> Optional[float]:
        if raw_val is None:
            return None
        cleaned = clean_cell_text(raw_val)
        if not cleaned or cleaned.lower() in _MISSING_TOKENS:
            return None
        parsed_num = parse_argentine_number(cleaned, locale_mode=locale_mode)
        if parsed_num is None or not math.isfinite(parsed_num):
            raise IOLHistoryError(
                f"Corrupted or non-finite numeric cell '{cleaned}' in column '{field_name}' "
                f"on {trade_dt.isoformat()} for IOL symbol '{sym or 'unknown'}'.",
                asset_id=sym or None,
                code="iol_invalid_price_data",
                details={
                    "symbol": sym or None,
                    "trade_date": trade_dt.isoformat(),
                    "field": field_name,
                    "raw_value": cleaned,
                },
            )
        if parsed_num < 0.0:
            raise IOLHistoryError(
                f"Negative numeric value ({parsed_num}) in column '{field_name}' "
                f"on {trade_dt.isoformat()} for IOL symbol '{sym or 'unknown'}'.",
                asset_id=sym or None,
                code="iol_invalid_price_data",
                details={
                    "symbol": sym or None,
                    "trade_date": trade_dt.isoformat(),
                    "field": field_name,
                    "parsed_value": parsed_num,
                },
            )
        return parsed_num

    bars: List[IOLHistoricalBar] = []
    for dt, row in zip(parsed_dates, filtered_rows):
        bars.append(
            IOLHistoricalBar(
                trade_date=dt,
                open=_parse_strict_cell(_cell(row, "open"), field_name="open", trade_dt=dt),
                high=_parse_strict_cell(_cell(row, "high"), field_name="high", trade_dt=dt),
                low=_parse_strict_cell(_cell(row, "low"), field_name="low", trade_dt=dt),
                close=_parse_strict_cell(_cell(row, "close"), field_name="close", trade_dt=dt),
                adjusted_close=_parse_strict_cell(_cell(row, "adjusted_close"), field_name="adjusted_close", trade_dt=dt),
                volume_amount=_parse_strict_cell(_cell(row, "volume_amount"), field_name="volume_amount", trade_dt=dt),
                volume_nominal=_parse_strict_cell(_cell(row, "volume_nominal"), field_name="volume_nominal", trade_dt=dt),
            )
        )

    return bars, tuple(chosen_header)


def _parse_retry_after_seconds(
    raw_header: Optional[str],
    *,
    now_utc: datetime,
) -> Optional[float]:
    """Parse RFC 9110 ``Retry-After`` header (numeric delay-seconds or HTTP-date)."""
    if raw_header is None:
        return None
    text = str(raw_header).strip()
    if not text:
        return None
    try:
        secs = float(text)
        if math.isfinite(secs) and secs >= 0.0:
            return secs
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt.astimezone(timezone.utc) - now_utc).total_seconds()
        if math.isfinite(delta):
            return max(0.0, delta)
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _request_iol_with_retry(
    http: requests.Session,
    method: str,
    url: str,
    *,
    symbol: str,
    market: str,
    iol_config: IOLConfig,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Optional[Callable[[Any], datetime]] = None,
) -> requests.Response:
    """Execute an IOL HTTP request with strict TLS, explicit timeout, and bounded transient retries."""
    max_attempts = 1 + int(iol_config.historical_max_retries)
    clock = now_fn or datetime.now
    last_exc: Optional[Exception] = None
    last_status: Optional[int] = None

    for attempt in range(max_attempts):
        try:
            response = _send_https_only_request(
                http,
                method,
                url,
                headers=headers,
                timeout=iol_config.timeout_seconds,
                params=params,
                data=data,
                error_cls=IOLHistoryError,
                asset_id=symbol,
                context_label=f"IOL DatosHistoricos '{symbol}' ({market})",
                insecure_code="iol_insecure_redirect",
            )
        except requests.Timeout as exc:
            last_exc = exc
            if attempt + 1 < max_attempts:
                delay = min(iol_config.historical_retry_backoff_seconds * (2 ** attempt), _IOL_MAX_RETRY_DELAY_SECONDS)
                if delay > 0.0:
                    sleep_fn(delay)
                continue
            raise IOLHistoryError(
                f"Timeout querying IOL DatosHistoricos for '{symbol}' ({market}) after {max_attempts} attempt(s).",
                asset_id=symbol,
                code="iol_timeout",
                details={"symbol": symbol, "market": market, "attempts": max_attempts, "timeout_seconds": iol_config.timeout_seconds},
            ) from exc
        except requests.ConnectionError as exc:
            last_exc = exc
            if attempt + 1 < max_attempts:
                delay = min(iol_config.historical_retry_backoff_seconds * (2 ** attempt), _IOL_MAX_RETRY_DELAY_SECONDS)
                if delay > 0.0:
                    sleep_fn(delay)
                continue
            raise IOLHistoryError(
                f"Connection error querying IOL DatosHistoricos for '{symbol}' ({market}) after {max_attempts} attempt(s).",
                asset_id=symbol,
                code="iol_network_error",
                details={"symbol": symbol, "market": market, "attempts": max_attempts, "error_type": type(exc).__name__},
            ) from exc
        except requests.RequestException as exc:
            raise IOLHistoryError(
                f"HTTP client error querying IOL DatosHistoricos for '{symbol}' ({market}): {type(exc).__name__}.",
                asset_id=symbol,
                code="iol_network_error",
                details={"symbol": symbol, "market": market, "error_type": type(exc).__name__},
            ) from exc

        status = int(response.status_code)
        last_status = status

        if 200 <= status < 300:
            return response

        if status in _IOL_ACCESS_DENIED_HTTP_CODES:
            raise IOLHistoryError(
                f"Access denied (HTTP {status}) from IOL DatosHistoricos for '{symbol}' ({market}).",
                asset_id=symbol,
                code="iol_access_denied",
                details={"symbol": symbol, "market": market, "status_code": status},
            )

        if status in _IOL_TRANSIENT_HTTP_CODES and attempt + 1 < max_attempts:
            delay = min(iol_config.historical_retry_backoff_seconds * (2 ** attempt), _IOL_MAX_RETRY_DELAY_SECONDS)
            retry_after_raw = response.headers.get("Retry-After") if hasattr(response, "headers") and response.headers else None
            ra_sec = _parse_retry_after_seconds(retry_after_raw, now_utc=clock(timezone.utc))
            if ra_sec is not None:
                delay = min(max(delay, ra_sec), _IOL_MAX_RETRY_DELAY_SECONDS)
            if delay > 0.0:
                sleep_fn(delay)
            continue

        if status == 429:
            raise IOLHistoryError(
                f"Rate limited (HTTP 429) by IOL DatosHistoricos for '{symbol}' ({market}) after {max_attempts} attempt(s).",
                asset_id=symbol,
                code="iol_rate_limited",
                details={"symbol": symbol, "market": market, "status_code": 429, "attempts": max_attempts},
            )

        raise IOLHistoryError(
            f"IOL DatosHistoricos returned HTTP {status} for '{symbol}' ({market}).",
            asset_id=symbol,
            code="iol_http_error",
            details={"symbol": symbol, "market": market, "status_code": status, "attempts": attempt + 1},
        )

    # Defensive fallback (loop always returns or raises)
    raise IOLHistoryError(  # pragma: no cover
        f"Failed to query IOL DatosHistoricos for '{symbol}' ({market}).",
        asset_id=symbol,
        code="iol_network_error",
        details={"symbol": symbol, "market": market, "status_code": last_status, "error": str(last_exc) if last_exc else None},
    )


def _bar_price_value(bar: IOLHistoricalBar, price_field: str) -> Optional[float]:
    if price_field == "adjusted_close":
        return bar.adjusted_close
    return bar.close


def _is_zero_volume_no_trade_bar(bar: IOLHistoricalBar) -> bool:
    """Return True when both volume columns are explicitly reported as zero (IOL server forward-fill)."""
    return (
        bar.volume_amount is not None
        and bar.volume_nominal is not None
        and math.isfinite(bar.volume_amount)
        and math.isfinite(bar.volume_nominal)
        and bar.volume_amount == 0.0
        and bar.volume_nominal == 0.0
    )


def _deduplicate_historical_bars(
    bars: Sequence[IOLHistoricalBar],
    *,
    symbol: str,
    price_field: str,
    duplicate_conflict_policy: str = "error",
) -> Tuple[List[IOLHistoricalBar], int]:
    """Deterministically sort and deduplicate bars by ``trade_date``, detecting value conflicts."""
    if duplicate_conflict_policy not in ("error", "last"):
        raise ConfigurationError(
            f"Invalid duplicate_conflict_policy '{duplicate_conflict_policy}'. Expected 'error' or 'last'."
        )

    by_date: Dict[date, List[IOLHistoricalBar]] = {}
    for b in bars:
        px = _bar_price_value(b, price_field)
        if px is not None and (not math.isfinite(float(px)) or float(px) < 0.0):
            raise IOLHistoryError(
                f"Invalid non-finite or negative {price_field} ({px}) for '{symbol}' on {b.trade_date.isoformat()}.",
                asset_id=symbol,
                code="iol_invalid_price_data",
                details={"symbol": symbol, "trade_date": b.trade_date.isoformat(), "price_field": price_field, "value": px},
            )
        by_date.setdefault(b.trade_date, []).append(b)

    resolved: List[IOLHistoricalBar] = []
    dups_resolved = 0

    for dt in sorted(by_date.keys()):
        group = by_date[dt]
        if len(group) == 1:
            resolved.append(group[0])
            continue

        dups_resolved += len(group) - 1
        pos_bars = [b for b in group if _valid_positive(_bar_price_value(b, price_field))]
        if not pos_bars:
            resolved.append(group[-1])
            continue

        ref_bar = pos_bars[0]
        ref_px = float(_bar_price_value(ref_bar, price_field))  # type: ignore[arg-type]
        conflicting: List[float] = []
        for cand in pos_bars[1:]:
            cpx = float(_bar_price_value(cand, price_field))  # type: ignore[arg-type]
            if not math.isclose(cpx, ref_px, rel_tol=1e-9, abs_tol=1e-9):
                conflicting.append(cpx)

        if conflicting:
            if duplicate_conflict_policy == "error":
                raise IOLHistoryError(
                    f"Conflicting duplicate {price_field} observations for '{symbol}' on {dt.isoformat()}: "
                    f"{[ref_px] + conflicting}.",
                    asset_id=symbol,
                    code="iol_duplicate_conflict",
                    details={
                        "symbol": symbol,
                        "trade_date": dt.isoformat(),
                        "price_field": price_field,
                        "conflicting_values": [ref_px] + conflicting,
                    },
                )
            resolved.append(pos_bars[-1])
        else:
            # Prefer row with higher reported (volume_amount, volume_nominal) when price matches
            best = max(
                pos_bars,
                key=lambda b: (
                    1 if b.volume_amount is not None else 0,
                    b.volume_amount if b.volume_amount is not None else -1.0,
                    1 if b.volume_nominal is not None else 0,
                    b.volume_nominal if b.volume_nominal is not None else -1.0,
                ),
            )
            resolved.append(best)

    return resolved, dups_resolved


def fetch_iol_historical_prices(
    symbol: str,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    period: Optional[str] = None,
    market: Optional[str] = None,
    frequency: str = "daily",
    price_field: Optional[str] = None,
    as_of_date: Optional[date] = None,
    currency: Optional[str] = None,
    usd_settlement_type: Optional[str] = None,
    quote_unit: str = "per_100_original_nominal",
    duplicate_conflict_policy: str = "error",
    exclude_intraday_today: bool = True,
    drop_zero_volume_rows: Optional[bool] = None,
    iol_config: Optional[IOLConfig] = None,
    session: Optional[requests.Session] = None,
    cache: Optional[IOLHistoryCache] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
    now_fn: Optional[Callable[[Any], datetime]] = None,
    _initial_delay_seconds: float = 0.0,
) -> IOLHistoricalSeries:
    """Fetch normalized historical prices from public IOL ``/Titulo/DatosHistoricos``.

    Mechanism (verified against live IOL):
    1. ``GET /Titulo/DatosHistoricos?simbolo={SIMBOLO}&mercado={MERCADO}`` initializes
       session cookies and resolves ``IdTitulo`` and ``Simbolo`` from the form.
    2. Segmented ``POST /Titulo/DatosHistoricos`` queries with form fields
       ``desdehasta="DD/MM/YYYY - DD/MM/YYYY"``, ``idtitulo``, and ``idfrecuencias``
       retrieve the full requested range (bypassing the 500-row default GET cap).
       Because IOL treats the right boundary of ``desdehasta`` as exclusive, each
       segment requests ``(seg_end + 1 day)`` and strictly filters parsed dates back
       into ``[requested_start_date, effective_end_date]``. Adaptive bounded
       subdivision detects and recovers from unexpected 500-row POST truncations.

    Financial & integrity guarantees:
    - Queries the exact ``symbol`` without mutating suffixes.
    - Selects unadjusted ``Cierre`` by default (``price_field="close"``), preserving
      ``adjusted_close`` for audit; never labels fixed-income series as total return.
    - Drops zero-price and zero-volume server-forward-filled no-trade rows without
      forward-fill or fabricated bars, rejects negative/corrupted price cells, and
      excludes unclosed intraday/future observations.
    - Detects conflicting duplicate dates and differentiates legitimate partial
      coverage (pre-issuance start or matured/stale tail) from provider failures.
    """
    cfg = iol_config or IOLConfig()
    sym = str(symbol or "").strip().upper()
    if not sym:
        raise ConfigurationError("IOL historical query requires a non-empty 'symbol'.")
    mkt = str(market or cfg.default_market).strip().upper()
    if not mkt:
        raise ConfigurationError("IOL historical query requires a non-empty 'market'.")

    freq = str(frequency or "daily").strip().lower()
    if freq not in VALID_IOL_HISTORICAL_FREQUENCIES:
        raise ConfigurationError(
            f"Unsupported IOL historical frequency '{frequency}'. Expected one of {VALID_IOL_HISTORICAL_FREQUENCIES}."
        )
    freq_code = _IOL_HISTORICAL_FREQ_IDS[freq]

    eff_price_field = str(price_field or cfg.historical_price_field).strip().lower()
    if eff_price_field not in VALID_IOL_HISTORICAL_PRICE_FIELDS:
        raise ConfigurationError(
            f"Unsupported IOL historical price_field '{eff_price_field}'. Expected one of {VALID_IOL_HISTORICAL_PRICE_FIELDS}."
        )

    eff_drop_zero_vol = cfg.drop_zero_volume_rows if drop_zero_volume_rows is None else bool(drop_zero_volume_rows)
    clock = now_fn or datetime.now
    sleeper = sleep_fn or time.sleep
    now_art = clock(_AR_TZ)
    today_art = now_art.date()

    if start_date is not None or end_date is not None:
        if period is not None:
            raise ConfigurationError("Provide either explicit 'start_date'/'end_date' or 'period', not both.")
        req_end = end_date if end_date is not None else (as_of_date if as_of_date is not None else today_art)
        if start_date is None:
            raise ConfigurationError("'start_date' is required when 'end_date' is provided without 'period'.")
        req_start = start_date
    else:
        req_start, req_end = resolve_period_date_range(period or "5y", as_of_date=as_of_date, now_fn=clock)

    eff_end = min(req_end, as_of_date) if as_of_date is not None else req_end
    if req_start > eff_end:
        raise ConfigurationError(
            f"Effective historical range is empty: start_date ({req_start.isoformat()}) > effective_end_date ({eff_end.isoformat()})."
        )

    intraday_cache_bucket: Optional[Tuple[date, bool]] = None
    if exclude_intraday_today and (as_of_date is None or as_of_date >= today_art):
        intraday_cache_bucket = (today_art, now_art.hour >= _BCBA_CLOSE_HOUR_ART)

    cache_key = (
        sym,
        mkt,
        cfg.historical_url,
        req_start,
        req_end,
        eff_end,
        as_of_date,
        freq,
        eff_price_field,
        currency,
        usd_settlement_type,
        quote_unit,
        duplicate_conflict_policy,
        bool(exclude_intraday_today),
        eff_drop_zero_vol,
        intraday_cache_bucket,
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    if _initial_delay_seconds > 0.0:
        sleeper(_initial_delay_seconds)

    segments = build_iol_historical_segments(req_start, eff_end, chunk_days=cfg.historical_chunk_days)

    owns_session = session is None
    http = requests.Session() if owns_session else session
    truncation_splits = 0
    segments_fetched = 0
    try:
        get_resp = _request_iol_with_retry(
            http,
            "GET",
            cfg.historical_url,
            symbol=sym,
            market=mkt,
            iol_config=cfg,
            headers=IOL_DEFAULT_HEADERS,
            params={"simbolo": sym, "mercado": mkt},
            sleep_fn=sleeper,
            now_fn=clock,
        )
        get_html = (
            get_resp.content.decode("utf-8", errors="replace")
            if isinstance(get_resp.content, (bytes, bytearray))
            else str(get_resp.text)
        )
        source_url = str(getattr(get_resp, "url", "") or f"{cfg.historical_url}?simbolo={sym}&mercado={mkt}")
        page_meta = parse_iol_historical_page_metadata(
            get_html,
            requested_symbol=sym,
            market=mkt,
            url=source_url,
        )
        id_titulo = str(page_meta["id_titulo"])

        parsed_url = urlsplit(cfg.historical_url)
        origin = f"{parsed_url.scheme}://{parsed_url.netloc}" if parsed_url.scheme and parsed_url.netloc else "https://iol.invertironline.com"
        post_headers = {
            **IOL_DEFAULT_HEADERS,
            "X-Requested-With": "XMLHttpRequest",
            "Referer": source_url,
            "Origin": origin,
        }

        def _has_weekday(d_start: date, d_end: date) -> bool:
            if d_start > d_end:
                return False
            span_days = (d_end - d_start).days
            if span_days >= 2:
                return True
            cur = d_start
            while cur <= d_end:
                if cur.weekday() < 5:
                    return True
                cur += timedelta(days=1)
            return False

        all_bars: List[IOLHistoricalBar] = []
        # Queue of (seg_start, seg_end, is_truncation_probe)
        pending_segments: List[Tuple[date, date, bool]] = [(s, e, False) for s, e in segments]
        probed_windows: set[Tuple[date, date]] = set()
        while pending_segments:
            seg_start, seg_end, is_probe = pending_segments.pop(0)
            if segments_fetched > 0 and cfg.historical_request_delay_seconds > 0.0:
                sleeper(cfg.historical_request_delay_seconds)
            # IOL POST 'desdehasta' upper date is exclusive -> add +1 day and filter strictly after parsing
            post_end_exclusive = seg_end + timedelta(days=1)
            desdehasta = f"{seg_start:%d/%m/%Y} - {post_end_exclusive:%d/%m/%Y}"
            post_resp = _request_iol_with_retry(
                http,
                "POST",
                cfg.historical_url,
                symbol=sym,
                market=mkt,
                iol_config=cfg,
                headers=post_headers,
                data={
                    "desdehasta": desdehasta,
                    "idtitulo": id_titulo,
                    "idfrecuencias": freq_code,
                },
                sleep_fn=sleeper,
                now_fn=clock,
            )
            segments_fetched += 1
            post_html = (
                post_resp.content.decode("utf-8", errors="replace")
                if isinstance(post_resp.content, (bytes, bytearray))
                else str(post_resp.text)
            )
            seg_bars, _ = parse_iol_historical_table(
                post_html,
                symbol=sym,
                allow_empty_rows=True,
                require_adjusted_close=(eff_price_field == "adjusted_close"),
                expected_range=(seg_start, post_end_exclusive),
            )
            in_seg_dates = [b.trade_date for b in seg_bars if seg_start <= b.trade_date <= seg_end]
            if is_probe and in_seg_dates:
                truncation_splits += 1
            all_bars.extend(seg_bars)

            # Truncation detection: if POST returns exactly the known 500-row cap,
            # check for ignored date filters (0 in-range rows) and probe any uncovered
            # leading or trailing weekday sub-ranges to distinguish genuine partial
            # history from server-side truncation and recover missing observations.
            if len(seg_bars) == _IOL_SUSPECT_ROW_CAP:
                if not in_seg_dates:
                    raise IOLHistoryError(
                        f"IOL DatosHistoricos returned {_IOL_SUSPECT_ROW_CAP} out-of-range rows "
                        f"for '{sym}' ({mkt}) on {seg_start.isoformat()}..{seg_end.isoformat()} "
                        "(server appears to cap or ignore date filter).",
                        asset_id=sym,
                        code="iol_truncated_response",
                        details={
                            "symbol": sym,
                            "market": mkt,
                            "segment_start": seg_start.isoformat(),
                            "segment_end": seg_end.isoformat(),
                            "row_cap": _IOL_SUSPECT_ROW_CAP,
                            "truncation_splits": truncation_splits,
                        },
                    )
                seg_min_dt = min(in_seg_dates)
                seg_max_dt = max(in_seg_dates)
                candidate_probes: List[Tuple[date, date]] = []

                lead_end = seg_min_dt - timedelta(days=1)
                if _has_weekday(seg_start, lead_end) and (seg_start, lead_end) not in probed_windows:
                    candidate_probes.append((seg_start, lead_end))

                trail_start = seg_max_dt + timedelta(days=1)
                trail_limit = seg_end
                if exclude_intraday_today and (as_of_date is None or as_of_date >= today_art) and now_art.hour < _BCBA_CLOSE_HOUR_ART:
                    trail_limit = min(trail_limit, today_art - timedelta(days=1))
                if _has_weekday(trail_start, trail_limit) and (trail_start, trail_limit) not in probed_windows:
                    candidate_probes.append((trail_start, trail_limit))

                for p_start, p_end in reversed(candidate_probes):
                    if truncation_splits >= _IOL_MAX_TRUNCATION_SPLITS:
                        raise IOLHistoryError(
                            f"Detected persistent {_IOL_SUSPECT_ROW_CAP}-row truncation from IOL DatosHistoricos "
                            f"for '{sym}' ({mkt}) in segment {seg_start.isoformat()}..{seg_end.isoformat()} "
                            f"(returned span={seg_min_dt.isoformat()}..{seg_max_dt.isoformat()}) "
                            f"after {truncation_splits} adaptive split(s).",
                            asset_id=sym,
                            code="iol_truncated_response",
                            details={
                                "symbol": sym,
                                "market": mkt,
                                "segment_start": seg_start.isoformat(),
                                "segment_end": seg_end.isoformat(),
                                "earliest_returned_date": seg_min_dt.isoformat(),
                                "latest_returned_date": seg_max_dt.isoformat(),
                                "row_cap": _IOL_SUSPECT_ROW_CAP,
                                "truncation_splits": truncation_splits,
                            },
                        )
                    probed_windows.add((p_start, p_end))
                    pending_segments.insert(0, (p_start, p_end, True))

        observed_at = clock(timezone.utc).isoformat()
    finally:
        if owns_session:
            http.close()

    raw_row_count = len(all_bars)
    if raw_row_count == 0:
        raise IOLHistoryError(
            f"IOL DatosHistoricos returned zero historical rows for '{sym}' ({mkt}) "
            f"in requested range {req_start.isoformat()}..{eff_end.isoformat()}.",
            asset_id=sym,
            code="iol_empty_history",
            details={
                "symbol": sym,
                "market": mkt,
                "id_titulo": id_titulo,
                "requested_start_date": req_start.isoformat(),
                "requested_end_date": eff_end.isoformat(),
                "segments_fetched": segments_fetched,
            },
        )

    warnings_list: List[str] = []
    intraday_dropped = 0
    out_of_range_dropped = 0
    in_scope_bars: List[IOLHistoricalBar] = []

    for bar in all_bars:
        if bar.trade_date < req_start or bar.trade_date > eff_end:
            out_of_range_dropped += 1
            continue
        if as_of_date is None or as_of_date >= today_art:
            if bar.trade_date > today_art:
                intraday_dropped += 1
                continue
            if exclude_intraday_today and bar.trade_date == today_art and now_art.hour < _BCBA_CLOSE_HOUR_ART:
                intraday_dropped += 1
                continue
        in_scope_bars.append(bar)

    dedup_bars, dups_resolved = _deduplicate_historical_bars(
        in_scope_bars,
        symbol=sym,
        price_field=eff_price_field,
        duplicate_conflict_policy=duplicate_conflict_policy,
    )

    zero_no_trade_dropped = 0
    zero_volume_no_trade_dropped = 0
    kept_bars: List[IOLHistoricalBar] = []
    for bar in dedup_bars:
        px = _bar_price_value(bar, eff_price_field)
        if not _valid_positive(px):
            zero_no_trade_dropped += 1
            continue
        if eff_drop_zero_vol and _is_zero_volume_no_trade_bar(bar):
            zero_no_trade_dropped += 1
            zero_volume_no_trade_dropped += 1
            continue
        kept_bars.append(bar)

    if intraday_dropped > 0:
        warnings_list.append(
            f"Excluded {intraday_dropped} unclosed intraday/future observation(s) for '{sym}' prior to BCBA close ({_BCBA_CLOSE_HOUR_ART:02d}:00 ART)."
        )
    if truncation_splits > 0:
        warnings_list.append(
            f"Detected {_IOL_SUSPECT_ROW_CAP}-row server truncation for '{sym}'; recovered earlier observations via {truncation_splits} adaptive sub-segment request(s)."
        )

    if not kept_bars:
        raise IOLHistoryError(
            f"IOL DatosHistoricos for '{sym}' ({mkt}) contained {raw_row_count} raw row(s) but zero valid positive "
            f"{eff_price_field} trades in {req_start.isoformat()}..{eff_end.isoformat()} "
            f"(no-trade/zero rows dropped={zero_no_trade_dropped}, zero-volume dropped={zero_volume_no_trade_dropped}, "
            f"out-of-range={out_of_range_dropped}, intraday={intraday_dropped}).",
            asset_id=sym,
            code="iol_no_valid_trades",
            details={
                "symbol": sym,
                "market": mkt,
                "id_titulo": id_titulo,
                "raw_row_count": raw_row_count,
                "zero_or_no_trade_rows_dropped": zero_no_trade_dropped,
                "zero_volume_no_trade_rows_dropped": zero_volume_no_trade_dropped,
                "out_of_range_rows_dropped": out_of_range_dropped,
                "intraday_rows_dropped": intraday_dropped,
            },
        )

    kept_bars.sort(key=lambda b: b.trade_date)
    idx_ns = pd.DatetimeIndex([pd.Timestamp(b.trade_date) for b in kept_bars]).as_unit("ns")
    bars_df = pd.DataFrame(
        {
            "open": [b.open for b in kept_bars],
            "high": [b.high for b in kept_bars],
            "low": [b.low for b in kept_bars],
            "close": [b.close for b in kept_bars],
            "adjusted_close": [b.adjusted_close for b in kept_bars],
            "volume_amount": [b.volume_amount for b in kept_bars],
            "volume_nominal": [b.volume_nominal for b in kept_bars],
        },
        index=idx_ns,
        dtype=float,
    )
    price_values = [float(_bar_price_value(b, eff_price_field)) for b in kept_bars]  # type: ignore[arg-type]
    price_series = pd.Series(price_values, index=idx_ns, name=sym, dtype=float)

    adj_diff_count = sum(
        1
        for b in kept_bars
        if b.close is not None
        and b.adjusted_close is not None
        and not math.isclose(float(b.close), float(b.adjusted_close), rel_tol=1e-9, abs_tol=1e-9)
    )

    eff_start = kept_bars[0].trade_date
    eff_end_actual = kept_bars[-1].trade_date
    leading_partial = eff_start > (req_start + timedelta(days=7))
    trailing_partial = eff_end_actual < (eff_end - timedelta(days=7))
    partial_coverage = leading_partial or trailing_partial

    if leading_partial and trailing_partial:
        warnings_list.append(
            f"Partial historical coverage for IOL symbol '{sym}': requested {req_start.isoformat()}..{eff_end.isoformat()}, "
            f"available valid trades cover {eff_start.isoformat()}..{eff_end_actual.isoformat()} ({len(kept_bars)} daily obs; "
            "both leading pre-issuance gap and trailing stale/matured tail detected)."
        )
    elif leading_partial:
        warnings_list.append(
            f"Partial historical coverage for IOL symbol '{sym}': requested {req_start.isoformat()}..{eff_end.isoformat()}, "
            f"available valid trades cover {eff_start.isoformat()}..{eff_end_actual.isoformat()} ({len(kept_bars)} daily obs; "
            "typical for recent issuance)."
        )
    elif trailing_partial:
        warnings_list.append(
            f"Partial historical coverage (trailing gap/stale tail) for IOL symbol '{sym}': requested {req_start.isoformat()}..{eff_end.isoformat()}, "
            f"last valid traded observation is {eff_end_actual.isoformat()} ({len(kept_bars)} daily obs; "
            "typical for matured, delisted, or illiquid fixed income)."
        )

    if zero_no_trade_dropped > 0:
        if zero_volume_no_trade_dropped > 0:
            warnings_list.append(
                f"Dropped {zero_no_trade_dropped} zero-price/no-trade row(s) from IOL history for '{sym}' "
                f"(including {zero_volume_no_trade_dropped} zero-volume carried-forward row(s)) without forward-filling."
            )
        else:
            warnings_list.append(
                f"Dropped {zero_no_trade_dropped} zero-price/no-trade row(s) from IOL history for '{sym}' without forward-filling."
            )
    if dups_resolved > 0:
        warnings_list.append(
            f"Collapsed {dups_resolved} identical duplicate date row(s) deterministically for IOL symbol '{sym}'."
        )

    if eff_price_field == "adjusted_close":
        warnings_list.append(
            f"IOL historical series for '{sym}' uses 'Cierre ajustado' (differs from 'Cierre' on {adj_diff_count}/{len(kept_bars)} rows). "
            "For Argentine fixed income, IOL 'Cierre ajustado' does NOT incorporate coupon payments or principal amortizations; "
            "treat as market price return only, not total return."
        )
    else:
        warnings_list.append(
            f"IOL historical series for '{sym}' uses unadjusted market 'Cierre' (price return only, NOT total return; "
            "coupon payments and principal amortizations are not reinvested or adjusted by IOL)."
        )

    result = IOLHistoricalSeries(
        symbol=sym,
        market=mkt,
        id_titulo=id_titulo,
        requested_start_date=req_start,
        requested_end_date=eff_end,
        effective_start_date=eff_start,
        effective_end_date=eff_end_actual,
        frequency=freq,
        price_field=eff_price_field,
        series_type="market_price",
        currency=currency,
        usd_settlement_type=usd_settlement_type,
        quote_unit=quote_unit,
        observed_at=observed_at,
        source_url=source_url,
        bars=bars_df,
        price_series=price_series,
        raw_row_count=raw_row_count,
        valid_row_count=len(kept_bars),
        zero_or_no_trade_rows_dropped=zero_no_trade_dropped,
        zero_volume_no_trade_rows_dropped=zero_volume_no_trade_dropped,
        duplicate_dates_resolved=dups_resolved,
        out_of_range_rows_dropped=out_of_range_dropped,
        intraday_rows_dropped=intraday_dropped,
        segments_fetched=segments_fetched,
        truncation_splits_performed=truncation_splits,
        adjusted_close_differs_count=adj_diff_count,
        partial_coverage=partial_coverage,
        is_total_return=False,
        warnings=warnings_list,
    )

    if cache is not None:
        cache[cache_key] = result
    return result


def _is_known_ars_sibling_of_usd_species(candidate_sym: str, usd_species_sym: str) -> bool:
    """Return True when ``candidate_sym`` is the ARS root sibling of a verified D/C USD species."""
    c = candidate_sym.strip().upper()
    u = usd_species_sym.strip().upper()
    if not c or not u or c == u:
        return False
    if u in (f"{c}D", f"{c}C"):
        return True
    if len(c) == 5 and c.endswith("O") and u in (f"{c[:-1]}D", f"{c[:-1]}C"):
        return True
    return False


def _validate_unquoted_iol_symbol_for_currency(
    target_sym: str,
    declared_currency: str,
    instrument_id: str,
) -> None:
    """Verify an unquoted/unmapped IOL fixed-income symbol does not contradict declared USD/USD_CABLE currency.

    Without a live IOL panel quote or explicit ``iol_species_map`` entry confirming the species,
    BCBA Argentine fixed-income symbols MUST explicitly denote species ``D`` (for ``USD``) or
    species ``C`` (for ``USD_CABLE``) so ARS roots (e.g. ``AL30``, ``GD30``, ``MTCGO``, ``AEC3O``)
    and cable species (``AL30C``) can never silently enter a local USD series.
    """
    sym = target_sym.strip().upper()
    if declared_currency == "USD":
        if len(sym) >= 4 and sym.endswith("C"):
            raise IOLHistoryError(
                f"Currency/species conflict for '{instrument_id}': symbol '{sym}' indicates USD Cable (C) "
                "while declared currency is 'USD'. Supply an explicit 'iol_species_map' or configure USD_CABLE with opt-in.",
                asset_id=instrument_id,
                code="iol_currency_incompatible",
                details={"symbol": sym, "declared_currency": declared_currency},
            )
        if not (len(sym) >= 4 and sym.endswith("D")):
            raise IOLHistoryError(
                f"Cannot verify that unmapped IOL symbol '{sym}' for '{instrument_id}' is denominated in local USD "
                "(lacks verified panel quote, explicit iol_species_map['D'], and BCBA 'D' species suffix; "
                "refusing to risk ingesting ARS root prices as USD).",
                asset_id=instrument_id,
                code="iol_currency_incompatible",
                details={"symbol": sym, "declared_currency": declared_currency},
            )
    elif declared_currency == "USD_CABLE":
        if not (len(sym) >= 4 and sym.endswith("C")):
            raise IOLHistoryError(
                f"Cannot verify that unmapped IOL symbol '{sym}' for '{instrument_id}' settles in USD Cable (C) "
                "(lacks verified panel quote, explicit iol_species_map['C'], and BCBA 'C' species suffix).",
                asset_id=instrument_id,
                code="iol_currency_incompatible",
                details={"symbol": sym, "declared_currency": declared_currency},
            )


def _resolve_iol_historical_target(
    inst: InstrumentConfig,
    *,
    iol_quote: Optional[IOLQuoteResult],
    iol_config: IOLConfig,
) -> Tuple[str, str, str, str, str, List[str]]:
    """Resolve validated ``(iol_symbol, iol_market, currency, usd_settlement_type, quote_unit, warnings)`` for an IOL instrument.

    Never guesses currency or bond species from arbitrary trailing suffixes and never
    silently mixes ARS, USD Cable (C), and USD Local (D) into a USD portfolio series.
    """
    iid = inst.instrument_id
    warnings_list: List[str] = []
    iol_market = inst.iol_market or (inst.market if inst.market != "US" else iol_config.default_market)
    quote_unit = (
        inst.fixed_income_terms.price_convention
        if inst.fixed_income_terms is not None
        else (inst.quote_unit or (iol_quote.quote_unit if iol_quote is not None else "per_100_original_nominal"))
    )

    # Reject explicit non-USD currency declarations on instruments entering the USD portfolio
    if inst.currency is not None and inst.currency not in ("USD", "USD_CABLE"):
        raise IOLHistoryError(
            f"Instrument '{iid}' declares currency '{inst.currency}', which is incompatible with direct USD IOL historical "
            "series ingestion (spot MEP and official ARS=X are never applied to historical Argentine fixed-income prices).",
            asset_id=iid,
            code="iol_currency_incompatible",
            details={"instrument_id": iid, "declared_currency": inst.currency},
        )

    species_map = inst.iol_species_map or {}
    if inst.iol_symbol is not None and species_map:
        matched_keys = [k for k, v in species_map.items() if v == inst.iol_symbol]
        if not matched_keys:
            raise IOLHistoryError(
                f"Explicit iol_symbol='{inst.iol_symbol}' for '{iid}' is not present in explicit iol_species_map={species_map}; "
                "refusing mismatched species configuration.",
                asset_id=iid,
                code="iol_currency_incompatible",
                details={"iol_symbol": inst.iol_symbol, "iol_species_map": species_map},
            )
        if len(matched_keys) > 1:
            raise IOLHistoryError(
                f"Explicit iol_symbol='{inst.iol_symbol}' for '{iid}' matches multiple species {matched_keys} in iol_species_map.",
                asset_id=iid,
                code="iol_currency_incompatible",
                details={"iol_symbol": inst.iol_symbol, "matched_species": matched_keys},
            )

    def _resolve_from_species_entry(sp_key: str, sp_sym: str) -> Tuple[str, str, str, str, str, List[str]]:
        if sp_key == "ARS":
            raise IOLHistoryError(
                f"IOL species '{sp_sym}' for '{iid}' maps to ARS; refusing to mix ARS historical prices into USD series.",
                asset_id=iid,
                code="iol_currency_incompatible",
                details={"symbol": sp_sym, "species": "ARS"},
            )
        if sp_key == "C":
            if inst.currency == "USD":
                raise IOLHistoryError(
                    f"Currency/settlement conflict for '{iid}': declared 'USD' conflicts with explicit USD Cable species '{sp_sym}' (C).",
                    asset_id=iid,
                    code="iol_currency_incompatible",
                    details={"symbol": sp_sym, "declared_currency": inst.currency, "species": "C"},
                )
            if not iol_config.allow_usd_cable_history:
                raise IOLHistoryError(
                    f"Explicit species '{sp_sym}' for '{iid}' maps to USD Cable (C); set IOLConfig(allow_usd_cable_history=True) to opt in.",
                    asset_id=iid,
                    code="iol_currency_incompatible",
                    details={"symbol": sp_sym, "species": "C"},
                )
            warnings_list.append(
                f"Using explicitly mapped USD Cable (C) historical symbol '{sp_sym}' for '{iid}' (allow_usd_cable_history=True)."
            )
            return sp_sym, iol_market, "USD_CABLE", "USD_CABLE_C", quote_unit, warnings_list
        if sp_key == "D":
            if inst.currency == "USD_CABLE":
                raise IOLHistoryError(
                    f"Currency/settlement conflict for '{iid}': declared 'USD_CABLE' conflicts with explicit USD Local species '{sp_sym}' (D).",
                    asset_id=iid,
                    code="iol_currency_incompatible",
                    details={"symbol": sp_sym, "declared_currency": inst.currency, "species": "D"},
                )
            return sp_sym, iol_market, "USD", "USD_LOCAL_D", quote_unit, warnings_list
        raise IOLHistoryError(
            f"Unsupported species key '{sp_key}' in iol_species_map for '{iid}'.",
            asset_id=iid,
            code="iol_currency_incompatible",
            details={"species": sp_key, "symbol": sp_sym},
        )

    if iol_quote is not None:
        if inst.iol_symbol is None or inst.iol_symbol == iol_quote.species_used:
            target_sym = iol_quote.species_used
            if iol_quote.usd_settlement_type == "USD_LOCAL_D":
                if inst.currency == "USD_CABLE":
                    raise IOLHistoryError(
                        f"Currency/settlement conflict for '{iid}': declared 'USD_CABLE' conflicts with IOL local USD species '{target_sym}' (USD_LOCAL_D).",
                        asset_id=iid,
                        code="iol_currency_incompatible",
                        details={"declared_currency": inst.currency, "species_used": target_sym, "usd_settlement_type": "USD_LOCAL_D"},
                    )
                return target_sym, iol_market, "USD", "USD_LOCAL_D", quote_unit, warnings_list

            if iol_quote.usd_settlement_type == "USD_CABLE_C":
                if inst.currency == "USD":
                    raise IOLHistoryError(
                        f"Currency/settlement conflict for '{iid}': declared 'USD' conflicts with IOL USD Cable species '{target_sym}' (USD_CABLE_C).",
                        asset_id=iid,
                        code="iol_currency_incompatible",
                        details={"declared_currency": inst.currency, "species_used": target_sym, "usd_settlement_type": "USD_CABLE_C"},
                    )
                if not iol_config.allow_usd_cable_history:
                    raise IOLHistoryError(
                        f"IOL historical species '{target_sym}' for '{iid}' settles in USD Cable (USD_CABLE_C), "
                        "which is economically distinct from USD Local (D) and US USD. Refusing silent mixing; "
                        "set IOLConfig(allow_usd_cable_history=True) to explicitly opt in.",
                        asset_id=iid,
                        code="iol_currency_incompatible",
                        details={"symbol": target_sym, "usd_settlement_type": "USD_CABLE_C"},
                    )
                warnings_list.append(
                    f"Using USD Cable (C) historical series '{target_sym}' for '{iid}' under explicit opt-in "
                    "(allow_usd_cable_history=True); USD Cable is not 1:1 equivalent to USD Local (D)."
                )
                return target_sym, iol_market, "USD_CABLE", "USD_CABLE_C", quote_unit, warnings_list

            # USD_MEP_IMPLIED -> ARS-only species on IOL
            raise IOLHistoryError(
                f"Instrument '{iid}' only resolved ARS species '{target_sym}' on IOL (conversion_method='{iol_quote.conversion_method}'). "
                "Refusing to ingest ARS historical prices as USD or convert history using spot AL30/AL30D MEP or official ARS=X. "
                "Configure a validated USD species via 'iol_symbol' / 'iol_species_map' or supply an external USD series.",
                asset_id=iid,
                code="iol_currency_incompatible",
                details={"symbol": target_sym, "original_currency": iol_quote.original_currency, "usd_settlement_type": iol_quote.usd_settlement_type},
            )

        # Explicit inst.iol_symbol differs from iol_quote.species_used -> verify explicit mapping/currency
        target_sym = inst.iol_symbol
        if species_map:
            matched_sp = next(k for k, v in species_map.items() if v == target_sym)
            return _resolve_from_species_entry(matched_sp, target_sym)
        if iol_quote.usd_settlement_type in ("USD_LOCAL_D", "USD_CABLE_C") and _is_known_ars_sibling_of_usd_species(
            target_sym, iol_quote.species_used
        ):
            raise IOLHistoryError(
                f"Explicit iol_symbol='{target_sym}' for '{iid}' corresponds to an ARS species (USD sibling '{iol_quote.species_used}'); "
                "refusing to mix ARS historical prices into USD series.",
                asset_id=iid,
                code="iol_currency_incompatible",
                details={"iol_symbol": target_sym, "quote_species_used": iol_quote.species_used, "species": "ARS"},
            )
        if inst.currency == "USD":
            _validate_unquoted_iol_symbol_for_currency(target_sym, "USD", iid)
            warnings_list.append(
                f"Using explicit iol_symbol='{target_sym}' for '{iid}' with declared currency='USD' (differs from panel quote species '{iol_quote.species_used}')."
            )
            return target_sym, iol_market, "USD", "USD_LOCAL_D", quote_unit, warnings_list
        if inst.currency == "USD_CABLE":
            _validate_unquoted_iol_symbol_for_currency(target_sym, "USD_CABLE", iid)
            if not iol_config.allow_usd_cable_history:
                raise IOLHistoryError(
                    f"Explicit iol_symbol='{target_sym}' for '{iid}' declares USD_CABLE without allow_usd_cable_history=True.",
                    asset_id=iid,
                    code="iol_currency_incompatible",
                    details={"iol_symbol": target_sym, "declared_currency": "USD_CABLE"},
                )
            warnings_list.append(
                f"Using explicit USD Cable iol_symbol='{target_sym}' for '{iid}' under explicit opt-in."
            )
            return target_sym, iol_market, "USD_CABLE", "USD_CABLE_C", quote_unit, warnings_list
        raise IOLHistoryError(
            f"Cannot verify currency/species comparability for explicit iol_symbol='{target_sym}' on '{iid}' "
            f"(differs from panel quote species '{iol_quote.species_used}' and lacks explicit currency='USD' or iol_species_map entry).",
            asset_id=iid,
            code="iol_currency_incompatible",
            details={"iol_symbol": target_sym, "quote_species_used": iol_quote.species_used, "declared_currency": inst.currency},
        )

    # No IOL panel quote supplied -> evaluate explicit iol_species_map FIRST, then strict verified currency + species checks
    if species_map:
        if inst.iol_symbol is not None:
            matched_sp = next(k for k, v in species_map.items() if v == inst.iol_symbol)
            return _resolve_from_species_entry(matched_sp, inst.iol_symbol)
        if "D" in species_map and inst.currency != "USD_CABLE":
            return _resolve_from_species_entry("D", species_map["D"])
        if "C" in species_map:
            return _resolve_from_species_entry("C", species_map["C"])
        if "D" in species_map:
            return _resolve_from_species_entry("D", species_map["D"])
        if "ARS" in species_map:
            return _resolve_from_species_entry("ARS", species_map["ARS"])

    target_sym = inst.iol_symbol or inst.ticker.strip().upper()
    if inst.currency == "USD":
        _validate_unquoted_iol_symbol_for_currency(target_sym, "USD", iid)
        return target_sym, iol_market, "USD", "USD_LOCAL_D", quote_unit, warnings_list
    if inst.currency == "USD_CABLE":
        _validate_unquoted_iol_symbol_for_currency(target_sym, "USD_CABLE", iid)
        if not iol_config.allow_usd_cable_history:
            raise IOLHistoryError(
                f"Instrument '{iid}' declares currency='USD_CABLE' without allow_usd_cable_history=True.",
                asset_id=iid,
                code="iol_currency_incompatible",
                details={"symbol": target_sym, "declared_currency": "USD_CABLE"},
            )
        warnings_list.append(
            f"Using declared USD Cable historical symbol '{target_sym}' for '{iid}' under explicit opt-in."
        )
        return target_sym, iol_market, "USD_CABLE", "USD_CABLE_C", quote_unit, warnings_list

    raise IOLHistoryError(
        f"Cannot verify quote currency/species for IOL historical symbol '{target_sym}' ('{iid}'): "
        "neither an IOL panel quote nor explicit currency='USD' metadata is available (never inferring currency from ticker suffix).",
        asset_id=iid,
        code="iol_currency_incompatible",
        details={"symbol": target_sym, "declared_currency": inst.currency},
    )


def _clip_preloaded_historical_series(
    pre_hit: IOLHistoricalSeries,
    *,
    start_date: Optional[date],
    end_date: Optional[date],
    as_of_date: Optional[date],
    extra_warnings: List[str],
) -> IOLHistoricalSeries:
    """Return a non-mutated copy of ``pre_hit`` strictly clipped to ``[start_date, eff_end]``."""
    eff_end: Optional[date] = None
    if end_date is not None and as_of_date is not None:
        eff_end = min(end_date, as_of_date)
    elif as_of_date is not None:
        eff_end = as_of_date
    elif end_date is not None:
        eff_end = end_date

    px_s = pre_hit.price_series.copy()
    bars_df = pre_hit.bars.copy()

    if not px_s.empty:
        px_idx = pd.to_datetime(px_s.index)
        if px_idx.tz is not None:
            px_idx = px_idx.tz_convert(_AR_TZ).tz_localize(None)
        px_s.index = px_idx.as_unit("ns")
        mask = px_s.index.notna()
        if start_date is not None:
            mask = mask & (px_s.index >= pd.Timestamp(start_date))
        if eff_end is not None:
            mask = mask & (px_s.index <= (pd.Timestamp(eff_end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)))
        px_s = px_s.iloc[mask]

    if not bars_df.empty:
        b_idx = pd.to_datetime(bars_df.index)
        if b_idx.tz is not None:
            b_idx = b_idx.tz_convert(_AR_TZ).tz_localize(None)
        bars_df.index = b_idx.as_unit("ns")
        b_mask = bars_df.index.notna()
        if start_date is not None:
            b_mask = b_mask & (bars_df.index >= pd.Timestamp(start_date))
        if eff_end is not None:
            b_mask = b_mask & (bars_df.index <= (pd.Timestamp(eff_end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)))
        bars_df = bars_df.iloc[b_mask]

    merged_warns = list(dict.fromkeys(list(pre_hit.warnings) + list(extra_warnings)))

    eff_start_dt = pd.Timestamp(px_s.index.min()).date() if not px_s.empty else None
    eff_end_dt = pd.Timestamp(px_s.index.max()).date() if not px_s.empty else None
    return replace(
        pre_hit,
        requested_start_date=start_date if start_date is not None else pre_hit.requested_start_date,
        requested_end_date=eff_end if eff_end is not None else pre_hit.requested_end_date,
        effective_start_date=eff_start_dt,
        effective_end_date=eff_end_dt,
        bars=bars_df,
        price_series=px_s,
        valid_row_count=len(px_s),
        warnings=merged_warns,
    )


def fetch_iol_historical_prices_for_instruments(
    instruments: Iterable[InstrumentConfig],
    *,
    iol_quotes: Optional[Dict[str, IOLQuoteResult]] = None,
    period: str = "10y",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    as_of_date: Optional[date] = None,
    iol_config: Optional[IOLConfig] = None,
    session: Optional[requests.Session] = None,
    cache: Optional[IOLHistoryCache] = None,
    preloaded_history: Optional[Dict[str, IOLHistoricalSeries]] = None,
    history_fetcher: Optional[Callable[..., IOLHistoricalSeries]] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
    now_fn: Optional[Callable[[Any], datetime]] = None,
) -> Dict[str, IOLHistoricalSeries]:
    """Fetch normalized IOL historical USD price series for all IOL fixed-income instruments.

    Uses a request-scoped in-memory cache and shared HTTP session so repeated requests
    for the same symbol/range within a run do not incur duplicate downloads, while
    preserving strict stateless isolation (zero disk or process-global state).
    """
    cfg = iol_config or IOLConfig()
    iol_instruments = [
        inst for inst in instruments if inst.quote_source in ("iol_bonos", "iol_ons")
    ]
    if not iol_instruments:
        return {}

    quotes_map = iol_quotes or {}
    req_cache: IOLHistoryCache = cache if cache is not None else {}
    fetcher = history_fetcher or fetch_iol_historical_prices
    sleeper = sleep_fn or time.sleep

    results: Dict[str, IOLHistoricalSeries] = {}
    owns_session = session is None and history_fetcher is None and preloaded_history is None
    http = requests.Session() if owns_session else session

    try:
        fetched_network_count = 0
        for inst in iol_instruments:
            iid = inst.instrument_id
            iq = quotes_map.get(iid)
            target_sym, target_mkt, eff_ccy, eff_settle, eff_unit, target_warns = _resolve_iol_historical_target(
                inst,
                iol_quote=iq,
                iol_config=cfg,
            )

            if preloaded_history is not None:
                pre_hit = (
                    preloaded_history.get(iid)
                    or preloaded_history.get(target_sym)
                    or preloaded_history.get(inst.ticker.strip().upper())
                )
                if pre_hit is None:
                    continue
                if pre_hit.currency is None or pre_hit.currency not in ("USD", "USD_CABLE"):
                    raise IOLHistoryError(
                        f"Preloaded IOL history for '{iid}' has missing or incompatible currency '{pre_hit.currency}'.",
                        asset_id=iid,
                        code="iol_currency_incompatible",
                    )
                if pre_hit.usd_settlement_type == "USD_MEP_IMPLIED":
                    raise IOLHistoryError(
                        f"Preloaded IOL history for '{iid}' is tagged USD_MEP_IMPLIED (ARS); refusing USD ingestion.",
                        asset_id=iid,
                        code="iol_currency_incompatible",
                    )
                if pre_hit.usd_settlement_type == "USD_CABLE_C" and not cfg.allow_usd_cable_history:
                    raise IOLHistoryError(
                        f"Preloaded IOL history for '{iid}' is USD_CABLE_C without allow_usd_cable_history=True.",
                        asset_id=iid,
                        code="iol_currency_incompatible",
                    )
                if pre_hit.currency != eff_ccy:
                    raise IOLHistoryError(
                        f"Preloaded IOL history currency '{pre_hit.currency}' for '{iid}' conflicts with resolved target currency '{eff_ccy}'.",
                        asset_id=iid,
                        code="iol_currency_incompatible",
                    )
                if pre_hit.usd_settlement_type is not None and pre_hit.usd_settlement_type != eff_settle:
                    raise IOLHistoryError(
                        f"Preloaded IOL history settlement type '{pre_hit.usd_settlement_type}' for '{iid}' conflicts with resolved target '{eff_settle}'.",
                        asset_id=iid,
                        code="iol_currency_incompatible",
                    )
                results[iid] = _clip_preloaded_historical_series(
                    pre_hit,
                    start_date=start_date,
                    end_date=end_date,
                    as_of_date=as_of_date,
                    extra_warnings=target_warns,
                )
                continue

            cache_size_before = len(req_cache)
            inter_delay = (
                cfg.historical_request_delay_seconds
                if fetched_network_count > 0 and cfg.historical_request_delay_seconds > 0.0 and history_fetcher is None
                else 0.0
            )

            kwargs: Dict[str, Any] = {
                "symbol": target_sym,
                "market": target_mkt,
                "frequency": "daily",
                "price_field": cfg.historical_price_field,
                "as_of_date": as_of_date,
                "currency": eff_ccy,
                "usd_settlement_type": eff_settle,
                "quote_unit": eff_unit,
                "iol_config": cfg,
                "session": http,
                "cache": req_cache,
                "sleep_fn": sleeper,
                "now_fn": now_fn,
            }
            if history_fetcher is None and inter_delay > 0.0:
                kwargs["_initial_delay_seconds"] = inter_delay
            if start_date is not None or end_date is not None:
                kwargs["start_date"] = start_date
                kwargs["end_date"] = end_date
            else:
                kwargs["period"] = period

            hist = fetcher(**kwargs)
            if history_fetcher is not None or len(req_cache) > cache_size_before:
                fetched_network_count += 1
            if target_warns:
                merged_warns = list(dict.fromkeys(list(hist.warnings) + target_warns))
                results[iid] = replace(hist, warnings=merged_warns)
            else:
                results[iid] = hist
    finally:
        if owns_session and http is not None:
            http.close()

    return results
