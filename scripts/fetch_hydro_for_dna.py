#!/usr/bin/env python3
"""Fetch daily useful reservoir level for Colombia's SIN from XM public API.

Temporary utility for the DNA vs reservoir-level analysis. It uses only the
Python standard library so it can run on a clean GitHub-hosted runner.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_URL = "https://servapibi.xm.com.co/daily"
START = dt.date(2019, 1, 1)
END = dt.date(2026, 7, 10)
OUT = Path("outputs/xm_embalses_sin_2019_2026.csv")


def chunks(start: dt.date, end: dt.date, days: int = 29):
    current = start
    while current <= end:
        stop = min(current + dt.timedelta(days=days), end)
        yield current, stop
        current = stop + dt.timedelta(days=1)


def extract_numeric(entity: dict):
    for key in ("Value", "value", "Valor", "valor", "Val", "val"):
        if key in entity:
            try:
                return float(entity[key])
            except (TypeError, ValueError):
                pass
    for key, value in entity.items():
        if str(key).lower() in {"id", "name", "code", "codigo"}:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def fetch_one(period):
    start, end = period
    payload = {
        "MetricId": "PorcVoluUtilDiar",
        "StartDate": start.isoformat(),
        "EndDate": end.isoformat(),
        "Entity": "Sistema",
        "Filter": [],
    }
    body = json.dumps(payload).encode("utf-8")
    last_error = None
    for attempt in range(6):
        req = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 DNA-Embalses-Analysis/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            rows = []
            for item in data.get("Items", []):
                date_text = item.get("Date") or item.get("date")
                if not date_text:
                    continue
                date_text = str(date_text)[:10]
                entities = item.get("DailyEntities") or item.get("dailyEntities") or []
                for entity in entities:
                    value = extract_numeric(entity)
                    if value is not None:
                        rows.append((date_text, value))
                        break
            if not rows:
                raise RuntimeError(f"No rows in response for {start} to {end}: {str(data)[:500]}")
            return rows
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Failed {start} to {end}: {last_error}")


def main():
    periods = list(chunks(START, END))
    rows = []
    errors = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_one, p): p for p in periods}
        for idx, future in enumerate(as_completed(futures), 1):
            period = futures[future]
            try:
                part = future.result()
                rows.extend(part)
                print(f"[{idx}/{len(periods)}] OK {period[0]} to {period[1]}: {len(part)} rows", flush=True)
            except Exception as exc:
                errors.append((period, str(exc)))
                print(f"[{idx}/{len(periods)}] ERROR {period}: {exc}", flush=True)

    if errors:
        raise SystemExit("Some periods failed: " + json.dumps(errors, default=str)[:4000])

    # Deduplicate by date; XM returns fractions in [0,1] for this metric.
    by_date = {}
    for date_text, raw_value in rows:
        by_date[date_text] = raw_value

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fecha", "nivel_util_fraccion", "nivel_util_pct", "fuente"])
        for date_text in sorted(by_date):
            value = by_date[date_text]
            pct = value * 100.0 if abs(value) <= 1.5 else value
            writer.writerow([date_text, f"{value:.8f}", f"{pct:.6f}", API_URL])

    print(f"Saved {len(by_date)} daily rows to {OUT}")
    print(f"Date coverage: {min(by_date)} to {max(by_date)}")


if __name__ == "__main__":
    main()
