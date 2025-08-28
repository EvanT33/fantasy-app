import asyncio
import re
import sys
import argparse
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

# =========================
# Config / helpers
# =========================

POSITIONS = ["QB", "RB", "WR", "TE", "K", "D/ST"]  # ESPN uses "D/ST" for defenses

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def to_float(x):
    try:
        if x in (None, "", "-", "—", "–"):
            return np.nan
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan

def compute_half_ppr(row: Dict[str, Any], pos: str) -> float:
    """
    Standard half-PPR:
      Pass: 1/25 yds, 4/TD, -2/INT
      Rush: 1/10 yds, 6/TD
      Rec : 1/10 yds, 6/TD, +0.5/REC
      Fumbles lost: -2
    """
    def g(k):
        return to_float(row.get(k))

    if pos in ("K", "D/ST"):
        return np.nan  # compute reliably only for offensive positions

    pass_yds = g("PASS YDS") or g("PASSING YDS") or g("PYDS") or 0
    pass_td  = g("PASS TD")  or g("PASSING TD")  or g("PTD")   or 0
    ints     = g("INT")      or g("INTS")        or 0

    rush_yds = g("RUSH YDS") or g("RUSHING YDS") or g("RYDS") or 0
    rush_td  = g("RUSH TD")  or g("RUSHING TD")  or g("RTD")  or 0

    rec      = g("REC") or g("RECEPTIONS") or 0
    rec_yds  = g("REC YDS") or g("RECEIVING YDS") or g("RECYDS") or 0
    rec_td   = g("REC TD")  or g("RECEIVING TD")  or g("RECTD")  or 0

    fum_lost = g("FL") or g("FUM") or g("FUM LOST") or 0
    two_pt   = g("2PT") or g("2-PT") or 0

    pts = 0.0
    pts += (pass_yds or 0) / 25.0
    pts += (pass_td or 0) * 4.0
    pts += (two_pt or 0) * 2.0
    pts -= (ints or 0) * 2.0

    pts += (rush_yds or 0) / 10.0
    pts += (rush_td or 0) * 6.0

    pts += (rec_yds or 0) / 10.0
    pts += (rec_td or 0) * 6.0
    pts += (rec or 0) * 0.5

    pts -= (fum_lost or 0) * 2.0
    return round(pts, 2)


# =========================
# Scraper core (Playwright)
# =========================

async def scrape_position(context, position: str, season: int, delay: float) -> pd.DataFrame:
    """
    Loads ESPN projections for the season, selects the position tab, paginates,
    and returns a DataFrame of rows with Name/Team/Pos + raw columns.
    """
    page = await context.new_page()
    url = f"https://fantasy.espn.com/football/players/projections?seasonId={season}"
    await page.goto(url, wait_until="networkidle")

    # --- Select the position tab / filter
    async def set_position():
        # Try tab by role=name
        try:
            tab = page.get_by_role("tab", name=re.compile(rf"^{re.escape(position)}$", re.I))
            if await tab.count() > 0:
                await tab.first.click()
                await page.wait_for_timeout(int(delay * 1000))
                return True
        except Exception:
            pass
        # Try buttons that look like tabs
        try:
            btn = page.get_by_role("button", name=re.compile(rf"^{re.escape(position)}$", re.I))
            if await btn.count() > 0:
                await btn.first.click()
                await page.wait_for_timeout(int(delay * 1000))
                return True
        except Exception:
            pass
        # If nothing found, we continue (page may default to "All")
        return False

    await set_position()

    rows: List[Dict[str, Any]] = []

    while True:
        # find the main table
        tables = await page.locator("table").all()
        if not tables:
            await page.wait_for_timeout(1500)
            tables = await page.locator("table").all()
            if not tables:
                break

        table = tables[0]

        # headers
        ths = table.locator("thead tr th")
        n_ths = await ths.count()
        headers: List[str] = []
        for i in range(n_ths):
            headers.append(norm(await ths.nth(i).inner_text()) or f"COL_{i}")

        # rows
        trs = table.locator("tbody tr")
        n_trs = await trs.count()

        for ri in range(n_trs):
            tds = trs.nth(ri).locator("td")
            n_tds = await tds.count()
            if n_tds == 0:
                continue

            row_map: Dict[str, Any] = {}

            # First cell: reliably get the player link text for the Name.
            first_td = tds.nth(0)
            name_text = ""
            try:
                # Most ESPN tables render the player as <a class="AnchorLink">Name</a>
                name_text = norm(await first_td.locator("a").first.inner_text())
            except Exception:
                # fallback to entire cell text
                name_text = norm(await first_td.inner_text())

            # Also try to get a team/pos hint from the first cell (e.g., "Ja'Marr Chase Cin WR")
            full_first = norm(await first_td.inner_text())
            team_hint = ""
            pos_hint = ""
            m = re.search(r"[ ,]([A-Za-z]{2,3})\s+(QB|RB|WR|TE|K|D/?ST)\b", full_first, re.I)
            if m:
                team_hint = m.group(1).upper()
                pos_hint = m.group(2).upper().replace("D/ST", "D/ST")

            # Map remaining columns
            for ci in range(1, min(n_tds, len(headers))):
                key = headers[ci]
                txt = norm(await tds.nth(ci).inner_text())
                row_map[key] = txt

            # Fill Name / Team / Pos
            row_map["Name"] = name_text
            row_map["Team"] = team_hint
            # Prefer explicit POS column if present; else use hint; else fall back to tab
            pos_from_table = None
            for key in headers:
                if key.upper() in ("POS", "POSITION"):
                    pos_from_table = norm(row_map.get(key, ""))
                    break
            pos_final = (pos_from_table or pos_hint or position).upper()
            pos_final = pos_final.replace("D/ST", "D/ST").replace("DEF", "D/ST")
            row_map["Pos"] = pos_final

            rows.append(row_map)

        # Go to the next page if available
        next_clicked = False
        try:
            # Primary "Next" button
            nxt = page.get_by_role("button", name=re.compile(r"^Next$", re.I))
            if await nxt.count() > 0 and not await nxt.first.is_disabled():
                await nxt.first.click()
                await page.wait_for_timeout(int(delay * 1000))
                next_clicked = True
        except Exception:
            pass
        if not next_clicked:
            # Some pages have "Next Page" control
            try:
                nxt2 = page.get_by_label(re.compile("Next Page", re.I))
                if await nxt2.count() > 0 and not await nxt2.first.is_disabled():
                    await nxt2.first.click()
                    await page.wait_for_timeout(int(delay * 1000))
                    next_clicked = True
            except Exception:
                pass

        if not next_clicked:
            break

    await page.close()
    df = pd.DataFrame(rows)

    # Try to detect an ESPN fantasy points column if present
    fpts_cols = [c for c in df.columns if re.search(r"\bfpts?\b", c, re.I)]
    if fpts_cols:
        df["FPTS_raw"] = pd.to_numeric(df[fpts_cols[0]].str.replace(",", "", regex=False), errors="coerce")
    else:
        df["FPTS_raw"] = np.nan
    return df


async def scrape_all(season: int, delay: float, headless: bool) -> pd.DataFrame:
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 1000},
        )

        frames: List[pd.DataFrame] = []
        for pos in tqdm(POSITIONS, desc="Scraping positions"):
            try:
                frames.append(await scrape_position(context, pos, season, delay))
            except Exception as e:
                print(f"[WARN] {pos} failed: {e}", file=sys.stderr)

        await context.close()
        await browser.close()

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Name", "Team", "Pos"], keep="first")

    # Normalize casing
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Team"] = df["Team"].astype(str).str.upper().str.strip()
    df["Pos"]  = df["Pos"].astype(str).str.upper().str.replace("D/ST", "DST", regex=False)

    # Compute Half-PPR when we can (offensive positions)
    half = []
    for _, r in df.iterrows():
        pos = r.get("Pos", "")
        fpts = r.get("FPTS_raw", np.nan)
        if pos in ("QB", "RB", "WR", "TE"):
            # if ESPN didn't provide FPTS (or looks blank), compute from stats
            half.append(compute_half_ppr(r, pos) if (pd.isna(fpts) or True) else to_float(fpts))
        else:
            half.append(to_float(fpts))  # K/DST: use raw if present, else NaN
    df["HalfPPR_Points"] = pd.to_numeric(half, errors="coerce").round(2)

    # Reorder columns: main first, then everything else for inspection
    keep = ["Name", "Team", "Pos", "HalfPPR_Points", "FPTS_raw"]
    others = [c for c in df.columns if c not in keep]
    out = df[keep + others]
    return out


def main():
    ap = argparse.ArgumentParser(description="Scrape ESPN Half-PPR projections for a season (all positions).")
    ap.add_argument("--season", type=int, default=2025, help="Season year (default 2025)")
    ap.add_argument("--delay", type=float, default=1.2, help="Delay between UI steps (seconds)")
    ap.add_argument("--no-headless", action="store_true", help="Show the browser window")
    ap.add_argument("--out", type=str, default="espn_2025_half_ppr_projections.csv", help="Output CSV")
    args = ap.parse_args()

    headless = not args.no_headless

    try:
        df = asyncio.run(scrape_all(args.season, args.delay, headless))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        df = loop.run_until_complete(scrape_all(args.season, args.delay, headless))

    if df.empty:
        print("No data scraped—ESPN layout may have changed.")
        sys.exit(2)

    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
