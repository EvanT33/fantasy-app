<!-- @format -->

# 🏈 Live Auction Optimizer

A Streamlit app to optimize your **auction draft** starters in real time.  
Now supports **custom starter slots** (QB/RB/WR/TE/K/FLEX) and a **custom bench size**, excluding DST.  
Projections are treated as **Half-PPR** based on the columns in your CSV.

---

## 📦 Quick Start (Windows)

These steps assume **Windows 10/11 with PowerShell**.

### 0) Prereqs

- **Python 3.9+** installed
- **Git** installed (optional but recommended)

### 1) Get the app files

**Option A — Clone the repo (recommended):**

```powershell
git clone <YOUR_REPO_URL> fantasy-app
cd fantasy-app
```

**Option B — Manual:**

- Create a folder, e.g. `fantasy-app`
- Save `app.py` and `requirements.txt` into that folder

### 2) Place your values CSV

Name the file **exactly**:

```
auction-values-half-ppr.csv
```

Place it in the **same folder** as `app.py`. The app **auto-loads** this file at startup.

### 3) Create & activate a virtual environment

```powershell
cd C:\Users\evant\fantasy-app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4) Install dependencies

```powershell
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, you can run:

```powershell
pip install pandas pulp streamlit openpyxl
```

### 5) Run the app

```powershell
python -m streamlit run app.py
```

Open your browser to `http://localhost:8501` if it doesn’t pop automatically.

---

## 🗂️ Folder layout

```
fantasy-app/
├─ app.py
├─ requirements.txt
└─ auction-values-half-ppr.csv   <-- auto-loaded at startup
```

---

## 🧠 How to use (in-draft flow)

1. **Starter & Bench Configuration (left sidebar)**

   - Set **QB / RB / WR / TE / K / FLEX** starter slots for your league
   - Set **Bench spots (non-DST)** for how many bench players you want the app to fill
   - Set **Reserve** dollars for DST + Bench; bench spend > reserve will **reduce starters’ working budget** automatically
   - Optionally limit **Max bench QBs**

2. **Add Purchases Live**

   - Use **Search player** (type-ahead), enter **Price paid ($)**, set **Starter?** → **Add / Update**
   - **Starter = On** → player is locked into the starting optimization at the price you paid
   - **Starter = Off** → player is treated as a **bench** or **DST** purchase and spends from **Reserve**

3. **Optimize**

   - Click **⚡ Optimize** to compute the best possible starters under the remaining budget
   - Bench is **not points-optimized**; it’s filled from Reserve using cheap RB/WR/TE first, then QB, then K (respecting your Max bench QBs)

4. **Review & Download**
   - See **Optimized Starters** (points + cost), **DST** (from Reserve), and **Bench**
   - Optionally view **Value Alternatives** per position (ranked by Points per $, with a points floor you control)
   - Click **Download roster CSV** to save a snapshot

---

## ⚙️ Settings (left sidebar)

- **Total budget** — auction budget (default 200)
- **Reserve for DST + Bench** — holdback used for DST and bench purchases
- **Max bench QBs** — cap the number of QBs on your bench
- **Starter slots (no DST)** — set exact counts:
  - QB, RB, WR, TE, K, FLEX (FLEX counts are RB/WR/TE only)
- **Bench spots (non-DST)** — how many bench players to auto-fill
- **Projection & Cost Blends** — control how projections/costs are combined:
  - `Consensus Proj` + `DS Proj` → **Points**
  - `AuctionMarketValue` + `DS AuctionValue` → **Cost**
- **Alternatives** — toggle, min projected points filter, and how many per position

> All **costs** are rounded **up** to whole dollars (auctions don’t do 50¢).  
> Projections are treated as **Half-PPR** based on your CSV columns.

---

## 🧾 CSV columns (expected)

Your `auction-values-half-ppr.csv` should include (typical sources already match this):

- **Player** (string)
- **Team** (string)
- **Fantasy Position** (QB/RB/WR/TE/K/DST)
- **Consensus Proj** (number) — optional but recommended
- **DS Proj** (number) — optional but recommended
- **AuctionMarketValue** (number or string with `$`) — optional but recommended
- **DS AuctionValue** (number or string with `$`) — optional but recommended

> The app requires at least one of `Consensus Proj` / `DS Proj` (for Points) **and** at least one of `AuctionMarketValue` / `DS AuctionValue` (for Cost).

---

## 🛠️ Troubleshooting

**“`streamlit` is not recognized” (Windows)**  
Use the `-m` form to ensure you’re invoking it from the active venv:

```powershell
python -m streamlit run app.py
```

**Dependencies fail to install**  
Upgrade pip and try again:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**App says it can’t find the CSV**

- Ensure the file is named **`auction-values-half-ppr.csv`**
- Ensure it’s in the **same folder** as `app.py`
- Restart the app after placing the file

**Solver errors (PuLP)**

- Reinstall PuLP: `pip install --upgrade pulp`
- PuLP uses CBC by default; most environments work out of the box.

---

## 📃 Requirements

`requirements.txt`:

```
pandas>=2.0.0
pulp>=2.7.0
streamlit>=1.34.0
openpyxl>=3.1.2
```

---

## 🍏 macOS / Linux notes

Use bash/zsh equivalents:

```bash
cd ~/fantasy-app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app.py
```

Everything else works the same.
