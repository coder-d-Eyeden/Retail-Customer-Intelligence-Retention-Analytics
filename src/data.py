from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "online-retail" / "Online Retail.xlsx"
PROC = ROOT / "data" / "processed"
MODEL = ROOT / "models"


def load_raw() -> pd.DataFrame:
    """Load and clean the UCI Online Retail dataset.

    Steps:
        - Parse dates, cast InvoiceNo to string
        - Drop exact duplicates
        - Remove rows with missing CustomerID
        - Remove zero/negative prices and quantities
        - Exclude cancelled invoices (prefix 'C')
        - Compute Revenue = Quantity * UnitPrice
    """
    if not RAW.exists():
        raise FileNotFoundError(
            f"Missing {RAW}. Download Online Retail.xlsx from UCI and place it in data/raw/online-retail/."
        )

    df = pd.read_excel(RAW)
    df.columns = [c.strip() for c in df.columns]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    df = df.drop_duplicates().copy()
    df = df[df["CustomerID"].notna()]
    df = df[df["UnitPrice"] > 0]
    df = df[df["Quantity"] > 0]
    df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


def save_clean(df: pd.DataFrame) -> Path:
    PROC.mkdir(parents=True, exist_ok=True)
    path = PROC / "clean_transactions.csv"
    df.to_csv(path, index=False)
    return path
