import pandas as pd

def dashboard_kpis(df):
    total = len(df)
    labeled = df[df["SLA_Breached"].notna()]
    breached = int((labeled["SLA_Breached"] == 1).sum())
    breach_rate = (breached / len(labeled) * 100) if len(labeled) else 0
    open_requests = int((df["Status"].astype(str).str.lower() != "completed").sum())
    return {
        "total_requests": total,
        "completed_labeled": len(labeled),
        "breached": breached,
        "breach_rate": breach_rate,
        "open_requests": open_requests
    }
