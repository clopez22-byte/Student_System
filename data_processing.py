import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Submission_Date",
    "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
    "Required_Documents_Complete", "Processing_Days", "SLA_Days",
    "Status", "Reopened"
]
FEATURES = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Queue_Length", "Required_Documents_Complete",
    "SLA_Days", "Submission_Weekday", "Queue_Category"
]

def validate_dataset(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    if df.empty:
        return False, "The uploaded CSV contains no records."
    return True, "Dataset schema is valid."

def prepare_dataset(df):
    data = df.copy()
    data["Submission_Date"] = pd.to_datetime(data["Submission_Date"], errors="coerce")
    for col in ["Processing_Days", "SLA_Days", "Queue_Length"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["Submission_Weekday"] = data["Submission_Date"].dt.day_name()
    data["Queue_Category"] = pd.cut(
        data["Queue_Length"], bins=[-1, 5, 15, float("inf")],
        labels=["Low", "Medium", "High"]
    )
    eligible = (
        data["Status"].eq("Completed")
        & data["Processing_Days"].notna()
        & data["SLA_Days"].notna()
        & data["Submission_Date"].notna()
        & data["Queue_Length"].notna()
    )
    data["SLA_Breached"] = np.nan
    data.loc[eligible, "SLA_Breached"] = (
        data.loc[eligible, "Processing_Days"] > data.loc[eligible, "SLA_Days"]
    ).astype(int)
    return data

def data_quality_summary(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values
    })
