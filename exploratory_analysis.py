import plotly.express as px
import pandas as pd

def service_volume_chart(df):
    counts = df["Service_Type"].value_counts().rename_axis("Service Type").reset_index(name="Requests")
    return px.bar(counts, x="Service Type", y="Requests", title="Request Volume by Service Type")

def breach_distribution_chart(df):
    labeled = df[df["SLA_Breached"].notna()].copy()
    labeled["SLA Result"] = labeled["SLA_Breached"].map({0:"No SLA Breach",1:"SLA Breach"})
    counts = labeled["SLA Result"].value_counts().rename_axis("SLA Result").reset_index(name="Requests")
    return px.pie(counts, names="SLA Result", values="Requests", title="SLA Breach Class Distribution")

def queue_chart(df):
    counts = df["Queue_Category"].value_counts().reindex(["Low","Medium","High"]).fillna(0).rename_axis("Queue Category").reset_index(name="Requests")
    return px.bar(counts, x="Queue Category", y="Requests", title="Requests by Queue Category")
