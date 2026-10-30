import streamlit as st
import pandas as pd
from pathlib import Path
from modules.data_processing import validate_dataset, prepare_dataset, data_quality_summary, FEATURES
from modules.exploratory_analysis import service_volume_chart, breach_distribution_chart, queue_chart
from modules.prediction import load_model, load_metrics, predict_one
from modules.dashboard import dashboard_kpis

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Student Service Request Monitoring System", page_icon="🎓", layout="wide")

@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def load_default_data():
    return pd.read_csv(ROOT / "student_service_requests.csv")

def get_data():
    uploaded = st.sidebar.file_uploader("Upload student_service_requests.csv", type=["csv"])
    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            ok, message = validate_dataset(incoming)
            if not ok:
                st.error(message)
                return None
            st.sidebar.success("Uploaded dataset validated.")
            return prepare_dataset(incoming)
        except Exception as exc:
            st.error(f"Unable to read CSV: {exc}")
            return None
    default = load_default_data()
    ok, message = validate_dataset(default)
    if not ok:
        st.error(message)
        return None
    return prepare_dataset(default)

st.sidebar.title("🎓 Student Services")
page = st.sidebar.selectbox("Navigation", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
st.sidebar.caption("Section B | Systems Analysis and Design")
df = get_data()

if page == "Home":
    st.title("Student Service Request Monitoring System")
    st.subheader("SLA Breach Prediction and Request Workload Monitoring")
    st.write("This application uses the Decision Tree model from the midterm examination to help student services staff identify requests that may need attention.")
    if df is not None:
        k = dashboard_kpis(df)
        a,b,c = st.columns(3)
        a.metric("Total Requests", f"{k['total_requests']:,}")
        b.metric("Completed and Labeled", f"{k['completed_labeled']:,}")
        c.metric("Open Requests", f"{k['open_requests']:,}")
    st.info("Predictions are advisory only. Authorized staff must review request details and make the final decision.")
    st.markdown("**Available modules:** Dataset validation, exploratory analysis, SLA prediction, decision-support dashboard, and system design.")

elif page == "Dataset":
    st.title("Dataset Upload and Validation")
    if df is not None:
        st.success("Dataset schema is valid.")
        st.write(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
        st.dataframe(df.head(20), use_container_width=True)
        st.subheader("Data quality summary")
        st.dataframe(data_quality_summary(df), use_container_width=True)
        d1,d2,d3 = st.columns(3)
        d1.metric("Missing cells", int(df.isna().sum().sum()))
        d2.metric("Duplicate Request IDs", int(df["Request_ID"].duplicated().sum()))
        d3.metric("Invalid submission dates", int(df["Submission_Date"].isna().sum()))
        st.subheader("Column data types")
        st.dataframe(pd.DataFrame({"Column":df.columns,"Type":df.dtypes.astype(str).values}), use_container_width=True)

elif page == "Data Analysis":
    st.title("Exploratory Data Analysis")
    if df is not None:
        st.subheader("Descriptive statistics")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
        c1,c2 = st.columns(2)
        c1.plotly_chart(service_volume_chart(df), use_container_width=True)
        c2.plotly_chart(breach_distribution_chart(df), use_container_width=True)
        st.plotly_chart(queue_chart(df), use_container_width=True)
        st.caption("SLA class distribution is calculated only for completed requests with known processing days and SLA days, matching the midterm labeling procedure.")

elif page == "Prediction":
    st.title("Predict SLA Breach Risk")
    st.write("Enter information available for a request at submission time. Do not enter actual processing outcomes.")
    if df is not None:
        model = get_model()
        col1,col2 = st.columns(2)
        with col1:
            service = st.selectbox("Service Type", sorted(df["Service_Type"].dropna().unique()))
            requester = st.selectbox("Requester Type", sorted(df["Requester_Type"].dropna().unique()))
            channel = st.selectbox("Requested Channel", sorted(df["Requested_Channel"].dropna().unique()))
            priority = st.selectbox("Priority", sorted(df["Priority"].dropna().unique()))
            office = st.selectbox("Assigned Office", sorted(df["Assigned_Office"].dropna().unique()))
        with col2:
            queue = st.number_input("Queue Length", min_value=0, value=5, step=1)
            docs = st.selectbox("Required Documents Complete", sorted(df["Required_Documents_Complete"].dropna().unique()))
            sla = st.number_input("SLA Days", min_value=1, value=int(df["SLA_Days"].median()), step=1)
            weekday = st.selectbox("Submission Weekday", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            qcat = "Low" if queue <= 5 else ("Medium" if queue <= 15 else "High")
            st.write(f"**Derived queue category:** {qcat}")
        if st.button("Predict SLA Breach", type="primary"):
            values = {
                "Service_Type":service, "Requester_Type":requester, "Requested_Channel":channel,
                "Priority":priority, "Assigned_Office":office, "Queue_Length":queue,
                "Required_Documents_Complete":docs, "SLA_Days":sla,
                "Submission_Weekday":weekday, "Queue_Category":qcat
            }
            label, probability = predict_one(model, values)
            if label == 1:
                st.error("Model result: Potential SLA Breach — staff review recommended.")
            else:
                st.success("Model result: No SLA Breach predicted.")
            if probability is not None:
                st.metric("Estimated SLA-breach probability", f"{probability:.1%}")
            st.caption("The prediction is an estimate from the trained Decision Tree, not a guarantee. Staff must verify documents, queue conditions, and SLA deadline.")
        st.subheader("Midterm model evaluation")
        m = load_metrics()
        a,b,c,d = st.columns(4)
        a.metric("Accuracy", f"{m['accuracy']:.1%}")
        b.metric("Precision", f"{m['precision']:.1%}")
        c.metric("Recall", f"{m['recall']:.1%}")
        d.metric("F1-score", f"{m['f1_score']:.1%}")
        st.write("Confusion matrix (rows = actual; columns = predicted; order: No breach, Breach)")
        st.dataframe(pd.DataFrame(m["confusion_matrix"], index=["Actual No Breach","Actual Breach"], columns=["Predicted No Breach","Predicted Breach"]))

elif page == "Dashboard":
    st.title("Decision-Support Dashboard")
    if df is not None:
        k = dashboard_kpis(df)
        a,b,c,d = st.columns(4)
        a.metric("Total Requests", f"{k['total_requests']:,}")
        b.metric("Completed / Labeled", f"{k['completed_labeled']:,}")
        c.metric("Observed SLA Breaches", f"{k['breached']:,}")
        d.metric("Observed Breach Rate", f"{k['breach_rate']:.1f}%")
        st.subheader("Queue Status")
        queue_counts = df["Queue_Category"].value_counts().reindex(["Low","Medium","High"]).fillna(0).astype(int)
        st.bar_chart(queue_counts)
        st.subheader("Requests for staff review")
        review = df[(df["Status"].astype(str).str.lower() != "completed")].copy()
        review["Review Reason"] = review.apply(
            lambda r: "Incomplete documents" if str(r["Required_Documents_Complete"]).lower() != "yes"
            else ("High queue" if pd.notna(r["Queue_Length"]) and r["Queue_Length"] > 15
            else ("Urgent priority" if str(r["Priority"]).lower() in ["urgent","high"] else "Open request — routine monitoring")), axis=1
        )
        cols = ["Request_ID","Service_Type","Priority","Assigned_Office","Queue_Length","SLA_Days","Status","Review Reason"]
        st.dataframe(review[cols].head(100), use_container_width=True)
        st.info("Review queue is a staff-facing monitoring list based on current request status and intake conditions. It is not an automatic rejection or final risk decision.")

elif page == "System Design":
    st.title("Systems Analysis and Design Integration")
    st.subheader("Actors and Use Case Diagram")
    image_path = ROOT / "assets" / "use_case_diagram.png"
    st.image(str(image_path), caption="Student Service Request Monitoring System — supplied UML Use Case Diagram", use_container_width=True)
    st.subheader("Actors")
    st.markdown("- **Student Services Administrator:** views service volume and SLA performance, reviews delay indicators, and monitors queue status.\n- **Service Staff:** monitors queue status and manages student requests.")
    st.subheader("Functional Requirements")
    requirements = [
        "The system shall validate uploaded CSV files against the required dataset schema.",
        "The system shall display request volume by service type and the distribution of SLA-breach classes.",
        "The system shall generate SLA-breach predictions using the saved Decision Tree pipeline and the original midterm features.",
        "The system shall display request workload, observed SLA performance, and requests requiring staff review.",
        "The system shall display the use case diagram and explain the authorized human review workflow."
    ]
    for i,r in enumerate(requirements,1): st.write(f"{i}. {r}")
    st.subheader("Non-Functional Requirements")
    st.markdown("1. **Usability:** navigation and charts should be understandable to student services personnel.\n2. **Reliability:** invalid files and missing columns should produce clear validation messages.\n3. **Security and privacy:** only synthetic or authorized records should be used; sensitive student data must not be published.")
    st.subheader("Analytical Decision Workflow")
    st.markdown("1. Staff receive or open a student service request.\n2. The system validates intake fields and derives submission weekday and queue category.\n3. The saved model estimates the likelihood of an SLA breach.\n4. The dashboard flags requests for authorized staff review.\n5. Staff verify documents, queue conditions, and the SLA target, then record the final action outside the model.")
    st.warning("Model limitations: the Decision Tree was trained only on completed requests with known processing and SLA days. Its performance is moderate; predictions should guide review, not replace staff judgment.")
