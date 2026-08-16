import io
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================================
# PROJECT PATH
# ==========================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from HealthPulse_AI_project.components.data_preprocessing import (
    DataPreprocessing,
)
from HealthPulse_AI_project.components.prediction import (
    PredictionPipeline,
)
from ecg_adapters import build_adapter_registry

# ==========================================================
# CONFIGURATION
# ==========================================================

ECG_DATA_PATH = "data/raw/ptbxl"
NORMALIZATION_PATH = "artifacts/preprocessing/normalization_params.npz"

LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

DEFAULT_RECORD = "records100/00000/00001_lr"

# Format that should be pre-selected in the UI. Must be a key present
# in the adapter registry.
DEFAULT_FORMAT = "wfdb"

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="HealthPulse-AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    .mi-badge {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }
    .mi-positive {background-color: #3b0d0d; color: #ff8080; border: 1px solid #7a1f1f;}
    .mi-negative {background-color: #0d3b1a; color: #7dffa0; border: 1px solid #1f7a35;}
    .format-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .format-ready {background-color: #0d3b1a; color: #7dffa0;}
    .format-soon {background-color: #3b330d; color: #ffdd80;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# CACHED RESOURCES
#
# NOTE: The adapter registry is the ONLY inference entry point. There is
# intentionally no direct DataPreprocessing/get_preprocessor() path here
# anymore — every prediction, single or batch, goes through
# registry["wfdb"].load() -> StandardECG -> PredictionPipeline.predict().
# ==========================================================


@st.cache_resource(show_spinner=False)
def get_predictor():
    return PredictionPipeline()


@st.cache_resource(show_spinner=False)
def get_adapter_registry():
    return build_adapter_registry(DataPreprocessing, NORMALIZATION_PATH)


def run_prediction_via_adapter(standard_ecg):
    """Run the CNN against a StandardECG produced by any adapter.
    This is the single place that calls PredictionPipeline.predict()."""
    predictor = get_predictor()
    result = predictor.predict(standard_ecg.signal)
    return result


def run_prediction_for_wfdb_path(record_path: str):
    """Run a WFDB prediction from a PTB-XL-style relative record path
    (e.g. 'records100/00000/00001_lr'), used by Batch Prediction.

    Splits the path into a directory (resolved against ECG_DATA_PATH)
    and a record base name, then routes through the same wfdb adapter
    as the upload flow — no separate preprocessing path.
    """
    rel_dir = os.path.dirname(record_path)
    record_base = os.path.basename(record_path)
    record_dir = os.path.join(ECG_DATA_PATH, rel_dir) if rel_dir else ECG_DATA_PATH

    registry = get_adapter_registry()
    wfdb_adapter = registry["wfdb"]

    standard_ecg = wfdb_adapter.load(
        record_dir=record_dir,
        record_base=record_base,
    )
    result = run_prediction_via_adapter(standard_ecg)

    return standard_ecg, result


def prepare_uploaded_wfdb_record(uploaded_files):
    """Validate and stage an uploaded PTB-XL WFDB record (.hea + matching .dat)
    into a temp directory. Does NOT preprocess or predict — that's the
    adapter's job. Returns (tmp_dir, record_base).

    Raises ValueError with a user-facing message if the upload doesn't
    form a valid WFDB record pair.
    """
    if not uploaded_files:
        raise ValueError(
            "Please upload the .hea and .dat files for one ECG record."
        )

    hea_files = [f for f in uploaded_files if f.name.lower().endswith(".hea")]
    dat_files = [f for f in uploaded_files if f.name.lower().endswith(".dat")]

    if len(hea_files) == 0:
        raise ValueError(
            "No .hea file found. WFDB records need a .hea header file."
        )
    if len(hea_files) > 1:
        raise ValueError(
            "Please upload files for only one record at a time "
            "(one .hea file plus its matching .dat file)."
        )
    if len(dat_files) == 0:
        raise ValueError(
            "No matching .dat file found. WFDB records need both a .hea "
            "header file and a .dat signal file with the same base name."
        )

    hea_file = hea_files[0]
    record_base = Path(hea_file.name).stem

    matching_dat = next(
        (f for f in dat_files if Path(f.name).stem == record_base), None
    )
    if matching_dat is None:
        raise ValueError(
            f"The uploaded .dat file doesn't match the record name "
            f"'{record_base}'. Make sure both files belong to the same "
            f"record (same base filename, e.g. {record_base}.hea / "
            f"{record_base}.dat)."
        )

    tmp_dir = tempfile.mkdtemp(prefix="healthpulse_upload_")
    with open(os.path.join(tmp_dir, hea_file.name), "wb") as out:
        out.write(hea_file.getbuffer())
    with open(os.path.join(tmp_dir, matching_dat.name), "wb") as out:
        out.write(matching_dat.getbuffer())

    return tmp_dir, record_base


# ==========================================================
# SESSION STATE
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_ecg" not in st.session_state:
    st.session_state.last_ecg = None

if "last_record" not in st.session_state:
    st.session_state.last_record = None

if "last_format" not in st.session_state:
    st.session_state.last_format = None


def add_to_history(record_path, result, source_format="wfdb"):
    st.session_state.history.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "record": record_path,
            "format": source_format,
            "prediction": result["prediction"],
            "probability": result["probability"],
            "threshold": result["threshold"],
            "class": result.get("class", ""),
        }
    )


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title("❤️ HealthPulse-AI")
st.sidebar.caption("AI-assisted MI detection from ECG")

page = st.sidebar.radio(
    "Navigate",
    ["Single Prediction", "Batch Prediction", "History", "About"],
)

st.sidebar.divider()
st.sidebar.markdown("**Model**")
st.sidebar.text("1D CNN · PTB-XL")

if st.sidebar.button("Clear cached model/adapters"):
    get_predictor.clear()
    get_adapter_registry.clear()
    st.sidebar.success("Cache cleared. Will reload on next prediction.")


def probability_gauge(probability: float, threshold: float):
    color = "#ff4d4d" if probability >= threshold else "#4dff88"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#1f2937"},
                    {"range": [threshold * 100, 100], "color": "#3b0d0d"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def plot_ecg(processed_ecg: np.ndarray):
    """Plot ECG leads. Supports (timesteps, leads) or (leads, timesteps)."""
    arr = np.asarray(processed_ecg)

    if arr.ndim != 2:
        st.info("ECG array is not 2D; skipping waveform plot.")
        return

    if arr.shape[0] < arr.shape[1]:
        arr = arr.T

    n_leads = min(arr.shape[1], 12)
    lead_labels = LEAD_NAMES[:n_leads] if n_leads <= 12 else [
        f"Lead {i+1}" for i in range(n_leads)
    ]

    fig = go.Figure()
    offset_step = 4
    for i in range(n_leads):
        fig.add_trace(
            go.Scatter(
                y=arr[:, i] + i * offset_step,
                mode="lines",
                name=lead_labels[i],
                line=dict(width=1.2),
            )
        )
    fig.update_layout(
        height=520,
        showlegend=True,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(showticklabels=False, title=""),
        xaxis=dict(title="Sample"),
        title="Preprocessed ECG (stacked leads)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_result_block(record_path, result, processed_ecg):
    probability = result["probability"]
    threshold = result["threshold"]
    prediction = result["prediction"]

    st.divider()
    st.subheader("Prediction Result")

    badge_class = "mi-positive" if prediction == "MI" else "mi-negative"
    badge_text = "⚠️ MI Detected" if prediction == "MI" else "✅ Non-MI"
    st.markdown(
        f'<span class="mi-badge {badge_class}">{badge_text}</span>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(
            probability_gauge(probability, threshold),
            use_container_width=True,
        )
    with col2:
        st.metric("MI Probability", f"{probability:.2%}")
        st.metric("Decision Threshold", f"{threshold:.2f}")
        st.metric("Predicted Class", str(result.get("class", "N/A")))

    with st.expander("ECG Waveform", expanded=False):
        plot_ecg(processed_ecg)

    with st.expander("Technical Details"):
        st.write(f"ECG record: `{record_path}`")
        st.write(f"Processed shape: `{np.asarray(processed_ecg).shape}`")
        st.write(f"Predicted class: `{result.get('class', '')}`")
        st.write(f"Probability: `{probability:.4f}`")
        st.write(f"Threshold: `{threshold:.2f}`")

    report = (
        f"HealthPulse-AI Prediction Report\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Record: {record_path}\n"
        f"Prediction: {prediction}\n"
        f"Probability: {probability:.4f}\n"
        f"Threshold: {threshold:.2f}\n"
        f"Class: {result.get('class', '')}\n\n"
        "Research/educational prototype. Not a medical diagnosis.\n"
    )
    st.download_button(
        "Download Report (.txt)",
        data=report,
        file_name=f"healthpulse_report_{record_path.replace('/', '_')}.txt",
        mime="text/plain",
    )


# ==========================================================
# PAGE: SINGLE PREDICTION
# ==========================================================

if page == "Single Prediction":
    st.title("❤️ HealthPulse-AI")
    st.subheader("AI-assisted Myocardial Infarction Detection")
    st.write(
        "Upload an ECG record to generate an MI prediction using the "
        "trained 1D CNN model."
    )
    st.divider()

    registry = get_adapter_registry()

    st.subheader("ECG Input Format")
    format_options = list(registry.keys())
    default_index = (
        format_options.index(DEFAULT_FORMAT)
        if DEFAULT_FORMAT in format_options
        else 0
    )
    format_labels = {
        fmt: f"{registry[fmt].display_name}"
        + (" ✅" if registry[fmt].is_implemented else " (coming soon)")
        for fmt in format_options
    }
    selected_format = st.radio(
        "Source format",
        format_options,
        index=default_index,
        format_func=lambda f: format_labels[f],
        horizontal=True,
    )
    adapter = registry[selected_format]

    if adapter.help_text:
        st.caption(adapter.help_text)

    if not adapter.is_implemented:
        st.info(
            f"**{adapter.display_name}** support isn't implemented yet — "
            "this option is here to show where it plugs into the "
            "architecture. Select **WFDB ECG record** to run a "
            "prediction today."
        )

    st.divider()
    st.subheader("Upload ECG")

    if selected_format == "wfdb":
        st.write(
            "Upload your ECG's two record files — a header file and a "
            "signal file with the same name (e.g. `00001_lr.hea` and "
            "`00001_lr.dat`). These are typically stored together."
        )

        uploaded_files = st.file_uploader(
            "Upload ECG record (.hea + .dat)",
            type=["hea", "dat"],
            accept_multiple_files=True,
            help="Select both files for the same record in one go.",
        )

        with st.expander("Why is this format-specific?"):
            st.write(
                "In a real deployment, ECGs arrive from the hospital "
                "system in whatever format the ECG machine exports "
                "(XML, DICOM, sometimes only a PDF/image). Each format "
                "needs its own adapter to turn it into a standardized "
                "signal before the CNN can use it — a PDF in particular "
                "would need a separate waveform-extraction step first, "
                "since it isn't raw signal data. For this demo, the "
                f"{adapter.technical_name} format is the only one "
                "implemented; others are stubbed in the architecture, "
                "ready to fill in."
            )

        run_disabled = False
    else:
        st.file_uploader(
            f"Upload {adapter.display_name} (not yet supported)",
            disabled=True,
        )
        uploaded_files = None
        run_disabled = True

    if st.button("Run MI Prediction", type="primary", disabled=run_disabled):
        tmp_dir = None
        try:
            # Upload -> validate/stage only. No preprocessing here.
            tmp_dir, record_base = prepare_uploaded_wfdb_record(uploaded_files)

            with st.spinner("Preprocessing ECG and running CNN prediction..."):
                wfdb_adapter = registry["wfdb"]
                standard_ecg = wfdb_adapter.load(
                    record_dir=tmp_dir,
                    record_base=record_base,
                )
                result = run_prediction_via_adapter(standard_ecg)

            st.session_state.last_result = result
            st.session_state.last_ecg = standard_ecg.signal
            st.session_state.last_record = standard_ecg.source_id
            st.session_state.last_format = standard_ecg.source_format
            add_to_history(
                standard_ecg.source_id, result, standard_ecg.source_format
            )

        except (ValueError, NotImplementedError) as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())
        finally:
            if tmp_dir and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    if st.session_state.last_result is not None:
        render_result_block(
            st.session_state.last_record,
            st.session_state.last_result,
            st.session_state.last_ecg,
        )

# ==========================================================
# PAGE: BATCH PREDICTION
# ==========================================================

elif page == "Batch Prediction":
    st.title("📋 Batch Prediction")
    st.write(
        "Batch mode currently supports WFDB ECG record paths (same "
        "PTB-XL dataset layout the model was trained on)."
    )
    st.write(
        "Upload a CSV with a single column named `record` containing "
        "PTB-XL record paths, or paste them below (one per line)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    pasted = st.text_area(
        "Or paste record paths (one per line)",
        placeholder="records100/00000/00001_lr\nrecords100/00000/00002_lr",
        height=120,
    )

    records = []
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            col = "record" if "record" in df_in.columns else df_in.columns[0]
            records = df_in[col].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    elif pasted.strip():
        records = [r.strip() for r in pasted.splitlines() if r.strip()]

    if records:
        st.info(f"{len(records)} record(s) queued.")

    if st.button("Run Batch Prediction", type="primary", disabled=not records):
        rows = []
        progress = st.progress(0.0)
        status = st.empty()

        for i, rec in enumerate(records):
            status.text(f"Processing {rec} ({i + 1}/{len(records)})")
            try:
                # Same single entry point as Single Prediction: adapter -> StandardECG -> CNN.
                _, result = run_prediction_for_wfdb_path(rec)
                rows.append(
                    {
                        "record": rec,
                        "prediction": result["prediction"],
                        "probability": result["probability"],
                        "threshold": result["threshold"],
                        "class": result.get("class", ""),
                        "status": "ok",
                    }
                )
                add_to_history(rec, result, source_format="wfdb")
            except Exception as e:
                rows.append(
                    {
                        "record": rec,
                        "prediction": None,
                        "probability": None,
                        "threshold": None,
                        "class": None,
                        "status": f"error: {e}",
                    }
                )
            progress.progress((i + 1) / len(records))

        status.empty()
        results_df = pd.DataFrame(rows)
        st.subheader("Batch Results")
        st.dataframe(results_df, use_container_width=True)

        n_mi = (results_df["prediction"] == "MI").sum()
        n_ok = (results_df["status"] == "ok").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Processed", f"{n_ok}/{len(records)}")
        c2.metric("MI Detected", int(n_mi))
        c3.metric("Non-MI", int(n_ok - n_mi))

        csv_buf = io.StringIO()
        results_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download Results (.csv)",
            data=csv_buf.getvalue(),
            file_name=f"healthpulse_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ==========================================================
# PAGE: HISTORY
# ==========================================================

elif page == "History":
    st.title("🕓 Prediction History")

    if not st.session_state.history:
        st.info("No predictions yet this session. Run one from the Single or Batch Prediction pages.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", len(hist_df))
        c2.metric("MI Detected", int((hist_df["prediction"] == "MI").sum()))
        c3.metric(
            "Avg. Probability",
            f"{hist_df['probability'].mean():.2%}" if len(hist_df) else "N/A",
        )

        st.line_chart(hist_df["probability"])

        csv_buf = io.StringIO()
        hist_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download History (.csv)",
            data=csv_buf.getvalue(),
            file_name="healthpulse_history.csv",
            mime="text/csv",
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

# ==========================================================
# PAGE: ABOUT
# ==========================================================

else:
    st.title("ℹ️ About HealthPulse-AI")
    st.markdown(
        """
        **HealthPulse-AI** is a research/educational prototype that uses a
        1D convolutional neural network trained on the PTB-XL ECG dataset
        to flag potential signs of myocardial infarction (MI) from a
        12-lead ECG recording.

        **Realistic hospital workflow**

        ECG machine → digital ECG / ECG report → hospital system →
        HealthPulse-AI → convert to model input → 1D CNN → MI probability

        A hospital may hand us WFDB, XML, DICOM waveform data — or only a
        PDF/image, which the CNN can't consume directly and would need a
        separate waveform-extraction step before it could be used at all.

        **Architecture**

        The adapter registry is the single inference entry point. Every
        prediction — single upload or batch — goes:

        `Upload/path → ECG Adapter → StandardECG (1000, 12) → PredictionPipeline → 1D CNN → probability + threshold`

        The UI never calls `DataPreprocessing` directly and never sees
        WFDB-specific details — it only ever talks to adapters and
        `StandardECG`. Only WFDB is implemented today; XML and DICOM are
        registered as stubs so they can be filled in later without
        touching the UI or prediction code.

        **Pages**
        - *Single Prediction*: run one uploaded record and inspect the
          waveform, probability gauge, and technical details.
        - *Batch Prediction*: upload or paste multiple WFDB record paths
          and get an aggregated results table.
        - *History*: review and export predictions made this session.
        """
    )

# ==========================================================
# DISCLAIMER
# ==========================================================

st.divider()
st.caption(
    "Research/educational prototype. This system is not a medical "
    "diagnosis or a substitute for evaluation by a qualified healthcare "
    "professional."
)