import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.io as pio

# Add current directory to path to resolve imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR
from src.graph import run_workflow

# Setup page config
st.set_page_config(
    page_title="Autonomous Data Science Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Premium Title styling */
    .app-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1A5276 0%, #117A65 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Card styling */
    .insight-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #1A5276;
    }
    
    .recommendation-card {
        border-left-color: #117A65;
    }
    
    .risk-card {
        border-left-color: #C0392B;
    }
    
    .action-card {
        border-left-color: #E67E22;
    }
    
    .card-title {
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 12px;
        color: #2C3E50;
    }
    
    /* Terminal Console logs */
    .terminal-container {
        background-color: #1E1E1E;
        color: #00FF00;
        font-family: 'Courier New', Courier, monospace;
        padding: 15px;
        border-radius: 8px;
        height: 250px;
        overflow-y: scroll;
        margin-bottom: 20px;
    }
    .terminal-line {
        margin-bottom: 4px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Initialize Session State
# ----------------------------------------------------
if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = None

if "is_running" not in st.session_state:
    st.session_state["is_running"] = False

# Load API Keys into Environment if entered in sidebar
st.sidebar.markdown("<h2 style='text-align: center; color: #1A5276;'>⚙️ Configuration</h2>", unsafe_allow_html=True)

# Provider Select
llm_provider = st.sidebar.selectbox(
    "LLM Provider", 
    ["Gemini", "OpenAI"], 
    index=0 if os.getenv("DEFAULT_LLM_PROVIDER", "gemini") == "gemini" else 1
)
os.environ["DEFAULT_LLM_PROVIDER"] = llm_provider.lower()

# Dynamic key inputs
if llm_provider == "Gemini":
    gemini_key = st.sidebar.text_input(
        "Gemini API Key", 
        value=os.getenv("GEMINI_API_KEY", ""), 
        type="password",
        help="Get your key from Google AI Studio."
    )
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key.strip().strip('"').strip("'")
else:
    openai_key = st.sidebar.text_input(
        "OpenAI API Key", 
        value=os.getenv("OPENAI_API_KEY", ""), 
        type="password"
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key.strip().strip('"').strip("'")

# Page Selection
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation", 
    [
        "Home", 
        "Upload Dataset", 
        "EDA Dashboard", 
        "Visualizations", 
        "Model Results", 
        "AI Insights", 
        "Download Report"
    ]
)

# Sidebar helper info
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Autonomous DS Team**
    * • Data Cleaner Agent
    * • EDA Agent
    * • Visualization Agent
    * • ML & SHAP Agent
    * • Business Insight Agent
    * • PDF Report Agent
    """
)

if st.sidebar.button("Reset Session State", type="secondary"):
    st.session_state["agent_state"] = None
    st.session_state["is_running"] = False
    st.rerun()


# ----------------------------------------------------
# Auxiliary Helper: Generate Synthetic Datasets
# ----------------------------------------------------
def create_synthetic_dataset(type_name: str) -> str:
    """Creates a sample CSV file in the output folder for instant user testing."""
    np.random.seed(42)
    n_samples = 150
    
    if type_name == "Classification (Churn Predictor)":
        data = {
            "CustomerID": range(1001, 1001 + n_samples),
            "Age": np.random.randint(18, 70, size=n_samples),
            "Tenure_Months": np.random.randint(1, 72, size=n_samples),
            "Monthly_Charge": np.round(np.random.uniform(20.0, 120.0, size=n_samples), 2),
            "Support_Calls": np.random.poisson(lam=1.5, size=n_samples),
            "Contract_Type": np.random.choice(["Month-to-Month", "One Year", "Two Year"], size=n_samples, p=[0.5, 0.3, 0.2]),
            "Paperless_Billing": np.random.choice(["Yes", "No"], size=n_samples),
            "Total_Spend": np.nan, # Introduce some missing values to clean
            "Churn": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        }
        df = pd.DataFrame(data)
        # Calculate Total Spend with missing values
        df["Total_Spend"] = df["Tenure_Months"] * df["Monthly_Charge"]
        df.loc[df.sample(frac=0.1).index, "Total_Spend"] = np.nan
        # Add duplicates for testing
        df = pd.concat([df, df.iloc[:3]], ignore_index=True)
        
        path = os.path.join(OUTPUT_DIR, "synthetic_churn_data.csv")
        df.to_csv(path, index=False)
        return path
        
    else: # Regression (House Price)
        data = {
            "HouseID": range(1, 1 + n_samples),
            "Square_Footage": np.random.randint(800, 4000, size=n_samples),
            "Bedrooms": np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
            "Bathrooms": np.random.choice([1, 1.5, 2, 2.5, 3], size=n_samples),
            "Neighborhood": np.random.choice(["Downtown", "Suburbs", "Rural"], size=n_samples, p=[0.4, 0.4, 0.2]),
            "Year_Built": np.random.randint(1950, 2024, size=n_samples),
            "Has_Pool": np.random.choice(["Yes", "No"], size=n_samples, p=[0.15, 0.85]),
            "House_Price": 0 # to compute below
        }
        df = pd.DataFrame(data)
        # Pricing logic
        df["House_Price"] = (
            df["Square_Footage"] * 150 +
            df["Bedrooms"] * 10000 +
            df["Bathrooms"] * 15000 +
            (df["Neighborhood"] == "Downtown") * 50000 +
            (df["Year_Built"] - 1950) * 500 +
            (df["Has_Pool"] == "Yes") * 30000 +
            np.random.normal(0, 15000, size=n_samples)
        ).astype(int)
        
        # Add a couple of outliers and missing values
        df.loc[5, "Square_Footage"] = 12000 # Outlier square footage
        df.loc[12, "Neighborhood"] = np.nan # Missing neighborhood
        
        path = os.path.join(OUTPUT_DIR, "synthetic_housing_data.csv")
        df.to_csv(path, index=False)
        return path


# ----------------------------------------------------
# Routing & Rendering Pages
# ----------------------------------------------------

if page == "Home":
    st.markdown("<h1 class='app-title'>🤖 Autonomous Data Science Agent</h1>", unsafe_allow_html=True)
    st.markdown("### A Collaborative Multi-Agent System managed by LangGraph")
    
    st.write(
        "Welcome to the next generation of automated analytics. "
        "The Autonomous Data Science Agent brings together a team of specialized AI agents "
        "to execute a complete end-to-end data science pipeline. Simply upload a CSV file and watch the team work in real-time."
    )
    
    st.markdown("---")
    
    st.markdown("### Meet the Team:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            * 🧼 **Data Cleaner Agent**: Inspects dataset quality. Programmatically profiles missing values, duplicate records, outliers, and type conflicts, then generates a custom cleaning strategy and executes it with Pandas.
            * 📊 **Exploratory Data Analysis (EDA) Agent**: Performs descriptive statistics, skewness indexing, and correlation analysis. Synthesizes findings using LLMs to offer feature engineering recommendations.
            * 📈 **Visualization Agent**: Automatically checks variables and constructs premium, interactive Plotly charts (histograms, box plots, scatterplots, heatmaps, bar charts) tailored to your dataset.
            """
        )
    with col2:
        st.markdown(
            """
            * ⚙️ **Machine Learning Agent**: Automatically detects whether the task is Classification, Regression, or Clustering. Handles preprocessing scaling and one-hot encoding, evaluates multiple baseline algorithms (Logistic/Linear Regression, Random Forests, XGBoost, KMeans), trains the champion model, and generates global and local SHAP explanations.
            * 💡 **Business Insights Agent**: Interprets the numerical and statistical outputs to build a narrative business case: executive summaries, findings, risk factors, and strategic next steps.
            * 📄 **Report Generation Agent**: Packages the entire run (text, tables, and charts) into a clean, stylized, page-numbered PDF report.
            """
        )
        
    st.markdown("---")
    st.markdown("### Getting Started:")
    st.write(
        "1. Enter your **LLM API Key** in the sidebar (Gemini is recommended for fast execution).\n"
        "2. Navigate to the **Upload Dataset** page.\n"
        "3. Upload a CSV file or load one of our synthetic datasets.\n"
        "4. Set the Target Column (or choose Clustering) and click **Run AI Team**."
    )

elif page == "Upload Dataset":
    st.markdown("<h1 class='app-title'>📂 Upload & Configure Dataset</h1>", unsafe_allow_html=True)
    
    # Check if API Key is configured
    active_provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    active_key = os.getenv("GEMINI_API_KEY") if active_provider == "gemini" else os.getenv("OPENAI_API_KEY")
    
    if not active_key:
        st.warning(f"⚠️ No API Key set for {active_provider.upper()}. Please enter your key in the sidebar before proceeding.")
        
    # File options: upload or sample
    file_source = st.radio("Select Data Source", ["Upload Custom CSV", "Load Synthetic Test Dataset"])
    
    csv_path = None
    df_preview = None
    
    if file_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            # Save uploaded file temporarily in the output folder
            temp_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            csv_path = temp_path
            df_preview = pd.read_csv(csv_path)
    else:
        sample_choice = st.selectbox("Select Test Dataset Type", [
            "Classification (Churn Predictor)", 
            "Regression (House Price Predictor)"
        ])
        if st.button("Load Sample Dataset", type="primary"):
            csv_path = create_synthetic_dataset(sample_choice)
            df_preview = pd.read_csv(csv_path)
            st.success(f"Loaded {sample_choice} sample dataset successfully!")
            
    # Show preview and configuration
    if df_preview is not None and csv_path is not None:
        st.markdown("### Dataset Preview (First 5 Rows)")
        st.dataframe(df_preview.head(5))
        
        # Configure Target Variable
        st.markdown("---")
        st.markdown("### Pipeline Configuration")
        
        columns_list = ["None (Clustering)"] + df_preview.columns.tolist()
        # Heuristically guess target (e.g. Churn, House_Price)
        default_idx = 0
        for name in ["churn", "price", "house_price", "target", "label"]:
            for idx, col in enumerate(columns_list):
                if name in col.lower():
                    default_idx = idx
                    break
            if default_idx > 0:
                break
                
        target_column = st.selectbox(
            "Select Target Column (Selecting 'None (Clustering)' triggers unsupervised KMeans clustering)",
            options=columns_list,
            index=default_idx
        )
        
        st.write(f"**Selected Target:** {target_column}")
        
        # Run Button
        if st.button("🚀 Run AI Data Science Team", type="primary", disabled=not active_key):
            st.session_state["is_running"] = True
            
            # Placeholders for progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            console_log = st.empty()
            
            # Setup logging observer to output logs to Streamlit
            # We can capture state['logs'] as they update!
            st.info("Running LangGraph workflow...")
            
            # Create a simple background executing function or synchronous executor
            with st.spinner("Agents are collaborating..."):
                try:
                    # Invoke workflow
                    final_state = run_workflow(csv_path, target_column)
                    
                    if final_state.get("error"):
                        st.error(f"Execution failed: {final_state['error']}")
                    else:
                        st.success("🎉 Pipeline executed successfully!")
                        
                    st.session_state["agent_state"] = final_state
                except Exception as ex:
                    st.error(f"Critical workflow crash: {ex}")
                finally:
                    st.session_state["is_running"] = False
                    st.rerun()

    # If state already exists, show active workflow execution log console
    if st.session_state["agent_state"]:
        state = st.session_state["agent_state"]
        st.markdown("---")
        st.markdown("### Active Run Execution Logs")
        
        log_html = "<div class='terminal-container'>"
        for line in state.get("logs", []):
            log_html += f"<div class='terminal-line'>&gt; {line}</div>"
        log_html += "</div>"
        
        st.markdown(log_html, unsafe_allow_html=True)
        
        if state.get("error"):
            st.error(f"Pipeline exited with error: {state['error']}")
        else:
            st.info("💡 Review the individual dashboard tabs in the sidebar navigation to view EDA, Visualizations, ML metrics, and download the report.")

else:
    # ----------------------------------------------------
    # Guard Rails: If pipeline hasn't run yet
    # ----------------------------------------------------
    if not st.session_state["agent_state"]:
        st.warning("⚠️ No active analysis results found. Please go to the 'Upload Dataset' page to upload a CSV file and run the agent team first.")
    else:
        state = st.session_state["agent_state"]
        
        if page == "EDA Dashboard":
            st.markdown("<h1 class='app-title'>📊 Exploratory Data Analysis (EDA)</h1>", unsafe_allow_html=True)
            
            eda = state.get("eda_results")
            if not eda:
                st.info("EDA results are not available.")
            else:
                # LLM Dataset Summary
                st.markdown(
                    f"<div class='insight-card'>"
                    f"<div class='card-title'>Executive Data Overview</div>"
                    f"{eda['llm_analysis']['dataset_overview']}"
                    f"</div>", 
                    unsafe_allow_html=True
                )
                
                # Metadata Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Dataset Shape", f"{eda['shape'][0]} rows x {eda['shape'][1]} cols")
                col2.metric("Numerical Features", f"{len(eda['numeric_cols'])}")
                col3.metric("Categorical Features", f"{len(eda['categorical_cols'])}")
                
                st.markdown("---")
                
                # Column Summaries Tabs
                tab1, tab2, tab3 = st.tabs(["Numerical Features", "Categorical Features", "Missing Values & dtypes"])
                
                with tab1:
                    if len(eda.get("numeric_summary", {})) > 0:
                        st.markdown("### Descriptive Statistics")
                        num_df = pd.DataFrame(eda["numeric_summary"]).T
                        st.dataframe(num_df.style.format("{:.2f}", na_rep="N/A"))
                    else:
                        st.write("No numeric columns in the dataset.")
                        
                with tab2:
                    if len(eda.get("categorical_summary", {})) > 0:
                        st.markdown("### Top Values and Cardinality")
                        for col, summary in eda["categorical_summary"].items():
                            col_col1, col_col2 = st.columns([1, 2])
                            with col_col1:
                                st.write(f"**Column:** `{col}`")
                                st.write(f"Unique Values: {summary['cardinality']}")
                            with col_col2:
                                counts_df = pd.DataFrame(
                                    list(summary["top_values"].items()), 
                                    columns=["Value", "Count"]
                                )
                                st.dataframe(counts_df, hide_index=True)
                            st.markdown("---")
                    else:
                        st.write("No categorical columns in the dataset.")
                        
                with tab3:
                    meta_df = pd.DataFrame({
                    "Column": list(eda["dtypes"].keys()),
                    "Data Type": list(eda["dtypes"].values()),
                    "Missing Count": [
                        eda["missing_values"].get(col, 0)
                        for col in eda["dtypes"].keys()
                    ]
                })

                meta_df["Missing Pct (%)"] = (
                    meta_df["Missing Count"] / eda["shape"][0] * 100
                ).round(2)

                st.dataframe(meta_df, use_container_width=True)
                    
                st.markdown("---")
                st.markdown("### LLM Key Findings & Observations")
                for idx, finding in enumerate(eda["llm_analysis"]["key_findings"]):
                    st.info(f"🔍 **Key Finding {idx+1}:** {finding}")

        elif page == "Visualizations":
            st.markdown("<h1 class='app-title'>📈 Interactive Visualizations</h1>", unsafe_allow_html=True)
            st.write("The Visualization Agent automatically selected and rendered the following interactive Plotly charts:")
            
            visuals = state.get("visualizations", [])
            if not visuals:
                st.info("No charts generated.")
            else:
                for viz in visuals:
                    name_clean = viz["name"].replace("_", " ").title()
                    st.markdown(f"### {name_clean}")
                    # Render plotly figure from serialized dictionary
                    try:
                        st.plotly_chart(viz["fig_dict"], use_container_width=True)
                    except Exception as ex:
                        st.error(f"Failed to render chart {viz['name']}: {ex}")
                    st.markdown("---")

        elif page == "Model Results":
            st.markdown("<h1 class='app-title'>⚙️ Automated ML Training & Explainability</h1>", unsafe_allow_html=True)
            
            ml = state.get("ml_results")
            if not ml:
                st.info("Machine Learning results not available.")
            else:
                st.subheader(f"Problem Type: {ml['problem_type'].upper()}")
                
                # Comparisons Table
                if ml.get("comparison"):
                    st.markdown("### Baseline Algorithms Performance Matrix")
                    comp_df = pd.DataFrame(ml["comparison"])
                    st.dataframe(comp_df.style.highlight_max(
                        subset=[c for c in comp_df.columns if "accuracy" in c or "f1" in c or "r2" in c],
                        color="#D4EFDF"
                    ).highlight_min(
                        subset=[c for c in comp_df.columns if "mse" in c or "rmse" in c],
                        color="#D4EFDF"
                    ))
                    
                st.markdown("---")
                
                # Best Model Card
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(
                        f"<div class='insight-card recommendation-card'>"
                        f"<div class='card-title'>🏆 Champion Model</div>"
                        f"The algorithm selection module chose <b>{ml['best_model_name']}</b> "
                        f"as the best model based on validation test splits."
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown("### Champion Model Metrics")
                    for m, val in ml.get("metrics", {}).items():
                        m_clean = m.replace("_", " ").title()
                        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                        st.metric(m_clean, val_str)
                
                # Explainable AI (SHAP) Tab
                st.markdown("---")
                st.markdown("### 🔍 Explainable AI (SHAP Summary Plots)")
                
                if ml.get("has_shap"):
                    tab1, tab2 = st.tabs(["Global Feature Impact", "Local Row Explanation"])
                    with tab1:
                        st.write(
                            "The SHAP summary plot ranks features by their average impact on predictions. "
                            "Features at the top represent the strongest predictive signals."
                        )
                        # Load generated global shap plot
                        summary_plot_file = os.path.join(PLOTS_DIR, "shap_summary.png")
                        if os.path.exists(summary_plot_file):
                            st.image(summary_plot_file, caption="Global SHAP Summary Plot", use_container_width=True)
                        else:
                            st.warning("SHAP global summary plot image file not found.")
                            
                    with tab2:
                        st.write(
                            "Local explanation decomposes the model's output for a single customer or data row. "
                            "It demonstrates how features push the prediction above or below the average expected value."
                        )
                        local_plot_file = os.path.join(PLOTS_DIR, "shap_local_explanation.png")
                        if os.path.exists(local_plot_file):
                            st.image(local_plot_file, caption="Local Instance Breakdown", use_container_width=True)
                        else:
                            st.warning("Local SHAP explanation plot image file not found.")
                else:
                    st.info("SHAP explainability plots are not available for this model type.")

        elif page == "AI Insights":
            st.markdown("<h1 class='app-title'>💡 AI Strategic Insights & Recommendations</h1>", unsafe_allow_html=True)
            
            insights = state.get("ai_insights")
            if not insights:
                st.info("AI Insights not available.")
            else:
                # Executive Summary
                st.markdown(
                    f"<div class='insight-card'>"
                    f"<div class='card-title'>Executive Summary</div>"
                    f"{insights['executive_summary']}"
                    f"</div>",
                    unsafe_allow_html=True
                )
                 
                col1, col2 = st.columns(2)
                
                with col1:
                    # Key Findings
                    st.markdown("<div class='insight-card recommendation-card'>", unsafe_allow_html=True)
                    st.markdown("<div class='card-title'>📊 Key Findings</div>", unsafe_allow_html=True)
                    for f in insights.get("key_findings", []):
                        st.markdown(f"• {f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Business Insights
                    st.markdown("<div class='insight-card recommendation-card'>", unsafe_allow_html=True)
                    st.markdown("<div class='card-title'>💡 Business Impact</div>", unsafe_allow_html=True)
                    for i in insights.get("business_insights", []):
                        st.markdown(f"• {i}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with col2:
                    # Risks & limitations
                    st.markdown("<div class='insight-card risk-card'>", unsafe_allow_html=True)
                    st.markdown("<div class='card-title'>⚠️ Operational Risks & Constraints</div>", unsafe_allow_html=True)
                    for r in insights.get("risks_and_limitations", []):
                        st.markdown(f"• {r}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Recommendations
                    st.markdown("<div class='insight-card recommendation-card'>", unsafe_allow_html=True)
                    st.markdown("<div class='card-title'>🎯 Strategic Actions</div>", unsafe_allow_html=True)
                    for rec in insights.get("recommendations", []):
                        st.markdown(f"• {rec}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                # Next Actions
                st.markdown("<div class='insight-card action-card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>🚀 Recommended Next Actions</div>", unsafe_allow_html=True)
                for a in insights.get("next_actions", []):
                    st.markdown(f"• {a}")
                st.markdown("</div>", unsafe_allow_html=True)

        elif page == "Download Report":
            st.markdown("<h1 class='app-title'>📄 Download PDF Report</h1>", unsafe_allow_html=True)
            
            report_path = state.get("report_path")
            if not report_path or not os.path.exists(report_path):
                st.error("Report PDF file does not exist or failed to generate.")
            else:
                st.success("🎉 PDF Report successfully compiled!")
                st.write(f"The Report Generation Agent has successfully compiled all statistics, visualizations, model parameters, and AI explanations into a premium, printable PDF.")
                
                # Read file bytes for download button
                with open(report_path, "rb") as f:
                    pdf_bytes = f.read()
                    
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=os.path.basename(report_path),
                    mime="application/pdf",
                    type="primary"
                )
                
                st.markdown("---")
                st.markdown("### Report Contents Preview")
                st.write(
                    "- **Page 1:** Title Page and Dataset Metadata Summary\n"
                    "- **Page 2:** Executive Business Summary & Data Quality profiling\n"
                    "- **Page 3:** Exploratory Data Analysis (EDA) and Correlation Heatmaps\n"
                    "- **Page 4:** Machine Learning Baseline Comparison & Best Model selection\n"
                    "- **Page 5:** Explainable AI (SHAP summary & local explanation plots)\n"
                    "- **Page 6:** AI Strategic Business Insights, Risks, & Action roadmap"
                )
