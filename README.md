# 🤖 Autonomous Data Science Agent

An end-to-end, production-grade autonomous data science platform. Powered by **LangGraph** and **LangChain**, a team of specialized AI agents collaborates to clean, explore, visualize, model, explain, and write business strategies for any uploaded CSV dataset.

---

## Key Features

1. **🧼 Data Cleaning Agent**: Profiles missing values, duplicate records, outliers, and data types, proposing and applying a Pandas preprocessing strategy.
2. **📊 Exploratory Data Analysis (EDA) Agent**: Calculates data statistics, correlations, skewness, and uses LLMs to interpret statistical properties and suggest feature engineering.
3. **📈 Visualization Agent**: Dynamically generates interactive **Plotly** figures (Heatmap, Histograms, Boxplots, Scatterplots, Bar charts) and saves PNG copies for PDF reporting.
4. **⚙️ Machine Learning Agent**: Detects the problem type (Classification, Regression, Clustering), fits and evaluates candidate models (Linear/Logistic Regression, Random Forest, XGBoost, KMeans), selects the best, and computes **SHAP** global and local explanations.
5. **💡 Business Insight Agent**: Integrates quantitative metrics from previous steps to compose executive reports (Executive Summary, Key Findings, Business Insights, Risks, Recommendations, Next Steps).
6. **📄 Report Generation Agent**: Generates a page-numbered, header/footer decorated **PDF report** via **ReportLab**, packing the narrative and images into a single download.

---

## Architecture Flow

```
START 
  ↓
[Data Cleaning Agent]  # Imputes nulls, handles outliers, drops duplicates
  ↓
[EDA Agent]            # Computes descriptive statistics & skewness
  ↓
[Visualization Agent]  # Creates Plotly charts & saves static PNGs
  ↓
[ML Agent (with SHAP)] # Preprocesses, trains models, scores, run SHAP plots
  ↓
[Insight Agent]        # LLM writes executive and strategic reports
  ↓
[Report Agent]         # ReportLab compiles layout, text, tables, and images to PDF
  ↓
END
```

---

## Tech Stack

- **Frontend**: Streamlit
- **Backend & Logic**: Python, LangGraph, LangChain, Pydantic
- **Data & ML**: Pandas, NumPy, Scikit-Learn, XGBoost, SHAP, Optuna
- **Visualizations**: Plotly, Matplotlib, Seaborn
- **LLM Engine**: Gemini (via `google-generativeai`) or OpenAI (via `openai`)
- **Reporting**: ReportLab
- **Containerization**: Docker

---

## Installation & Setup

### 1. Clone & Initialize Directory
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy the example environment file:
```bash
cp .env.example .env
```
Open `.env` and configure your API keys:
- **Gemini Key** (Recommended): Set `GEMINI_API_KEY`
- **OpenAI Key**: Set `OPENAI_API_KEY`

*Note: You can also enter these keys dynamically directly in the Streamlit Sidebar during application runtime.*

---

## Running the Application

### Local Run
Start the Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Docker Run
To compile and build the docker container:
```bash
docker build -t autonomous-ds-agent .
```
Run the container:
```bash
docker run -p 8501:8501 --env-file .env autonomous-ds-agent
```
Open your browser at `http://localhost:8501`.

---

## Instant Testing (No File Upload Required)
To verify the application instantly:
1. Navigate to the **Upload Dataset** page in the Streamlit sidebar.
2. Select **Load Synthetic Test Dataset**.
3. Choose either **Classification (Churn Predictor)** or **Regression (House Price Predictor)**.
4. Click **Load Sample Dataset**, select the target column, and click **Run AI Data Science Team**.
