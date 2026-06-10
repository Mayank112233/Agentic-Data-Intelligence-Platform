import os
import logging
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from src.utils.llm import get_llm
from src.state import AgentState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EDAAgent")

# Pydantic schema for LLM interpretation
class ColumnAnalysis(BaseModel):
    column_name: str = Field(description="Name of the column")
    distribution_type: str = Field(description="Distribution type: e.g., Normal, Skewed-Right, Skewed-Left, Uniform, Categorical, Binary")
    interpretation: str = Field(description="Brief statistical interpretation or observation of this column")

class EdaLlmAnalysis(BaseModel):
    dataset_overview: str = Field(description="High-level natural language summary of the dataset structure and contents")
    key_findings: List[str] = Field(description="Key patterns, high correlation relationships, or insights discovered")
    feature_engineering_suggestions: List[str] = Field(description="Specific suggestions for creating new features, transformations, or binning based on the statistics")
    column_interpretations: List[ColumnAnalysis] = Field(description="Detailed analysis per key column")


class EDAAgent:
    """
    Agent responsible for performing Exploratory Data Analysis, generating descriptive statistics, 
    calculating correlations, and interpreting the overall data landscape using an LLM.
    """
    
    def analyze(self, state: AgentState) -> AgentState:
        """Main entry point for the agent execution in the LangGraph workflow."""
        state["logs"].append("Started EDA Agent.")
        csv_path = state.get("cleaned_csv_path") or state.get("csv_path")
        target_column = state.get("target_column")
        
        if not csv_path or not os.path.exists(csv_path):
            error_msg = f"CSV path '{csv_path}' does not exist or was not provided."
            logger.error(error_msg)
            state["error"] = error_msg
            state["logs"].append(f"Error: {error_msg}")
            return state
            
        try:
            # Load dataset
            df = pd.read_csv(csv_path)
            
            # Programmatic EDA computations
            shape = df.shape
            dtypes = df.dtypes.astype(str).to_dict()
            missing_values = df.isnull().sum().to_dict()
            unique_values = df.nunique().to_dict()
            
            # Separate numeric and categorical
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            
            # Numeric stats summary
            numeric_summary = {}
            if len(numeric_cols) > 0:
                desc = df[numeric_cols].describe()
                # Calculate skewness
                skewness = df[numeric_cols].skew().to_dict()
                for col in numeric_cols:
                    col_stats = desc[col].to_dict()
                    col_stats["skewness"] = skewness.get(col, 0.0)
                    numeric_summary[col] = col_stats
            
            # Categorical stats summary
            categorical_summary = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts(dropna=False).head(5).to_dict()
                categorical_summary[col] = {
                    "top_values": value_counts,
                    "cardinality": unique_values[col]
                }
                
            # Correlation Matrix
            correlations = {}
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr(method='pearson')
                correlations = corr_matrix.to_dict()
                
            # Target Column Information
            target_info = {}
            if target_column and target_column in df.columns:
                target_info = {
                    "name": target_column,
                    "type": str(df[target_column].dtype),
                    "unique_count": unique_values[target_column]
                }
                if pd.api.types.is_numeric_dtype(df[target_column]):
                    target_info["mean"] = float(df[target_column].mean())
                    target_info["std"] = float(df[target_column].std())
                else:
                    target_info["top_categories"] = df[target_column].value_counts().head(3).to_dict()
            
            # Prepare summarized profile for LLM ingestion (keep size reasonable)
            profile_for_llm = {
                "shape": shape,
                "numeric_columns_count": len(numeric_cols),
                "categorical_columns_count": len(categorical_cols),
                "target_column": target_info,
                "numeric_columns_summary": {k: {
                    "mean": v.get("mean"),
                    "std": v.get("std"),
                    "min": v.get("min"),
                    "max": v.get("max"),
                    "skewness": v.get("skewness")
                } for k, v in list(numeric_summary.items())[:15]}, # Limit columns to prevent token bloat
                "categorical_columns_summary": {k: v for k, v in list(categorical_summary.items())[:15]},
                "top_correlations": self._extract_top_correlations(df, numeric_cols)
            }
            
            # LLM Analysis
            llm = get_llm()
            parser = PydanticOutputParser(pydantic_object=EdaLlmAnalysis)
            
            prompt_template = PromptTemplate(
                template="""You are a Senior Data Scientist and EDA expert.
You have been given a statistical summary of a dataset. Your task is to interpret this statistical information and output:
1. A natural language executive-level overview of the dataset.
2. Key statistical patterns or insights (e.g. highly correlated features, class imbalances, skewed distributions).
3. Advanced feature engineering suggestions (e.g. binning, scaling, log transforms, ratio feature creation).
4. Individual interpretations of key variables.

Dataset statistical profile:
{profile_json}

Format Instructions:
{format_instructions}

Provide the analysis strictly as a JSON object matching the format instructions.
""",
                input_variables=["profile_json"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            # Invoke LLM
            formatted_prompt = prompt_template.format(profile_json=json.dumps(profile_for_llm, indent=2))
            response = llm.invoke(formatted_prompt)
            
            # Parse output
            llm_analysis_parsed = parser.parse(response.content)
            
            # Assemble full EDA results dict
            eda_results = {
                "shape": shape,
                "dtypes": dtypes,
                "missing_values": missing_values,
                "unique_values": unique_values,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "numeric_summary": numeric_summary,
                "categorical_summary": categorical_summary,
                "correlations": correlations,
                "llm_analysis": llm_analysis_parsed.model_dump()
            }
            
            state["eda_results"] = eda_results
            state["logs"].append(
                f"EDA Analysis completed. Shape: {shape[0]} rows, {shape[1]} columns. "
                f"LLM generated {len(llm_analysis_parsed.key_findings)} key findings."
            )
            
        except Exception as e:
            error_msg = f"Failed in EDA Agent: {str(e)}"
            logger.exception(error_msg)
            state["error"] = error_msg
            state["logs"].append(f"Error: {error_msg}")
            
        return state

    @staticmethod
    def _extract_top_correlations(df: pd.DataFrame, numeric_cols: List[str], threshold: float = 0.4) -> List[Dict[str, Any]]:
        """Utility to extract high correlation pairs for LLM context."""
        if len(numeric_cols) < 2:
            return []
            
        corr_matrix = df[numeric_cols].corr().abs()
        pairs = []
        
        # Avoid duplicates and self correlations
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]
                val = float(df[col1].corr(df[col2]))
                if abs(val) >= threshold:
                    pairs.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(val, 4)
                    })
                    
        # Sort by strength of correlation
        pairs = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)
        return pairs[:10] # Return top 10 strong correlations
