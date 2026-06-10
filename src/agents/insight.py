import logging
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from src.utils.llm import get_llm
from src.state import AgentState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InsightAgent")

# Pydantic schema for structured business analysis
class BusinessInsights(BaseModel):
    executive_summary: str = Field(description="A concise executive summary of the project goals, findings, and predictive success.")
    key_findings: List[str] = Field(description="Key findings discovered during EDA and model analysis.")
    business_insights: List[str] = Field(description="Actionable business insights. Translate correlation and feature importance into business impact.")
    risks_and_limitations: List[str] = Field(description="Operational risks, data quality constraints, or model deployment limitations.")
    recommendations: List[str] = Field(description="Strategic recommendations for business leaders.")
    next_actions: List[str] = Field(description="Immediate next actions (e.g. gather more features, pilot model, retrain).")

class InsightAgent:
    """
    Agent responsible for compiling statistical results, model metrics, and features importances,
    and generating high-level business narratives and strategic recommendations.
    """
    
    def generate_insights(self, state: AgentState) -> AgentState:
        """Main entry point for the agent execution in the LangGraph workflow."""
        state["logs"].append("Started Insight Generation Agent.")
        
        eda_results = state.get("eda_results")
        ml_results = state.get("ml_results")
        target_column = state.get("target_column")
        
        # Check if previous steps completed
        if not eda_results or not ml_results:
            error_msg = "EDA results or ML results are missing. Cannot generate insights."
            logger.error(error_msg)
            state["error"] = error_msg
            state["logs"].append(f"Error: {error_msg}")
            return state
            
        try:
            # Extract key stats
            dataset_shape = eda_results.get("shape", (0, 0))
            num_cols = len(eda_results.get("numeric_cols", []))
            cat_cols = len(eda_results.get("categorical_cols", []))
            
            # Extract cleaning report summary
            cleaning_report = state.get("cleaning_report", {})
            cleaning_actions = cleaning_report.get("actions_taken", [])
            
            # Extract ML summary
            problem_type = ml_results.get("problem_type", "N/A")
            best_model_name = ml_results.get("best_model_name", "N/A")
            metrics = ml_results.get("metrics", {})
            feature_importance = ml_results.get("feature_importance", [])[:5] # Top 5 features
            
            # Compile summary payload for the LLM
            summary_payload = {
                "dataset": {
                    "original_rows": cleaning_report.get("original_shape", [dataset_shape[0]])[0],
                    "final_rows": dataset_shape[0],
                    "numeric_columns_count": num_cols,
                    "categorical_columns_count": cat_cols,
                    "target_column": target_column,
                    "cleaning_actions_applied": cleaning_actions
                },
                "eda_insights": eda_results.get("llm_analysis", {}),
                "machine_learning": {
                    "problem_type": problem_type,
                    "selected_model": best_model_name,
                    "evaluation_metrics": metrics,
                    "top_predictive_features": feature_importance
                }
            }
            
            # LLM Prompt Setup
            llm = get_llm()
            parser = PydanticOutputParser(pydantic_object=BusinessInsights)
            
            prompt_template = PromptTemplate(
                template="""You are a Senior Solution Architect and Principal AI Business Analyst.
You are given a comprehensive data profiling, EDA analysis, and machine learning model evaluation report.
Your goal is to translate these technical results into a professional, clear, and actionable business-friendly report.

Technical Summary Data:
{summary_payload_json}

Format Instructions:
{format_instructions}

Please write a business-ready analysis:
- Executive Summary: Clear overview. Avoid raw code or variable references where possible; speak in business terms (e.g. 'Customer Tenure' instead of 'cust_tenure_months').
- Key Findings: Extract the most important statistics and model coefficients/importances.
- Business Insights: Bridge features and outcomes. What do these relationships mean for business operations, revenue, or costs?
- Risks and Limitations: Highlight metrics (e.g. is accuracy low? are there missing variables? is sample size small?).
- Recommendations: Strategic advice for business decision-makers.
- Next Actions: Actions the data science team or business unit should execute next.

Provide the insights strictly as a JSON object matching the format instructions.
""",
                input_variables=["summary_payload_json"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            # Invoke LLM
            formatted_prompt = prompt_template.format(summary_payload_json=json.dumps(summary_payload, indent=2))
            response = llm.invoke(formatted_prompt)
            
            # Parse output
            insights_parsed = parser.parse(response.content)
            
            state["ai_insights"] = insights_parsed.model_dump()
            state["logs"].append(
                f"Insight Generation complete. Generated Executive Summary ({len(insights_parsed.executive_summary)} chars) "
                f"and {len(insights_parsed.recommendations)} strategic recommendations."
            )
            
        except Exception as e:
            error_msg = f"Failed in Insight Agent: {str(e)}"
            logger.exception(error_msg)
            state["error"] = error_msg
            state["logs"].append(f"Error: {error_msg}")
            
        return state
