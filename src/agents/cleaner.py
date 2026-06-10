import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from src.utils.llm import get_llm
from src.state import AgentState
from config.settings import CLEANED_DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataCleaningAgent")

# Pydantic Schemas for Structured Output
class ColumnCleaningStrategy(BaseModel):
    column_name: str = Field(description="Name of the column")
    missing_value_action: str = Field(
        description="Action for missing values. Options: 'impute_mean', 'impute_median', 'impute_mode', 'impute_constant', 'drop', 'none'"
    )
    missing_value_constant: Optional[str] = Field(
        default=None, 
        description="Constant value if missing_value_action is 'impute_constant'"
    )
    outlier_action: str = Field(
        description="Action for outliers. Options: 'cap_iqr', 'remove', 'none'"
    )
    type_conversion: str = Field(
        description="Data type conversion. Options: 'int', 'float', 'str', 'category', 'datetime', 'none'"
    )

class CleaningPlan(BaseModel):
    drop_duplicates: bool = Field(description="Whether to drop duplicate rows")
    columns: List[ColumnCleaningStrategy] = Field(description="Cleaning actions for each column")


class DataCleaningAgent:
    """
    Agent responsible for analyzing dataset anomalies (missing values, duplicates, outliers, incorrect types)
    and applying automated cleaning strategies.
    """
    
    @staticmethod
    def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Programmatically profile the dataframe to extract metadata for LLM analysis."""
        profile = {
            "total_rows": int(df.shape[0]),
            "total_cols": int(df.shape[1]),
            "duplicate_rows": int(df.duplicated().sum()),
            "columns": []
        }
        
        for col in df.columns:
            # Handle mixed types gracefully
            col_series = df[col]
            missing_count = int(col_series.isnull().sum())
            missing_pct = float((missing_count / df.shape[0]) * 100)
            unique_count = int(col_series.nunique())
            
            col_info = {
                "name": col,
                "type": str(col_series.dtype),
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "unique_count": unique_count,
                "sample_values": [str(x) for x in col_series.dropna().head(3).tolist()]
            }
            
            # Outlier detection (IQR method) for numeric columns
            if pd.api.types.is_numeric_dtype(col_series) and unique_count > 5:
                q1 = col_series.quantile(0.25)
                q3 = col_series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = int(((col_series < lower_bound) | (col_series > upper_bound)).sum())
                col_info["outlier_count"] = outliers
            else:
                col_info["outlier_count"] = 0
                
            profile["columns"].append(col_info)
            
        return profile

    def clean(self, state: AgentState) -> AgentState:
        """Main entry point for the agent execution in the LangGraph workflow."""
        state["logs"].append("Started Data Cleaning Agent.")
        csv_path = state.get("csv_path")
        
        if not csv_path or not os.path.exists(csv_path):
            error_msg = f"CSV path '{csv_path}' does not exist or was not provided."
            logger.error(error_msg)
            state["error"] = error_msg
            state["logs"].append(f"Error: {error_msg}")
            return state
            
        try:
            # Load dataset
            df = pd.read_csv(csv_path)
            original_shape = df.shape
            
            # Profile dataset
            profile = self.profile_data(df)
            state["logs"].append(
                f"Dataset profiled: {original_shape[0]} rows, {original_shape[1]} columns. "
                f"Found {profile['duplicate_rows']} duplicate rows."
            )
            
            # Get cleaning plan from LLM
            llm = get_llm()
            parser = PydanticOutputParser(pydantic_object=CleaningPlan)
            
            prompt_template = PromptTemplate(
                template="""You are a Data Quality and Preprocessing expert.
You have been given a dataset profile with missing values, duplicate rows, outliers, and incorrect data types.
Your goal is to design a robust cleaning plan for the dataset.

Dataset Profile:
{profile_json}

Format Instructions:
{format_instructions}

Design the cleaning plan carefully:
- If duplicate_rows is > 0, set drop_duplicates to true.
- For numerical columns with missing values: select 'impute_median' or 'impute_mean'.
- For categorical columns with missing values: select 'impute_mode' or 'impute_constant'.
- For outliers: select 'cap_iqr' (capping at 1.5 * IQR) or 'none'. Cap is generally safer than deleting rows.
- For incorrect types (e.g. numerical fields read as objects), specify type_conversion. Otherwise select 'none'.

Provide the plan strictly as a JSON object matching the format instructions.
""",
                input_variables=["profile_json"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            # Invoke LLM
            import json
            formatted_prompt = prompt_template.format(profile_json=json.dumps(profile, indent=2))
            response = llm.invoke(formatted_prompt)
            
            # Parse response
            cleaning_plan = parser.parse(response.content)
            state["logs"].append("LLM generated cleaning plan successfully.")
            
            # Apply cleaning plan using Pandas
            df_cleaned = df.copy()
            actions_taken = []
            
            # Drop duplicates
            if cleaning_plan.drop_duplicates:
                before_count = len(df_cleaned)
                df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
                actions_taken.append(f"Dropped {before_count - len(df_cleaned)} duplicate rows.")
                
            # Apply column transformations
            for col_strategy in cleaning_plan.columns:
                col = col_strategy.column_name
                if col not in df_cleaned.columns:
                    continue
                    
                # 1. Type Conversion
                if col_strategy.type_conversion != "none":
                    try:
                        if col_strategy.type_conversion == "int":
                            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0).astype(int)
                        elif col_strategy.type_conversion == "float":
                            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                        elif col_strategy.type_conversion == "str":
                            df_cleaned[col] = df_cleaned[col].astype(str)
                        elif col_strategy.type_conversion == "category":
                            df_cleaned[col] = df_cleaned[col].astype("category")
                        elif col_strategy.type_conversion == "datetime":
                            df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                        actions_taken.append(f"Converted column '{col}' to type '{col_strategy.type_conversion}'.")
                    except Exception as e:
                        logger.warning(f"Failed type conversion for column {col}: {e}")
                        
                # 2. Impute missing values
                null_mask = df_cleaned[col].isnull()
                null_count = null_mask.sum()
                if null_count > 0 and col_strategy.missing_value_action != "none":
                    if col_strategy.missing_value_action == "impute_mean":
                        val = df_cleaned[col].mean()
                        df_cleaned[col] = df_cleaned[col].fillna(val)
                        actions_taken.append(f"Imputed {null_count} missing values in '{col}' using mean ({val:.2f}).")
                    elif col_strategy.missing_value_action == "impute_median":
                        val = df_cleaned[col].median()
                        df_cleaned[col] = df_cleaned[col].fillna(val)
                        actions_taken.append(f"Imputed {null_count} missing values in '{col}' using median ({val:.2f}).")
                    elif col_strategy.missing_value_action == "impute_mode":
                        if not df_cleaned[col].mode().empty:
                            val = df_cleaned[col].mode()[0]
                            df_cleaned[col] = df_cleaned[col].fillna(val)
                            actions_taken.append(f"Imputed {null_count} missing values in '{col}' using mode ({val}).")
                    elif col_strategy.missing_value_action == "impute_constant":
                        val = col_strategy.missing_value_constant or "Unknown"
                        df_cleaned[col] = df_cleaned[col].fillna(val)
                        actions_taken.append(f"Imputed {null_count} missing values in '{col}' using constant '{val}'.")
                    elif col_strategy.missing_value_action == "drop":
                        df_cleaned = df_cleaned.dropna(subset=[col]).reset_index(drop=True)
                        actions_taken.append(f"Dropped {null_count} rows containing missing values in '{col}'.")
                        
                # 3. Cap or Remove Outliers
                if col_strategy.outlier_action != "none" and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    q1 = df_cleaned[col].quantile(0.25)
                    q3 = df_cleaned[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                    outliers_count = outliers_mask.sum()
                    
                    if outliers_count > 0:
                        if col_strategy.outlier_action == "cap_iqr":
                            df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
                            actions_taken.append(f"Capped {outliers_count} outliers in '{col}' within bounds [{lower_bound:.2f}, {upper_bound:.2f}].")
                        elif col_strategy.outlier_action == "remove":
                            df_cleaned = df_cleaned[~outliers_mask].reset_index(drop=True)
                            actions_taken.append(f"Removed {outliers_count} outlier rows in '{col}'.")
            
            # Save cleaned dataset
            filename = os.path.basename(csv_path)
            cleaned_filename = f"cleaned_{filename}"
            cleaned_path = os.path.join(CLEANED_DATA_DIR, cleaned_filename)
            df_cleaned.to_csv(cleaned_path, index=False)
            
            report = {
                "original_shape": original_shape,
                "cleaned_shape": df_cleaned.shape,
                "actions_taken": actions_taken,
                "cleaning_plan": cleaning_plan.model_dump()
            }
            
            state["cleaned_csv_path"] = cleaned_path
            state["cleaning_report"] = report
            state["logs"].append(
                f"Data Cleaning complete. Final shape: {df_cleaned.shape}. Cleaned CSV saved at {cleaned_path}"
            )
            
        except Exception as e:
            error_msg = f"Failed in Data Cleaning Agent: {str(e)}"
            logger.exception(error_msg)
            state["error"] = error_msg
            state["logs"].append(f"Error: {error_msg}")
            
        return state
