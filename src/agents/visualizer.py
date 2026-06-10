import os
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server/thread safety
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from src.state import AgentState
from config.settings import PLOTS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VisualizationAgent")

class VisualizationAgent:
    """
    Agent responsible for automatically generating relevant interactive Plotly charts
    and saving static PNG copies for PDF reporting.
    """
    
    def generate_charts(self, state: AgentState) -> AgentState:
        """Main entry point for the agent execution in the LangGraph workflow."""
        state["logs"].append("Started Visualization Agent.")
        csv_path = state.get("cleaned_csv_path") or state.get("csv_path")
        target_col = state.get("target_column")
        
        if not csv_path or not os.path.exists(csv_path):
            error_msg = f"CSV path '{csv_path}' does not exist or was not provided."
            logger.error(error_msg)
            state["error"] = error_msg
            state["logs"].append(f"Error: {error_msg}")
            return state
            
        try:
            # Load dataset
            df = pd.read_csv(csv_path)
            
            # Identify columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            
            # Limit listing to prevent memory issues
            numeric_cols = [c for c in numeric_cols if df[c].nunique() > 1]
            
            visualizations = []
            saved_paths = []
            
            # Clear previous plots
            for f in os.listdir(PLOTS_DIR):
                if f.endswith('.png'):
                    try:
                        os.remove(os.path.join(PLOTS_DIR, f))
                    except Exception:
                        pass

            # ----------------------------------------------------
            # 1. Heatmap (Correlation Matrix)
            # ----------------------------------------------------
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                
                # Plotly Fig
                fig_heatmap = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap"
                )
                fig_heatmap.update_layout(title_x=0.5, margin=dict(t=50, l=50, r=50, b=50))
                visualizations.append({
                    "name": "correlation_heatmap",
                    "type": "heatmap",
                    "fig_dict": fig_heatmap.to_dict()
                })
                
                # Save static
                self._save_static_heatmap(corr_matrix)
                saved_paths.append("correlation_heatmap.png")
                
            # ----------------------------------------------------
            # 2. Histogram (Distribution Analysis)
            # ----------------------------------------------------
            if len(numeric_cols) > 0:
                # Select top numeric column (highest skewness or target itself)
                col_to_plot = target_col if (target_col and target_col in numeric_cols) else numeric_cols[0]
                
                # Plotly Fig
                fig_hist = px.histogram(
                    df,
                    x=col_to_plot,
                    nbins=30,
                    marginal="box",
                    title=f"Distribution of {col_to_plot}",
                    color_discrete_sequence=["#2E86C1"]
                )
                fig_hist.update_layout(title_x=0.5)
                visualizations.append({
                    "name": f"distribution_{col_to_plot}",
                    "type": "histogram",
                    "fig_dict": fig_hist.to_dict()
                })
                
                # Save static
                self._save_static_histogram(df, col_to_plot)
                saved_paths.append(f"distribution_{col_to_plot}.png")

            # ----------------------------------------------------
            # 3. Boxplot (Outliers or Grouped Numeric)
            # ----------------------------------------------------
            if len(numeric_cols) > 0:
                num_col = numeric_cols[0]
                # If target is categorical, group by target
                group_col = None
                if target_col and target_col in categorical_cols and df[target_col].nunique() <= 10:
                    group_col = target_col
                elif len(categorical_cols) > 0:
                    # Choose a low cardinality category
                    for c in categorical_cols:
                        if 1 < df[c].nunique() <= 5:
                            group_col = c
                            break
                
                # Plotly Fig
                fig_box = px.box(
                    df,
                    y=num_col,
                    x=group_col,
                    color=group_col,
                    title=f"Boxplot of {num_col}" + (f" grouped by {group_col}" if group_col else ""),
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_box.update_layout(title_x=0.5)
                visualizations.append({
                    "name": f"boxplot_{num_col}",
                    "type": "boxplot",
                    "fig_dict": fig_box.to_dict()
                })
                
                # Save static
                self._save_static_boxplot(df, num_col, group_col)
                saved_paths.append(f"boxplot_{num_col}.png")

            # ----------------------------------------------------
            # 4. Scatterplot (Relationship between numeric variables)
            # ----------------------------------------------------
            if len(numeric_cols) >= 2:
                # Find strongest correlation (non-diagonal)
                corr_matrix = df[numeric_cols].corr().abs()
                for col in corr_matrix.columns:
                    corr_matrix.loc[col, col] = 0
                max_corr_idx = corr_matrix.stack().idxmax()
                x_col, y_col = max_corr_idx
                
                # If target is categorical, color by target
                color_col = target_col if (target_col and target_col in categorical_cols and df[target_col].nunique() <= 8) else None
                
                # Plotly Fig
                fig_scatter = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    trendline="ols" if not color_col else None, # Add trendline if simple
                    title=f"Scatterplot: {x_col} vs {y_col}",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_scatter.update_layout(title_x=0.5)
                visualizations.append({
                    "name": f"scatterplot_{x_col}_{y_col}",
                    "type": "scatterplot",
                    "fig_dict": fig_scatter.to_dict()
                })
                
                # Save static
                self._save_static_scatterplot(df, x_col, y_col, color_col)
                saved_paths.append(f"scatterplot_{x_col}_{y_col}.png")

            # ----------------------------------------------------
            # 5. Bar Chart (Categorical Frequencies)
            # ----------------------------------------------------
            # Find a suitable categorical column
            cat_col = target_col if (target_col and target_col in categorical_cols) else None
            if not cat_col and len(categorical_cols) > 0:
                # Select categorical column with low cardinality but > 1
                for c in categorical_cols:
                    if 1 < df[c].nunique() <= 20:
                        cat_col = c
                        break
                        
            if cat_col:
                counts = df[cat_col].value_counts().reset_index()
                counts.columns = [cat_col, "count"]
                
                # Plotly Fig
                fig_bar = px.bar(
                    counts,
                    x=cat_col,
                    y="count",
                    title=f"Frequency of {cat_col}",
                    color="count",
                    color_continuous_scale="Viridis"
                )
                fig_bar.update_layout(title_x=0.5)
                visualizations.append({
                    "name": f"bar_{cat_col}",
                    "type": "bar",
                    "fig_dict": fig_bar.to_dict()
                })
                
                # Save static
                self._save_static_bar(counts, cat_col)
                saved_paths.append(f"bar_{cat_col}.png")

            state["visualizations"] = visualizations
            state["logs"].append(
                f"Generated {len(visualizations)} interactive Plotly charts and saved static PNG images: {saved_paths}."
            )
            
        except Exception as e:
            error_msg = f"Failed in Visualization Agent: {str(e)}"
            logger.exception(error_msg)
            state["error"] = error_msg
            state["logs"].append(f"Error: {error_msg}")
            
        return state
        
    # Helper methods for saving static plots using Matplotlib/Seaborn
    # This acts as a robust fallback and generates highly clean output for PDFs.
    
    def _save_static_heatmap(self, corr_matrix: pd.DataFrame):
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", fmt=".2f", center=0, cbar=True, square=True)
        plt.title("Correlation Heatmap", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=200)
        plt.close()
        
    def _save_static_histogram(self, df: pd.DataFrame, col: str):
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True, color="#2E86C1")
        plt.title(f"Distribution of {col}", fontsize=14, pad=15)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"distribution_{col}.png"), dpi=200)
        plt.close()
        
    def _save_static_boxplot(self, df: pd.DataFrame, num_col: str, group_col: Optional[str]):
        plt.figure(figsize=(8, 5))
        if group_col:
            sns.boxplot(x=group_col, y=num_col, hue=group_col, data=df, palette="Set2", legend=False)
            plt.title(f"Boxplot of {num_col} Grouped by {group_col}", fontsize=14, pad=15)
            plt.xlabel(group_col)
        else:
            sns.boxplot(y=num_col, data=df, color="#A569BD")
            plt.title(f"Boxplot of {num_col}", fontsize=14, pad=15)
        plt.ylabel(num_col)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"boxplot_{num_col}.png"), dpi=200)
        plt.close()
        
    def _save_static_scatterplot(self, df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str]):
        plt.figure(figsize=(8, 5))
        if color_col:
            sns.scatterplot(x=x_col, y=y_col, hue=color_col, data=df, palette="deep")
        else:
            sns.scatterplot(x=x_col, y=y_col, data=df, color="#239B56")
            # Fit simple regression line
            sns.regplot(x=x_col, y=y_col, data=df, scatter=False, color="red")
        plt.title(f"Relationship: {x_col} vs {y_col}", fontsize=14, pad=15)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"scatterplot_{x_col}_{y_col}.png"), dpi=200)
        plt.close()
        
    def _save_static_bar(self, counts: pd.DataFrame, cat_col: str):
        plt.figure(figsize=(8, 5))
        sns.barplot(x=cat_col, y="count", hue=cat_col, data=counts, palette="viridis", legend=False)
        plt.title(f"Frequency Count of {cat_col}", fontsize=14, pad=15)
        plt.xlabel(cat_col)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"bar_{cat_col}.png"), dpi=200)
        plt.close()
