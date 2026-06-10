import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import shap
from typing import Dict, Any, List, Tuple, Optional

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, silhouette_score
)

from src.state import AgentState
from config.settings import OUTPUT_DIR, PLOTS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLAgent")

class MLAgent:
    """
    Agent responsible for detecting problem type, training and comparing ML models,
    selecting the best performer, and generating SHAP explainability analyses.
    """
    
    def detect_problem_type(self, df: pd.DataFrame, target_col: Optional[str]) -> str:
        """Detect if the task is classification, regression, or clustering."""
        if not target_col or target_col not in df.columns:
            return "clustering"
            
        target_series = df[target_col].dropna()
        unique_count = target_series.nunique()
        dtype_str = str(target_series.dtype)
        
        # Heuristics
        if "object" in dtype_str or "category" in dtype_str or "bool" in dtype_str:
            return "classification"
        elif "int" in dtype_str and unique_count <= 10:
            return "classification"
        else:
            return "regression"

    def preprocess_data(
        self, df: pd.DataFrame, target_col: Optional[str], problem_type: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], Any]:
        """Set up automated preprocessing pipelines using ColumnTransformer."""
        
        if problem_type != "clustering" and target_col:
            X = df.drop(columns=[target_col])
            # Drop target NaNs for supervised learning
            valid_idx = df[target_col].notnull()
            X = X[valid_idx]
            y = df.loc[valid_idx, target_col].values
            
            # Label encode categorical target if classification
            if problem_type == "classification" and df[target_col].dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
        else:
            X = df.copy()
            y = None
            
        # Separate numeric and categorical
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        
        # Pipelines
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, numeric_cols),
                ('cat', cat_transformer, categorical_cols)
            ]
        )
        
        X_processed = preprocessor.fit_transform(X)
        
        # Extract processed feature names
        feature_names = []
        if len(numeric_cols) > 0:
            feature_names.extend(numeric_cols)
        if len(categorical_cols) > 0:
            try:
                onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_features = onehot_encoder.get_feature_names_out(categorical_cols).tolist()
                feature_names.extend(cat_features)
            except Exception:
                # Fallback to general indexing if feature names fail
                cat_features = [f"{col}_{i}" for col in categorical_cols for i in range(X_processed.shape[1] - len(numeric_cols))]
                feature_names.extend(cat_features)
                
        # Handle cases where there might be dimension mismatches
        if X_processed.shape[1] != len(feature_names):
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
            
        return X_processed, y, feature_names, preprocessor

    def train(self, state: AgentState) -> AgentState:
        """Train models, select best, generate predictions and SHAP plots."""
        state["logs"].append("Started Machine Learning Agent.")
        csv_path = state.get("cleaned_csv_path") or state.get("csv_path")
        target_col = state.get("target_column")
        
        if not csv_path or not os.path.exists(csv_path):
            error_msg = f"CSV path '{csv_path}' does not exist or was not provided."
            logger.error(error_msg)
            state["error"] = error_msg
            state["logs"].append(f"Error: {error_msg}")
            return state
            
        try:
            # 1. Load data
            df = pd.read_csv(csv_path)
            
            # 2. Detect problem type
            problem_type = self.detect_problem_type(df, target_col)
            state["logs"].append(f"Detected problem type: {problem_type.upper()}")
            
            # 3. Preprocess
            X_proc, y, feature_names, preprocessor = self.preprocess_data(df, target_col, problem_type)
            
            # Define outputs
            best_model_name = ""
            best_model = None
            metrics_summary = {}
            comparison_results = []
            
            if problem_type == "clustering":
                # KMeans Clustering
                # Automatically evaluate silhouette score for clusters K=2 to K=5
                best_k = 3
                best_sil = -1
                cluster_models = {}
                
                for k in range(2, 6):
                    if len(X_proc) > k:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = km.fit_predict(X_proc)
                        sil = float(silhouette_score(X_proc, labels))
                        cluster_models[k] = (km, sil)
                        comparison_results.append({"algorithm": f"KMeans (k={k})", "silhouette_score": sil})
                        if sil > best_sil:
                            best_sil = sil
                            best_k = k
                
                best_model, best_sil = cluster_models[best_k]
                best_model_name = f"KMeans (k={best_k})"
                metrics_summary = {"silhouette_score": best_sil, "n_clusters": best_k}
                
                # Save fitted labels to dataframe to inspect clusters
                df['Cluster'] = best_model.labels_
                df.to_csv(csv_path, index=False)
                state["logs"].append(f"KMeans clustered with k={best_k}. Silhouette score: {best_sil:.4f}")
                
            else:
                # Supervised: Classification/Regression
                # 4. Train-Test Split (80/20)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_proc, y, test_size=0.2, random_state=42
                )
                
                if problem_type == "classification":
                    # Candidate Models
                    models = {
                        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                    }
                    
                    best_f1 = -1
                    for name, model in models.items():
                        try:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                            # Metrics
                            acc = float(accuracy_score(y_test, y_pred))
                            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                            
                            metrics = {
                                "accuracy": acc,
                                "precision": float(prec),
                                "recall": float(rec),
                                "f1_score": float(f1)
                            }
                            
                            # Try ROC-AUC
                            try:
                                if len(np.unique(y_test)) == 2:
                                    y_prob = model.predict_proba(X_test)[:, 1]
                                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
                                else:
                                    y_prob = model.predict_proba(X_test)
                                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob, multi_class='ovr'))
                            except Exception:
                                metrics["roc_auc"] = 0.0
                                
                            comparison_results.append({"algorithm": name, **metrics})
                            
                            if metrics["f1_score"] > best_f1:
                                best_f1 = metrics["f1_score"]
                                best_model = model
                                best_model_name = name
                                metrics_summary = metrics
                        except Exception as e:
                            logger.warning(f"Failed to train classification model {name}: {e}")
                            
                elif problem_type == "regression":
                    # Candidate Models
                    models = {
                        "Linear Regression": LinearRegression(),
                        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
                        "XGBoost Regressor": XGBRegressor(random_state=42)
                    }
                    
                    best_r2 = -float('inf')
                    for name, model in models.items():
                        try:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                            # Metrics
                            mse = float(mean_squared_error(y_test, y_pred))
                            rmse = float(np.sqrt(mse))
                            mae = float(mean_absolute_error(y_test, y_pred))
                            r2 = float(r2_score(y_test, y_pred))
                            
                            metrics = {
                                "mse": mse,
                                "rmse": rmse,
                                "mae": mae,
                                "r2_score": r2
                            }
                            comparison_results.append({"algorithm": name, **metrics})
                            
                            if r2 > best_r2:
                                best_r2 = r2
                                best_model = model
                                best_model_name = name
                                metrics_summary = metrics
                        except Exception as e:
                            logger.warning(f"Failed to train regression model {name}: {e}")
            
            # Save best model and preprocessor
            model_save_path = os.path.join(OUTPUT_DIR, "best_model.joblib")
            preprocessor_save_path = os.path.join(OUTPUT_DIR, "preprocessor.joblib")
            joblib.dump(best_model, model_save_path)
            joblib.dump(preprocessor, preprocessor_save_path)
            
            # 5. Explainable AI (SHAP)
            # Pre-populate shap flags in results
            ml_results = {
                "problem_type": problem_type,
                "best_model_name": best_model_name,
                "metrics": metrics_summary,
                "comparison": comparison_results,
                "model_path": model_save_path,
                "feature_names": feature_names,
                "has_shap": False,
                "feature_importance": []
            }
            
            if problem_type != "clustering" and best_model:
                try:
                    # Construct DataFrame for SHAP explanation
                    X_test_df = pd.DataFrame(X_test, columns=feature_names)
                    X_train_df = pd.DataFrame(X_train, columns=feature_names)
                    
                    # Compute SHAP
                    logger.info("Computing SHAP values...")
                    
                    # Choose explainer based on model type
                    if "Random Forest" in best_model_name or "XGBoost" in best_model_name:
                        explainer = shap.TreeExplainer(best_model)
                        shap_values = explainer.shap_values(X_test_df)
                    else:
                        # Linear model or fallback
                        explainer = shap.Explainer(best_model, X_train_df)
                        shap_values = explainer(X_test_df)
                        
                    # Handle multi-class outputs where shap_values is a list
                    if isinstance(shap_values, list):
                        # Use first class (index 0) or aggregate. Let's take index 0 or class with high variance
                        shap_values_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    else:
                        shap_values_to_plot = shap_values
                        
                    # Extract numeric feature importance from SHAP (mean absolute SHAP)
                    # Handle case where shap_values is Explanation object vs raw array
                    if hasattr(shap_values_to_plot, "values"):
                        raw_values = shap_values_to_plot.values
                    else:
                        raw_values = shap_values_to_plot
                        
                    mean_shap = np.abs(raw_values).mean(axis=0)
                    feat_imp = []
                    for name, val in zip(feature_names, mean_shap):
                        feat_imp.append({"feature": name, "importance": float(val)})
                    feat_imp = sorted(feat_imp, key=lambda x: x["importance"], reverse=True)
                    ml_results["feature_importance"] = feat_imp
                    
                    # 1. Save Global Summary Plot
                    plt.figure(figsize=(8, 5))
                    shap.summary_plot(shap_values_to_plot, X_test_df, show=False)
                    plt.title("SHAP Feature Importance Summary", fontsize=12, pad=15)
                    plt.tight_layout()
                    plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), dpi=200)
                    plt.close()
                    
                    # 2. Save Feature Importance Bar Plot
                    plt.figure(figsize=(8, 5))
                    shap.summary_plot(shap_values_to_plot, X_test_df, plot_type="bar", show=False)
                    plt.title("SHAP Feature Importance (Bar Plot)", fontsize=12, pad=15)
                    plt.tight_layout()
                    plt.savefig(os.path.join(PLOTS_DIR, "shap_feature_importance.png"), dpi=200)
                    plt.close()
                    
                    # 3. Save Local Explanation for the first row of test set
                    plt.figure(figsize=(8, 4))
                    if hasattr(shap_values_to_plot, "values"):
                        # If Explanation object, plots.waterfall works
                        shap.plots.waterfall(shap_values_to_plot[0], show=False)
                    else:
                        # Otherwise build custom bar chart for local explanation
                        row_vals = raw_values[0]
                        sorted_idx = np.argsort(np.abs(row_vals))[::-1][:10] # Top 10 features
                        plt.barh(
                            [feature_names[i] for i in sorted_idx][::-1],
                            [row_vals[i] for i in sorted_idx][::-1],
                            color=["red" if row_vals[i] > 0 else "blue" for i in sorted_idx][::-1]
                        )
                        plt.xlabel("SHAP Value (Impact on Prediction)")
                    
                    plt.title("Local Explanation for Sample Row", fontsize=12, pad=15)
                    plt.tight_layout()
                    plt.savefig(os.path.join(PLOTS_DIR, "shap_local_explanation.png"), dpi=200)
                    plt.close()
                    
                    ml_results["has_shap"] = True
                    state["logs"].append("SHAP values computed and plots saved successfully.")
                    
                except Exception as shap_err:
                    # Fallback to native feature importances if SHAP fails
                    logger.warning(f"SHAP explanation failed, falling back to native importances: {shap_err}")
                    state["logs"].append(f"Warning: SHAP failed ({shap_err}). Generating fallback importances.")
                    self._generate_fallback_importance(best_model, best_model_name, feature_names, ml_results)
            
            state["ml_results"] = ml_results
            state["logs"].append(
                f"Model training complete. Selected Best Model: {best_model_name}. "
                f"Accuracy/R2: {metrics_summary.get('accuracy') or metrics_summary.get('r2_score') or 'N/A'}"
            )
            
        except Exception as e:
            error_msg = f"Failed in ML Agent: {str(e)}"
            logger.exception(error_msg)
            state["error"] = error_msg
            state["logs"].append(f"Error: {error_msg}")
            
        return state

    def _generate_fallback_importance(
        self, model: Any, model_name: str, feature_names: List[str], ml_results: Dict[str, Any]
    ):
        """Generate native feature importances or coefficients if SHAP fails."""
        try:
            importances = []
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Handle multi-class coefficient vectors
                if len(model.coef_.shape) > 1:
                    importances = np.mean(np.abs(model.coef_), axis=0)
                else:
                    importances = np.abs(model.coef_)
                    
            if len(importances) > 0:
                feat_imp = []
                for name, val in zip(feature_names, importances):
                    feat_imp.append({"feature": name, "importance": float(val)})
                feat_imp = sorted(feat_imp, key=lambda x: x["importance"], reverse=True)
                ml_results["feature_importance"] = feat_imp
                
                # Plot fallback bar plot
                plt.figure(figsize=(8, 5))
                top_imp = feat_imp[:15]
                plt.barh([x["feature"] for x in top_imp][::-1], [x["importance"] for x in top_imp][::-1], color="#E67E22")
                plt.title("Feature Importance (Model Native)", fontsize=12, pad=15)
                plt.xlabel("Importance Score")
                plt.tight_layout()
                plt.savefig(os.path.join(PLOTS_DIR, "shap_feature_importance.png"), dpi=200)
                plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), dpi=200) # Duplicate as summary for layout consistency
                plt.close()
                
                # Plot mock local explanation
                plt.figure(figsize=(8, 4))
                plt.text(0.5, 0.5, "Local SHAP explanation not available for this model type.", 
                         ha='center', va='center', fontsize=10, style='italic')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(PLOTS_DIR, "shap_local_explanation.png"), dpi=200)
                plt.close()
                ml_results["has_shap"] = True
        except Exception as e:
            logger.error(f"Fallback feature importance generation failed: {e}")
            ml_results["has_shap"] = False
