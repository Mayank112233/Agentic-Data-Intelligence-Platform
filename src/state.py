from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    State representing the context passed between agents in the LangGraph workflow.
    """
    csv_path: str
    target_column: Optional[str]
    
    # Cleaning phase
    cleaned_csv_path: Optional[str]
    cleaning_report: Optional[Dict[str, Any]]
    
    # EDA phase
    eda_results: Optional[Dict[str, Any]]
    
    # Visualization phase
    visualizations: Optional[List[Dict[str, Any]]] # Serialized Plotly figures
    
    # Machine Learning phase
    ml_results: Optional[Dict[str, Any]]
    
    # Insights phase
    ai_insights: Optional[Dict[str, Any]]
    
    # Reporting phase
    report_path: Optional[str]
    
    # Error handling and logs
    error: Optional[str]
    logs: List[str]
