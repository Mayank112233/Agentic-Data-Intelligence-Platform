import logging
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

# Import State and Agents
from src.state import AgentState
from src.agents.cleaner import DataCleaningAgent
from src.agents.eda import EDAAgent
from src.agents.visualizer import VisualizationAgent
from src.agents.ml import MLAgent
from src.agents.insight import InsightAgent
from src.agents.reporter import ReportAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LangGraphWorkflow")

# Instantiate Agent Classes
cleaner_agent = DataCleaningAgent()
eda_agent = EDAAgent()
visualizer_agent = VisualizationAgent()
ml_agent = MLAgent()
insight_agent = InsightAgent()
report_agent = ReportAgent()

# ----------------------------------------------------
# Define Node Functions
# ----------------------------------------------------

def cleaner_node(state: AgentState) -> AgentState:
    logger.info("Entering node: Data Cleaning")
    if state.get("error"):
        state["logs"].append("Skipping Data Cleaning due to prior error.")
        return state
    return cleaner_agent.clean(state)

def eda_node(state: AgentState) -> AgentState:
    logger.info("Entering node: EDA")
    if state.get("error"):
        state["logs"].append("Skipping EDA due to prior error.")
        return state
    return eda_agent.analyze(state)

def visualizer_node(state: AgentState) -> AgentState:
    logger.info("Entering node: Visualizations")
    if state.get("error"):
        state["logs"].append("Skipping Visualizations due to prior error.")
        return state
    return visualizer_agent.generate_charts(state)

def ml_node(state: AgentState) -> AgentState:
    logger.info("Entering node: Machine Learning")
    if state.get("error"):
        state["logs"].append("Skipping Machine Learning due to prior error.")
        return state
    return ml_agent.train(state)

def insight_node(state: AgentState) -> AgentState:
    logger.info("Entering node: Insight Generation")
    if state.get("error"):
        state["logs"].append("Skipping Insight Generation due to prior error.")
        return state
    return insight_agent.generate_insights(state)

def reporter_node(state: AgentState) -> AgentState:
    logger.info("Entering node: PDF Report Generation")
    if state.get("error"):
        state["logs"].append("Skipping Report Generation due to prior error.")
        return state
    return report_agent.generate_pdf(state)

# ----------------------------------------------------
# Define Conditional Routers
# ----------------------------------------------------

def check_error_cleaner(state: AgentState) -> str:
    if state.get("error"):
        return "end"
    return "eda"

def check_error_eda(state: AgentState) -> str:
    if state.get("error"):
        return "end"
    return "visualizer"

def check_error_visualizer(state: AgentState) -> str:
    if state.get("error"):
        return "end"
    return "ml"

def check_error_ml(state: AgentState) -> str:
    if state.get("error"):
        return "end"
    return "insight"

def check_error_insight(state: AgentState) -> str:
    if state.get("error"):
        return "end"
    return "reporter"

# ----------------------------------------------------
# Build and Compile the Graph
# ----------------------------------------------------

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("cleaner", cleaner_node)
workflow.add_node("eda", eda_node)
workflow.add_node("visualizer", visualizer_node)
workflow.add_node("ml", ml_node)
workflow.add_node("insight", insight_node)
workflow.add_node("reporter", reporter_node)

# Add Edges
workflow.add_edge(START, "cleaner")

# Route conditional edges based on whether an error occurred
workflow.add_conditional_edges(
    "cleaner", 
    check_error_cleaner, 
    {"eda": "eda", "end": END}
)
workflow.add_conditional_edges(
    "eda", 
    check_error_eda, 
    {"visualizer": "visualizer", "end": END}
)
workflow.add_conditional_edges(
    "visualizer", 
    check_error_visualizer, 
    {"ml": "ml", "end": END}
)
workflow.add_conditional_edges(
    "ml", 
    check_error_ml, 
    {"insight": "insight", "end": END}
)
workflow.add_conditional_edges(
    "insight", 
    check_error_insight, 
    {"reporter": "reporter", "end": END}
)

# final edge to END
workflow.add_edge("reporter", END)

# Compile Graph
graph_app = workflow.compile()

# ----------------------------------------------------
# Execution Wrapper
# ----------------------------------------------------

def run_workflow(csv_path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Utility helper to execute the autonomous data science workflow.
    Returns the final state after processing.
    """
    initial_state: AgentState = {
        "csv_path": csv_path,
        "target_column": target_column if target_column != "None (Clustering)" else None,
        "cleaned_csv_path": None,
        "cleaning_report": None,
        "eda_results": None,
        "visualizations": [],
        "ml_results": None,
        "ai_insights": None,
        "report_path": None,
        "error": None,
        "logs": ["Initialized LangGraph workflow state."]
    }
    
    logger.info(f"Triggering LangGraph workflow for dataset: {csv_path}")
    final_state = graph_app.invoke(initial_state)
    return final_state
