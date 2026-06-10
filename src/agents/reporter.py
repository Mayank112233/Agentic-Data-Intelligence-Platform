import os
import logging
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

from src.state import AgentState
from config.settings import REPORTS_DIR, PLOTS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReportAgent")

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to draw running headers, footers, and dynamic page counts
    ('Page X of Y') on all pages except the cover.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # Skip decorations on cover page
        if self._pageNumber == 1:
            return
            
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7F8C8D"))
        
        # Draw running header
        self.drawString(54, 750, "Autonomous Data Science Agent — Analysis Report")
        self.setStrokeColor(colors.HexColor("#D5D8DC"))
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)
        
        # Draw running footer
        self.line(54, 55, 558, 55)
        self.drawString(54, 40, "Generated Autonomously by AI Agent Team")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, page_text)
        
        self.restoreState()


class ReportAgent:
    """
    Agent responsible for rendering all previous results (Data details, EDA summaries,
    Plotly static charts, ML baseline tables, SHAP feature importances, and AI insights)
    into a professional business PDF report.
    """
    
    def generate_pdf(self, state: AgentState) -> AgentState:
        """Main entry point for the agent execution in the LangGraph workflow."""
        state["logs"].append("Started Report Generation Agent.")
        
        eda_results = state.get("eda_results")
        ml_results = state.get("ml_results")
        ai_insights = state.get("ai_insights")
        target_column = state.get("target_column") or "None (Clustering)"
        
        if not eda_results or not ml_results or not ai_insights:
            error_msg = "Cannot generate report: state elements (EDA, ML, or Insights) are missing."
            logger.error(error_msg)
            state["error"] = error_msg
            state["logs"].append(f"Error: {error_msg}")
            return state
            
        try:
            # Set output path
            report_name = f"data_science_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            report_path = os.path.join(REPORTS_DIR, report_name)
            
            # Setup document properties
            # Letter is 612 x 792 pt. Margins: 0.75in (54pt)
            doc = SimpleDocTemplate(
                report_path,
                pagesize=letter,
                leftMargin=54,
                rightMargin=54,
                topMargin=72,
                bottomMargin=72
            )
            
            # Base Styles
            styles = getSampleStyleSheet()
            
            # Colors
            primary_color = colors.HexColor("#1A5276")  # Navy
            secondary_color = colors.HexColor("#117A65")  # Teal
            dark_neutral = colors.HexColor("#2C3E50")     # Charcoal
            light_neutral = colors.HexColor("#F8F9F9")    # Warm White
            accent_red = colors.HexColor("#C0392B")       # Crimson
            
            # Custom Typography Styles
            title_style = ParagraphStyle(
                'CoverTitle',
                parent=styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=28,
                leading=34,
                textColor=primary_color,
                alignment=0,
                spaceAfter=15
            )
            
            subtitle_style = ParagraphStyle(
                'CoverSubtitle',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=14,
                leading=18,
                textColor=dark_neutral,
                alignment=0,
                spaceAfter=40
            )
            
            h1_style = ParagraphStyle(
                'SectionHeader',
                parent=styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=18,
                leading=22,
                textColor=primary_color,
                spaceBefore=15,
                spaceAfter=10,
                keepWithNext=True
            )
            
            h2_style = ParagraphStyle(
                'SubSectionHeader',
                parent=styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=12,
                leading=16,
                textColor=secondary_color,
                spaceBefore=10,
                spaceAfter=6,
                keepWithNext=True
            )
            
            body_style = ParagraphStyle(
                'ReportBody',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=10,
                leading=14,
                textColor=dark_neutral,
                spaceAfter=8
            )
            
            bullet_style = ParagraphStyle(
                'ReportBullet',
                parent=body_style,
                leftIndent=15,
                bulletIndent=5,
                spaceAfter=4
            )
            
            meta_style = ParagraphStyle(
                'MetaStyle',
                parent=styles['Normal'],
                fontName='Helvetica-Oblique',
                fontSize=9,
                leading=12,
                textColor=colors.HexColor("#7F8C8D")
            )
            
            story = []
            
            # ----------------------------------------------------
            # PAGE 1: COVER PAGE
            # ----------------------------------------------------
            story.append(Spacer(1, 100))
            story.append(Paragraph("AUTONOMOUS DATA SCIENCE REPORT", title_style))
            story.append(Paragraph("A Comprehensive Data Preprocessing, EDA, Machine Learning, and Insight Generation Dashboard", subtitle_style))
            story.append(Spacer(1, 40))
            
            # Metadata Table
            meta_data = [
                [Paragraph("<b>Target Variable:</b>", body_style), Paragraph(target_column, body_style)],
                [Paragraph("<b>Problem Type:</b>", body_style), Paragraph(ml_results.get("problem_type", "").upper(), body_style)],
                [Paragraph("<b>Dataset Cleaned:</b>", body_style), Paragraph(os.path.basename(state.get("cleaned_csv_path", "")), body_style)],
                [Paragraph("<b>Generated Date:</b>", body_style), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), body_style)],
                [Paragraph("<b>Generated By:</b>", body_style), Paragraph("Autonomous Agent Team", body_style)]
            ]
            meta_table = Table(meta_data, colWidths=[120, 300])
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), light_neutral),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7E9")),
            ]))
            story.append(meta_table)
            
            story.append(Spacer(1, 150))
            story.append(Paragraph("<b>Notice:</b> This document contains analysis generated by AI. Use findings to guide business strategies while cross-validating critical decisions.", meta_style))
            story.append(PageBreak())
            
            # ----------------------------------------------------
            # PAGE 2: EXECUTIVE SUMMARY & DATASET OVERVIEW
            # ----------------------------------------------------
            story.append(Paragraph("1. Executive Summary", h1_style))
            story.append(Paragraph(ai_insights.get("executive_summary", ""), body_style))
            story.append(Spacer(1, 15))
            
            story.append(Paragraph("Dataset Structural Summary", h2_style))
            dataset_meta = [
                ["Property", "Value"],
                ["Original Rows", str(state.get("cleaning_report", {}).get("original_shape", [eda_results.get("shape")[0]])[0])],
                ["Cleaned Rows", str(eda_results.get("shape")[0])],
                ["Columns", str(eda_results.get("shape")[1])],
                ["Numeric Features", str(len(eda_results.get("numeric_cols", [])))],
                ["Categorical Features", str(len(eda_results.get("categorical_cols", [])))],
                ["Target Field", target_column]
            ]
            ds_table = Table(dataset_meta, colWidths=[200, 200])
            ds_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), primary_color),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_neutral]),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#D5D8DC")),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7E9"))
            ]))
            story.append(ds_table)
            story.append(PageBreak())
            
            # ----------------------------------------------------
            # PAGE 3: EDA & CORRELATION ANALYSIS
            # ----------------------------------------------------
            story.append(Paragraph("2. Exploratory Data Analysis (EDA)", h1_style))
            eda_overview = eda_results.get("llm_analysis", {}).get("dataset_overview", "")
            story.append(Paragraph(eda_overview, body_style))
            story.append(Spacer(1, 15))
            
            # Embed Heatmap if present
            heatmap_path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
            if os.path.exists(heatmap_path):
                story.append(Paragraph("Feature Interaction Heatmap", h2_style))
                story.append(Image(heatmap_path, width=400, height=300))
                story.append(Spacer(1, 10))
            
            # List some key findings from EDA
            story.append(Paragraph("Key EDA Findings:", h2_style))
            for finding in eda_results.get("llm_analysis", {}).get("key_findings", [])[:4]:
                story.append(Paragraph(f"• {finding}", bullet_style))
                
            story.append(PageBreak())
            
            # ----------------------------------------------------
            # PAGE 4: AUTOMATED MACHINE LEARNING RESULTS
            # ----------------------------------------------------
            story.append(Paragraph("3. Model Training & Comparison", h1_style))
            story.append(Paragraph(
                f"The ML Agent identified this problem as a <b>{ml_results.get('problem_type', '').upper()}</b> challenge. "
                f"Multiple models were trained and cross-validated. Below is the validation comparison table:", 
                body_style
            ))
            story.append(Spacer(1, 10))
            
            # Comparison table
            comparisons = ml_results.get("comparison", [])
            if comparisons:
                headers = list(comparisons[0].keys())
                # Format header text beautifully
                formatted_headers = [h.replace("_", " ").upper() for h in headers]
                table_rows = [formatted_headers]
                
                for comp in comparisons:
                    row = []
                    for h in headers:
                        val = comp[h]
                        if isinstance(val, float):
                            row.append(f"{val:.4f}")
                        else:
                            row.append(str(val))
                    table_rows.append(row)
                    
                comp_table = Table(table_rows, colWidths=[150] + [80]*(len(headers)-1))
                comp_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), secondary_color),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('ALIGN', (0,0), (0,-1), 'LEFT'), # Left align algorithm name
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ('TOPPADDING', (0,0), (-1,-1), 6),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_neutral]),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7E9"))
                ]))
                story.append(comp_table)
                story.append(Spacer(1, 15))
                
            story.append(Paragraph(f"<b>Selected Champion Model:</b> {ml_results.get('best_model_name')}", h2_style))
            story.append(Paragraph("Metrics for the best model on test validation dataset:", body_style))
            
            # Best metrics list
            for m, val in ml_results.get("metrics", {}).items():
                m_name = m.replace("_", " ").title()
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                story.append(Paragraph(f"• <b>{m_name}:</b> {val_str}", bullet_style))
                
            story.append(PageBreak())
            
            # ----------------------------------------------------
            # PAGE 5: EXPLAINABLE AI (SHAP ANALYSIS)
            # ----------------------------------------------------
            story.append(Paragraph("4. Explainable AI (XAI) & SHAP Analysis", h1_style))
            story.append(Paragraph(
                "SHAP (SHapley Additive exPlanations) values decompose the impact of each feature "
                "on the target variable. The charts below display overall global feature importance.",
                body_style
            ))
            story.append(Spacer(1, 10))
            
            # Summary plot image
            summary_plot_path = os.path.join(PLOTS_DIR, "shap_summary.png")
            local_plot_path = os.path.join(PLOTS_DIR, "shap_local_explanation.png")
            
            if os.path.exists(summary_plot_path):
                story.append(Paragraph("Global SHAP Contribution Summary", h2_style))
                story.append(Image(summary_plot_path, width=420, height=240))
                story.append(Spacer(1, 10))
                
            if os.path.exists(local_plot_path):
                story.append(Paragraph("Local Prediction Explanation (Sample Row)", h2_style))
                story.append(Image(local_plot_path, width=420, height=180))
                story.append(Spacer(1, 10))
                
            story.append(PageBreak())
            
            # ----------------------------------------------------
            # PAGE 6: AI-GENERATED BUSINESS INSIGHTS & ACTIONS
            # ----------------------------------------------------
            story.append(Paragraph("5. AI Generated Business Insights", h1_style))
            
            story.append(Paragraph("Actionable Business Insights:", h2_style))
            for insight in ai_insights.get("business_insights", []):
                story.append(Paragraph(f"• {insight}", bullet_style))
                
            story.append(Spacer(1, 10))
            story.append(Paragraph("Strategic Recommendations:", h2_style))
            for rec in ai_insights.get("recommendations", []):
                story.append(Paragraph(f"• {rec}", bullet_style))
                
            story.append(Spacer(1, 10))
            story.append(Paragraph("Identified Risks & Constraints:", h2_style))
            for risk in ai_insights.get("risks_and_limitations", []):
                story.append(Paragraph(f"• {risk}", bullet_style))
                
            story.append(Spacer(1, 10))
            story.append(Paragraph("Immediate Next Actions:", h2_style))
            for action in ai_insights.get("next_actions", []):
                story.append(Paragraph(f"• <b>{action.split(':')[0]}:</b>{':'.join(action.split(':')[1:]) if ':' in action else ''}", bullet_style))
            
            # Build PDF
            doc.build(story, canvasmaker=NumberedCanvas)
            
            state["report_path"] = report_path
            state["logs"].append(f"PDF Report generated successfully at {report_path}")
            
        except Exception as e:
            error_msg = f"Failed in PDF Report Agent: {str(e)}"
            logger.exception(error_msg)
            state["error"] = error_msg
            state["logs"].append(f"Error: {error_msg}")
            
        return state
