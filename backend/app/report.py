from io import BytesIO
import time
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_pdf_report(patient_data, risk_analysis, findings, images):
    """
    Generates a professional Clinical Decision Support (CDS) report.
    
    Args:
        patient_data (dict): Demographics and inputs.
        risk_analysis (dict): {'score': float, 'class': str, 'prob_5yr': str}.
        findings (list): List of strings detailing the visual/biomarker analysis.
        images (dict): Dictionary of PIL images or Matplotlib figures.
    
    Returns:
        BytesIO: The generated PDF file in memory.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []

    # --- HEADER ---
    # Mimic a hospital letterhead
    header_style = ParagraphStyle('Header', parent=styles['Normal'], fontSize=10, textColor=colors.gray)
    story.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M')} | System: Tri-Modal OA-AI v1.0", header_style))
    story.append(Spacer(1, 0.2*inch))
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, spaceAfter=12, textColor=colors.darkblue)
    story.append(Paragraph("Osteoarthritis Progression Risk Report", title_style))
    story.append(Paragraph("<i>Confidential - For Clinical Decision Support Only</i>", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # --- SECTION 1: PATIENT CONTEXT ---
    story.append(Paragraph("1. Patient Profile & Clinical Biomarkers", styles['Heading2']))
    
    # Format data for table
    # We group Clinical vs. Biomarkers for readability
    data = [
        ["Category", "Parameter", "Value", "Reference Range"],
        ["Demographics", "Patient ID/Name", f"{patient_data.get('ID', 'Anonymous')}", "-"],
        ["", "Age / Sex", f"{patient_data['Age']} / {patient_data['Sex']}", "-"],
        ["", "BMI", f"{patient_data['BMI']}", "18.5 - 24.9"],
        ["Clinical", "KL Grade", f"{patient_data['KL Grade']}", "0 (Healthy) - 4 (Severe)"],
        ["", "WOMAC Score", f"{patient_data['WOMAC']}", "0 (Best) - 96 (Worst)"],
        ["Biomarkers", "Serum COMP", f"{patient_data['COMP']:.1f} ng/mL", "< 1200"],
        ["", "Urine CTX-II", f"{patient_data['CTX']:.1f} ng/mmol", "< 400"]
    ]
    
    t = Table(data, colWidths=[100, 150, 120, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('SPAN', (0, 1), (0, 2)), # Span Demographics
        ('SPAN', (0, 3), (0, 4)), # Span Clinical
        ('SPAN', (0, 5), (0, 6)), # Span Biomarkers
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*inch))

    # --- SECTION 2: PROGNOSIS ---
    story.append(Paragraph("2. AI Prognostic Assessment", styles['Heading2']))
    
    # Color-coded risk text
    # Changed 'class' to 'risk_class' to match API payload
    r_class = risk_analysis.get('risk_class', 'Unknown')
    risk_color = "red" if r_class == "High" else "#FFA500" if r_class == "Moderate" else "green"
    
    risk_summary = f"""
    <b>Primary Analysis:</b> The multi-modal model predicts a <font color='{risk_color}'><b>{r_class.upper()} RISK</b></font> of progression.<br/><br/>
    <b>Normalized Risk Score (0-100):</b> {risk_analysis['risk_score']:.1f}<br/>
    <b>5-Year Surgery Probability:</b> {risk_analysis['prob_5yr']}<br/>
    """
    story.append(Paragraph(risk_summary, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Survival Graph
    # Survival Graph
    if 'graph' in images:
        img_buf = BytesIO()
        if hasattr(images['graph'], 'save'):
            images['graph'].save(img_buf, format='PNG')
        else:
            images['graph'].savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        img_buf.seek(0)
        story.append(RLImage(img_buf, width=450, height=220))
        story.append(Paragraph("<b>Figure 1:</b> Projected Probability of Surgery (10-Year Horizon)", styles['Italic']))
    
    story.append(Spacer(1, 0.3*inch))

    # --- SECTION 3: EXPLAINABILITY ---
    story.append(Paragraph("3. Generative Biomarker Analysis (XAI)", styles['Heading2']))
    story.append(Paragraph("""
    The system utilized a Diffusion Autoencoder to generate a 'Survival Counterfactual'. 
    The difference map below isolates structural features (Osteophytes, JSN) contributing to the risk score.
    """, styles['Normal']))
    story.append(Spacer(1, 0.1*inch))

    # Images Table (Side by Side)
    img_row = []
    for key in ['original', 'counterfactual', 'heatmap']:
        if key in images:
            ibuf = BytesIO()
            # Handle PIL vs Matplotlib
            if hasattr(images[key], 'save'):
                images[key].save(ibuf, format='PNG')
            else: # Matplotlib figure (Heatmap)
                images[key].savefig(ibuf, format='png', dpi=150, bbox_inches='tight')
            
            ibuf.seek(0)
            img_row.append(RLImage(ibuf, width=1.8*inch, height=1.8*inch))
    
    if img_row:
        t_img = Table([img_row, ["Patient Anatomy", "Projected Healthy", "Risk Biomarkers (Heatmap)"]], colWidths=[2*inch, 2*inch, 2*inch])
        t_img.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTSIZE', (0,1), (-1,1), 9),
            ('TOPPADDING', (0,1), (-1,1), 5),
        ]))
        story.append(t_img)

    story.append(Spacer(1, 0.2*inch))
    
    # Detailed Findings (Dynamic Text)
    story.append(Paragraph("<b>Detected Structural Deviations:</b>", styles['Heading4']))
    for item in findings:
        # Clean markdown bullets if present
        clean_text = item.replace("*", "").strip()
        story.append(Paragraph(f"• {clean_text}", styles['Normal']))

    # --- FOOTER ---
    story.append(Spacer(1, 0.5*inch))
    disclaimer = """
    <b>DISCLAIMER:</b> This report is generated by an AI research prototype (MTech Thesis). 
    It is not a certified medical diagnostic tool. All predictions should be verified by a qualified radiologist.
    """
    story.append(Paragraph(disclaimer, styles['BodyText']))

    doc.build(story)
    buffer.seek(0)
    return buffer