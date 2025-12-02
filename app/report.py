from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import matplotlib.pyplot as plt

def create_pdf_report(patient_data, risk_score, risk_class, images):
    """
    Generates a PDF report for the OA Prognosis System.
    
    Args:
        patient_data (dict): {'Age': 65, 'BMI': 28.5, ...}
        risk_score (float): The predicted log-hazard.
        risk_class (str): "High", "Moderate", or "Low".
        images (dict): {'original': PIL_Image, 'counterfactual': PIL_Image, 'graph': Matplotlib_Fig}
    
    Returns:
        bytes: The PDF file content.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # 1. Title
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=1, spaceAfter=20)
    story.append(Paragraph("Knee Osteoarthritis Prognosis Report", title_style))
    story.append(Spacer(1, 12))

    # 2. Patient Summary Table
    story.append(Paragraph("<b>Patient Summary</b>", styles['Heading2']))
    
    data = [
        ["Parameter", "Value", "Parameter", "Value"],
        ["Age", str(patient_data.get('Age')), "Sex", str(patient_data.get('Sex'))],
        ["BMI", str(patient_data.get('BMI')), "KL Grade", str(patient_data.get('KL Grade'))],
        ["WOMAC", str(patient_data.get('WOMAC')), "Serum COMP", f"{patient_data.get('COMP'):.1f}"]
    ]
    
    t = Table(data, colWidths=[100, 100, 100, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    # 3. Prognostic Assessment
    story.append(Paragraph("<b>Prognostic Assessment</b>", styles['Heading2']))
    
    risk_color = "red" if risk_class == "High" else "orange" if risk_class == "Moderate" else "green"
    risk_text = f"""
    <b>Predicted Log-Hazard Score:</b> {risk_score:.4f}<br/>
    <b>Risk Classification:</b> <font color='{risk_color}'>{risk_class}</font><br/><br/>
    AI analysis indicates a {risk_class.lower()} likelihood of rapid structural progression compared to the cohort baseline.
    """
    story.append(Paragraph(risk_text, styles['Normal']))
    story.append(Spacer(1, 20))

    # 4. Survival Curve (Save Matplotlib fig to buffer)
    if 'graph' in images:
        img_buf = BytesIO()
        images['graph'].savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
        img_buf.seek(0)
        story.append(RLImage(img_buf, width=400, height=200))
        story.append(Paragraph("<i>Figure 1: 10-Year Progression-Free Probability Curve</i>", styles['Italic']))
        story.append(Spacer(1, 20))

    # 5. Generative Biomarkers
    story.append(Paragraph("<b>Generative Biomarker Analysis</b>", styles['Heading2']))
    story.append(Paragraph("Visual comparison of current anatomy vs. projected healthy state (Counterfactual).", styles['Normal']))
    story.append(Spacer(1, 10))

    # Handle Images (We assume inputs are PIL images or paths)
    # Note: ReportLab needs file paths or file-like objects. 
    # We will assume the app passes PIL images, which we save to temp buffers.
    
    # (Layout for images side-by-side would involve a Table)
    img_data = []
    row = []
    
    for key, caption in [('original', 'Original X-Ray'), ('counterfactual', 'Counterfactual')]:
        if key in images:
            # Convert PIL to Buffer
            ibuf = BytesIO()
            images[key].save(ibuf, format='PNG')
            ibuf.seek(0)
            row.append(RLImage(ibuf, width=200, height=200))
    
    if row:
        img_data.append(row)
        img_data.append(["Current Anatomy", "Projected Healthy State"]) # Captions
        
        img_table = Table(img_data)
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, 1), 5),
        ]))
        story.append(img_table)

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer