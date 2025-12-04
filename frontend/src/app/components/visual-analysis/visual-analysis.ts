import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';

@Component({
  selector: 'app-visual-analysis',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatIconModule, MatProgressSpinnerModule],
  templateUrl: './visual-analysis.html',
  styleUrls: ['./visual-analysis.css']
})
export class VisualAnalysis {
  @Input() data: any;
  downloading = false;

  constructor(private api: ApiService) { }

  downloadReport() {
    this.downloading = true;

    // Construct the payload expected by the backend
    // Note: In a real app, we might want to pass the original form data too, 
    // but here we'll mock the patient_data structure based on what we have or can infer,
    // or we should have passed it down. 
    // For now, let's create a minimal payload that satisfies the backend.
    // The backend expects: patient_data, risk_analysis, findings, images.

    const payload = {
      patient_data: {
        "Age": "N/A", "Sex": "N/A", "BMI": "N/A" // We don't have this in 'data' prop currently, would need to pass it down.
        // For this demo, we'll send placeholders or we need to update the parent to pass form data.
      },
      risk_analysis: {
        risk_score: this.data.risk_score,
        risk_class: this.data.risk_class
      },
      findings: this.data.findings,
      images: this.data.images
    };

    this.api.generateReport(payload).subscribe({
      next: (blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'OA_Prognosis_Report.pdf';
        a.click();
        window.URL.revokeObjectURL(url);
        this.downloading = false;
      },
      error: (err) => {
        console.error('Report generation failed', err);
        alert('Failed to generate report.');
        this.downloading = false;
      }
    });
  }
}
