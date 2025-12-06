import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { PredictionResult, ReportRequest, RiskAnalysis } from '../../interfaces/prediction.interface';

@Component({
  selector: 'app-visual-analysis',
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatIconModule, MatProgressSpinnerModule],
  templateUrl: './visual-analysis.html',
  styleUrls: ['./visual-analysis.css']
})
export class VisualAnalysis implements OnChanges {
  @Input() data!: PredictionResult;
  downloading = false;

  /** Tracks which images have finished loading */
  imageLoaded = {
    original: false,
    counterfactual: false,
    heatmap: false
  };

  constructor(private api: ApiService) { }

  /** Reset loading states when new data arrives */
  ngOnChanges(changes: SimpleChanges): void {
    if (changes['data']) {
      this.imageLoaded = {
        original: false,
        counterfactual: false,
        heatmap: false
      };
    }
  }

  /** Called when an image finishes loading */
  onImageLoad(imageKey: 'original' | 'counterfactual' | 'heatmap'): void {
    this.imageLoaded[imageKey] = true;
  }

  downloadReport() {
    this.downloading = true;

    // Calculate 5-year probability from survival curve
    const prob5yr = this.calculate5YearProbability();

    const riskAnalysis: RiskAnalysis = {
      risk_score: this.data.risk_score,
      risk_class: this.data.risk_class,
      prob_5yr: prob5yr
    };

    const payload: ReportRequest = {
      patient_data: this.data.patient_data,
      risk_analysis: riskAnalysis,
      findings: this.data.findings || [],
      images: this.data.images || {}
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

  private calculate5YearProbability(): string {
    if (!this.data.survival_curve || this.data.survival_curve.length === 0) {
      return 'N/A';
    }

    const point5yr = this.data.survival_curve.find(p => p.x >= 5);
    if (point5yr) {
      return point5yr.y.toFixed(2);
    }

    // If no point at 5 years, use last available point
    const lastPoint = this.data.survival_curve[this.data.survival_curve.length - 1];
    return lastPoint.y.toFixed(2);
  }
}
