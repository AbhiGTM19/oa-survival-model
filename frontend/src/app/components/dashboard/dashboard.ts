import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatTabsModule } from '@angular/material/tabs';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { RiskCard } from '../risk-card/risk-card';
import { SurvivalChart } from '../survival-chart/survival-chart';
import { VisualAnalysis } from '../visual-analysis/visual-analysis';
import { PredictionResult } from '../../interfaces/prediction.interface';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    MatTabsModule,
    MatCardModule,
    MatIconModule,
    RiskCard,
    SurvivalChart,
    VisualAnalysis
  ],
  templateUrl: './dashboard.html',
  styleUrls: ['./dashboard.css']
})
export class Dashboard {
  @Input() data!: PredictionResult;

  /** Maps KL grade number to human-readable label */
  getKLGradeLabel(grade: number): string {
    const labels: Record<number, string> = {
      0: 'None (0)',
      1: 'Doubtful (1)',
      2: 'Minimal (2)',
      3: 'Moderate (3)',
      4: 'Severe (4)'
    };
    return labels[grade] || `Grade ${grade}`;
  }

  /** Categorizes BMI into risk impact levels */
  getBMICategory(bmi: number): string {
    if (bmi < 18.5) return 'Underweight';
    if (bmi < 25) return 'Normal';
    if (bmi < 30) return 'Elevated';
    if (bmi < 35) return 'High';
    return 'Very High';
  }

  /** Determines biomarker status based on COMP and CTX-II levels */
  getBiomarkerStatus(comp: number, ctx: number): string {
    // Reference ranges: COMP < 1200, CTX-II < 400
    const compElevated = comp > 1200;
    const ctxElevated = ctx > 400;

    if (compElevated && ctxElevated) {
      return 'Elevated COMP & CTX-II';
    } else if (compElevated) {
      return 'Elevated COMP';
    } else if (ctxElevated) {
      return 'Elevated CTX-II';
    }
    return 'Normal Range';
  }
}
