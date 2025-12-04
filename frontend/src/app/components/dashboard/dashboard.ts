import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatTabsModule } from '@angular/material/tabs';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { RiskCard } from '../risk-card/risk-card';
import { SurvivalChart } from '../survival-chart/survival-chart';
import { VisualAnalysis } from '../visual-analysis/visual-analysis';

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
  @Input() data: any;
}
