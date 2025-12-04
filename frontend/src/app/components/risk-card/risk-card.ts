import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatChipsModule } from '@angular/material/chips';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-risk-card',
  standalone: true,
  imports: [CommonModule, MatCardModule, MatChipsModule, MatIconModule],
  templateUrl: './risk-card.html',
  styleUrls: ['./risk-card.css']
})
export class RiskCard {
  @Input() score: number = 0;
  @Input() riskClass: string = 'Low';

  getChipColor(): string {
    switch (this.riskClass) {
      case 'High': return 'warn';
      case 'Moderate': return 'accent';
      default: return 'primary';
    }
  }

  getIcon(): string {
    switch (this.riskClass) {
      case 'High': return 'warning';
      case 'Moderate': return 'info';
      default: return 'check_circle';
    }
  }

  getDescription(): string {
    switch (this.riskClass) {
      case 'High': return 'Immediate intervention recommended.';
      case 'Moderate': return 'Monitor closely and consider lifestyle changes.';
      default: return 'Maintain current healthy lifestyle.';
    }
  }
}
