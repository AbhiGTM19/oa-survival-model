import { Component, Input, OnChanges, SimpleChanges, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import Chart from 'chart.js/auto';

@Component({
  selector: 'app-survival-chart',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './survival-chart.html',
  styleUrls: ['./survival-chart.css']
})
export class SurvivalChart implements AfterViewInit, OnChanges {
  @Input() data: any[] = [];
  @ViewChild('chartCanvas') chartCanvas!: ElementRef;
  chart: any;

  ngAfterViewInit() {
    if (this.data) {
      this.renderChart();
    }
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['data'] && this.chart) {
      this.renderChart();
    }
  }

  renderChart() {
    if (this.chart) this.chart.destroy();

    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    const labels = this.data.map(p => p.x.toFixed(1));
    const values = this.data.map(p => p.y);

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Surgery-Free Probability',
          data: values,
          borderColor: '#EF4444',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderWidth: 3,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 6,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            mode: 'index', intersect: false, backgroundColor: 'rgba(15, 23, 42, 0.9)',
            titleFont: { family: 'Inter', size: 13 }, bodyFont: { family: 'Inter', size: 13 },
            padding: 10, cornerRadius: 8, displayColors: false
          }
        },
        scales: {
          x: {
            type: 'linear', position: 'bottom',
            title: { display: true, text: 'Years from Baseline', font: { family: 'Inter', weight: 600 } },
            grid: { display: false }
          },
          y: {
            min: 0, max: 1,
            title: { display: true, text: 'Survival Probability', font: { family: 'Inter', weight: 600 } },
            // @ts-ignore
            grid: { color: '#E2E8F0', borderDash: [4, 4] }
          }
        },
        interaction: { mode: 'nearest', axis: 'x', intersect: false }
      }
    });
  }
}
