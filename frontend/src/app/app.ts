import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatDialogModule, MatDialog } from '@angular/material/dialog';
import { MatTooltipModule } from '@angular/material/tooltip';
import { LottieComponent, AnimationOptions } from 'ngx-lottie';
import { InputForm } from './components/input-form/input-form';
import { Dashboard } from './components/dashboard/dashboard';
import { HelpDialog } from './components/help-dialog/help-dialog';
import { SettingsDialog } from './components/settings-dialog/settings-dialog';
import { PredictionResult } from './interfaces/prediction.interface';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    MatToolbarModule,
    MatIconModule,
    MatButtonModule,
    MatDialogModule,
    MatTooltipModule,
    LottieComponent,
    InputForm,
    Dashboard
  ],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App {
  results: PredictionResult | null = null;
  loading = false;

  lottieOptions: AnimationOptions = {
    path: '/assets/animations/loading.json',
    loop: true,
    autoplay: true
  };

  constructor(private dialog: MatDialog) { }

  onAnalysisComplete(event: PredictionResult | 'loading' | null) {
    if (event === 'loading') {
      this.loading = true;
      this.results = null;
    } else {
      this.loading = false;
      this.results = event;
    }
  }

  openHelpDialog(): void {
    this.dialog.open(HelpDialog, {
      width: '600px',
      panelClass: 'help-dialog'
    });
  }

  openSettingsDialog(): void {
    this.dialog.open(SettingsDialog, {
      width: '500px',
      panelClass: 'settings-dialog'
    });
  }
}
