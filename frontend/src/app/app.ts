import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { LottieComponent, AnimationOptions } from 'ngx-lottie';
import { InputForm } from './components/input-form/input-form';
import { Dashboard } from './components/dashboard/dashboard';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    MatSidenavModule,
    MatToolbarModule,
    MatIconModule,
    MatButtonModule,
    LottieComponent,
    InputForm,
    Dashboard
  ],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App {
  results: any = null;
  loading = false;

  lottieOptions: AnimationOptions = {
    path: '/assets/animations/loading.json',
    loop: true,
    autoplay: true
  };

  onAnalysisComplete(event: any) {
    if (event === 'loading') {
      this.loading = true;
      this.results = null;
    } else {
      this.loading = false;
      this.results = event;
    }
  }
}
