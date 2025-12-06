import { Component, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ApiService } from '../../services/api';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatIconModule } from '@angular/material/icon';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { PredictionResult } from '../../interfaces/prediction.interface';

/** Reference ranges for biomarkers (used for hints and validation) */
export const BIOMARKER_RANGES = {
  COMP: { min: 0, max: 3000, normal: '< 1200 ng/mL' },
  CTX: { min: 0, max: 1000, normal: '< 400 ng/mmol' },
  HA: { min: 0, max: 200, normal: '20-60 ng/mL' },
  C2C: { min: 0, max: 500, normal: '< 200 ng/mL' },
  CPII: { min: 0, max: 1000, normal: '200-600 ng/mL' },
};

@Component({
  selector: 'app-input-form',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatInputModule,
    MatSelectModule,
    MatButtonModule,
    MatExpansionModule,
    MatIconModule,
    MatFormFieldModule,
    MatProgressBarModule
  ],
  templateUrl: './input-form.html',
  styleUrls: ['./input-form.css']
})
export class InputForm {
  @Output() analysisComplete = new EventEmitter<PredictionResult | 'loading' | null>();
  form: FormGroup;
  selectedFile: File | null = null;
  loading = false;

  /** Expose biomarker ranges to template */
  biomarkerRanges = BIOMARKER_RANGES;

  constructor(private fb: FormBuilder, private api: ApiService) {
    this.form = this.fb.group({
      patientId: [''],
      age: [65, [Validators.required, Validators.min(18), Validators.max(120)]],
      sex: ['Female', Validators.required],
      bmi: [28.5, [Validators.required, Validators.min(10), Validators.max(60)]],
      kl_grade: [2, Validators.required],
      womac: [45, [Validators.required, Validators.min(0), Validators.max(96)]],
      pase: [120, [Validators.min(0), Validators.max(500)]],
      koos: [60, [Validators.min(0), Validators.max(100)]],
      stiffness: [35, [Validators.min(0), Validators.max(100)]],
      nsaid: ['No'],
      bio_comp: [1200.5, [Validators.min(0), Validators.max(3000)]],
      bio_ctx: [0.45, [Validators.min(0), Validators.max(1000)]],
      bio_ha: [45.2, [Validators.min(0), Validators.max(200)]],
      bio_c2c: [110.0, [Validators.min(0), Validators.max(500)]],
      bio_cpii: [350.0, [Validators.min(0), Validators.max(1000)]],
      mri_bml: [1.5, [Validators.min(0), Validators.max(10)]],
      mri_cyst: [0.5, [Validators.min(0), Validators.max(10)]]
    });
  }

  /** Get validation error message for a form control */
  getErrorMessage(controlName: string): string {
    const control = this.form.get(controlName);
    if (!control || !control.errors) return '';

    if (control.errors['required']) return 'This field is required';
    if (control.errors['min']) return `Minimum value is ${control.errors['min'].min}`;
    if (control.errors['max']) return `Maximum value is ${control.errors['max'].max}`;
    return 'Invalid value';
  }

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
    }
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    if (event.dataTransfer?.files.length) {
      this.selectedFile = event.dataTransfer.files[0];
    }
  }

  onSubmit() {
    // Mark all fields as touched to show validation errors
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }

    if (!this.selectedFile) {
      alert('Please upload an X-Ray image');
      return;
    }

    this.loading = true;
    this.analysisComplete.emit('loading');

    const formData = new FormData();
    Object.keys(this.form.value).forEach(key => {
      formData.append(key, this.form.value[key]);
    });
    formData.append('file', this.selectedFile);

    this.api.predict(formData).subscribe({
      next: (res) => {
        this.loading = false;
        this.analysisComplete.emit(res);
      },
      error: (err) => {
        console.error(err);
        this.loading = false;
        alert('Prediction failed. Check console.');
        this.analysisComplete.emit(null);
      }
    });
  }
}
