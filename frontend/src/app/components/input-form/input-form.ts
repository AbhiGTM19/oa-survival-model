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
  @Output() analysisComplete = new EventEmitter<any>();
  form: FormGroup;
  selectedFile: File | null = null;
  loading = false;

  constructor(private fb: FormBuilder, private api: ApiService) {
    this.form = this.fb.group({
      patientId: [''],
      age: [65, Validators.required],
      sex: ['Female', Validators.required],
      bmi: [28.5, Validators.required],
      kl_grade: [2, Validators.required],
      womac: [45, Validators.required],
      pase: [120],
      koos: [60],
      stiffness: [35],
      nsaid: ['No'],
      bio_comp: [1200.5],
      bio_ctx: [0.45],
      bio_ha: [45.2],
      bio_c2c: [110.0],
      bio_cpii: [350.0],
      mri_bml: [1.5],
      mri_cyst: [0.5]
    });
  }

  onFileSelected(event: any) {
    this.selectedFile = event.target.files[0];
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
    if (this.form.valid && this.selectedFile) {
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
}
