import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatDividerModule } from '@angular/material/divider';

@Component({
    selector: 'app-settings-dialog',
    standalone: true,
    imports: [
        CommonModule,
        FormsModule,
        MatDialogModule,
        MatButtonModule,
        MatIconModule,
        MatSlideToggleModule,
        MatSelectModule,
        MatFormFieldModule,
        MatDividerModule
    ],
    templateUrl: './settings-dialog.html',
    styleUrls: ['./settings-dialog.css']
})
export class SettingsDialog {
    settings = {
        showReferenceRanges: true,
        autoExpandFields: false,
        includePatientId: true,
        highContrastHeatmaps: false
    };

    constructor(private dialogRef: MatDialogRef<SettingsDialog>) {
        // Load settings from localStorage if available
        const saved = localStorage.getItem('oa-settings');
        if (saved) {
            this.settings = { ...this.settings, ...JSON.parse(saved) };
        }
    }

    save(): void {
        localStorage.setItem('oa-settings', JSON.stringify(this.settings));
        this.dialogRef.close(this.settings);
    }

    close(): void {
        this.dialogRef.close();
    }
}
