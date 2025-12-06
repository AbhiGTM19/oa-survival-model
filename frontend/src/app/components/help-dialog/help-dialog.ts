import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatDividerModule } from '@angular/material/divider';

@Component({
    selector: 'app-help-dialog',
    standalone: true,
    imports: [
        CommonModule,
        MatDialogModule,
        MatButtonModule,
        MatIconModule,
        MatDividerModule
    ],
    templateUrl: './help-dialog.html',
    styleUrls: ['./help-dialog.css']
})
export class HelpDialog {
    constructor(private dialogRef: MatDialogRef<HelpDialog>) { }

    close(): void {
        this.dialogRef.close();
    }
}
