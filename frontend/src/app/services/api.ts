import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { PredictionResult, ReportRequest } from '../interfaces/prediction.interface';

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private baseUrl = '/api'; // Proxy will handle this

  constructor(private http: HttpClient) { }

  predict(formData: FormData): Observable<PredictionResult> {
    return this.http.post<PredictionResult>(`${this.baseUrl}/predict`, formData).pipe(
      catchError(this.handleError)
    );
  }

  generateReport(data: ReportRequest): Observable<Blob> {
    return this.http.post(`${this.baseUrl}/report`, data, { responseType: 'blob' }).pipe(
      catchError(this.handleError)
    );
  }

  private handleError(error: HttpErrorResponse) {
    let errorMessage = 'An unknown error occurred';
    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = `Error: ${error.error.message}`;
    } else {
      // Server-side error
      errorMessage = `Error Code: ${error.status}\nMessage: ${error.message}`;
    }
    console.error('API Error:', errorMessage);
    return throwError(() => new Error(errorMessage));
  }
}
