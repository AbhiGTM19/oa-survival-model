import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private baseUrl = '/api'; // Proxy will handle this

  constructor(private http: HttpClient) { }

  predict(formData: FormData): Observable<any> {
    return this.http.post(`${this.baseUrl}/predict`, formData);
  }

  generateReport(data: any): Observable<Blob> {
    return this.http.post(`${this.baseUrl}/report`, data, { responseType: 'blob' });
  }
}
