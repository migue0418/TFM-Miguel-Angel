// index.jsx o main.jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import AppRoutes from './AppRoutes';
import { AuthProvider } from './functions/AuthContext';
import { ToastProvider } from './components/ToastProvider';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap-icons/font/bootstrap-icons.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <AuthProvider>
    <ToastProvider>
      <BrowserRouter basename="/app">
        <AppRoutes />
      </BrowserRouter>
    </ToastProvider>
  </AuthProvider>
);
