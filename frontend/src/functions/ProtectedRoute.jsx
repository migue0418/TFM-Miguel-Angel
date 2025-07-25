import React, { useContext } from 'react';
import { Navigate } from 'react-router-dom';
import AuthContext from './AuthContext';
import { esTokenExpirado, isAdmin } from '../functions/authUtils';

export const ProtectedRoute = ({ children }) => {
  const { token } = useContext(AuthContext);
  const storedToken = localStorage.getItem('access_token');
  const validToken  = token || storedToken;

  if (!validToken || esTokenExpirado(validToken)) {
    return <Navigate to="/login" replace />;
  }
  return children;
};

export const ProtectedAdminRoute = ({ children }) => {
  const { token } = useContext(AuthContext);
  const storedToken = localStorage.getItem('access_token');
  const validToken  = token || storedToken;

  if (!validToken || esTokenExpirado(validToken)) {
    return <Navigate to="/login" replace />;
  }
  if (!isAdmin(validToken)) {
    return <Navigate to="/app/encuestas" replace />;
  }
  return children;
};
