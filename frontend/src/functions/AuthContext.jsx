import React, { createContext, useState } from 'react';
import { obtenerToken } from './authUtils';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  // Carga el token almacenado en localStorage (si existe)
  const storedToken = localStorage.getItem("access_token");
  const [token, setToken] = useState(storedToken);

  // Función de login que utiliza la función obtenerToken
  const login = async (username, password) => {
    const newToken = await obtenerToken(username, password);
    setToken(newToken);
    return newToken;
  };

  // Función de logout que limpia el token
  const logout = () => {
    setToken(null);
    localStorage.removeItem("access_token");
    localStorage.removeItem("username");
  };

  return (
    <AuthContext.Provider value={{ token, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
