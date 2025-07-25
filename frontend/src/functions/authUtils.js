// authUtils.js
import api from './api';
import { jwtDecode } from "jwt-decode";

// Función para obtener el token (se ajusta la URL a /auth/token)
export async function obtenerToken(username, password) {
    const params = new URLSearchParams();
    params.append("grant_type", "password");
    params.append("username", username);
    params.append("password", password);

    try {
        const response = await api.post('/auth/token', params, {
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json"
            },
        });

        if (response.status === 200) {

            // Guardamos el token
            localStorage.setItem("access_token", response.data.access_token);

            // Guardamos el nombre de usuario
            localStorage.setItem("username", username);

            return response.data.access_token;
        } else {
            // Si el código no es 200, lanza un error con el detalle devuelto por FastAPI
            throw new Error(response.data.detail);
        }
    } catch (error) {
        console.error("Error al obtener el token:", error);
        throw new Error(error.message || "Error al obtener el token");
    }
}

// Función para verificar si el token ha expirado
export function esTokenExpirado(token) {
    try {
      // Elimina espacios y saltos de línea
      const cleanToken = token.trim();
      // Verifica que tenga tres partes separadas por puntos
      if (cleanToken.split('.').length !== 3) {
        console.error("Token mal formado");
        return true;
      }

      const decoded = jwtDecode(cleanToken);

      if (!decoded.exp) return false;

      const expirationTime = decoded.exp * 1000; // exp está en segundos

      return Date.now() > expirationTime;
    } catch (error) {
      console.error("Error al decodificar el token:", error);
      return true;
    }
  }

// Lee roles del token (ya tenías getRoles)
export function getRoles() {
    const token = localStorage.getItem('access_token');
    if (!token) return [];
    try {
        return jwtDecode(token)?.roles ?? [];
    } catch {
        return [];
    }
};


// Función para saber si tiene el rol admin
export function isAdmin(token) {
    try {
        // Obtenemos los roles
        const roles = getRoles(token);

        return Array.isArray(roles) && roles.includes('admin');
    } catch (error) {
        return false;
    }
}
