import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000", // URL relativa al proyecto backend que es el mismo
  withCredentials: true, // Para enviar cookies si es necesario
});

api.interceptors.response.use(
  response => response,
  async error => {
    if (error.response && error.response.status === 401) {
      // Por ejemplo, eliminar el token y redirigir al login
      localStorage.removeItem("access_token");
      localStorage.removeItem("username");
      window.location.href = '/app/login?error=sessionExpired';
    }
    return Promise.reject(error);
  }
);


export default api;
