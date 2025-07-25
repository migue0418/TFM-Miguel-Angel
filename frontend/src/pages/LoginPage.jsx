import React, { useContext, useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import AuthContext from '../functions/AuthContext';
import '../styles/LoginPage.css';
import LogoPrincipal from '../components/logos';
import { useToast } from '../components/ToastProvider';
import '../styles/Common.css';

const LoginPage = () => {
  const { login } = useContext(AuthContext);
  const navigate   = useNavigate();
  const location   = useLocation();

  const [username, setUsername]       = useState('');
  const [password, setPassword]       = useState('');
  // Toast para notificaciones
  const { showError } = useToast();

  /* ────────────────────────────────────────────────────────────────── */
  /*  Detectar sesión expirada mediante query param ?error=sessionExpired */
  /* ────────────────────────────────────────────────────────────────── */
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    if (params.get('error') === 'sessionExpired') {
      showError('Sesión expirada. Por favor, inicia sesión nuevamente.');
    }
  }, [location.search, showError]);

  /* ────────────────────────────────────────────────────────────────── */
  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      await login(username, password);
      navigate('/', {replace: true});
    } catch (error) {
      showError(error.message || 'Error de autenticación')
    }
  };
  /* ────────────────────────────────────────────────────────────────── */

  return (
    <div className="login-page">
      {/* Sección izquierda –––––––––––––––––––––––––––––––––––––––––– */}
      <section className="login-section">
        <article class="login-card" aria-labelledby="login-title">

        <h1 id="login-title">Iniciar sesión</h1>
        <form className="login-card" onSubmit={handleLogin} aria-label="Formulario de acceso">

          {/* <div class="input-group" style="margin-bottom: 1.25rem;"> */}
          <div class="input-group">
            <label for="username">Usuario</label>
            <div class="input-wrapper">
              <svg aria-hidden="true" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2"
                  stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="8" r="4"/>
                <path d="M4 20c0-4 4-6 8-6s8 2 8 6"/>
              </svg>
              <input
                type="text"
                id="username"
                name="username"
                placeholder="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>
          </div>


          {/* <div class="input-group" style="margin-bottom: 1.75rem;"> */}
          <div class="input-group mb-3">
            <label for="password">Contraseña</label>
            <div class="input-wrapper">
              <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="11" width="18" height="10" rx="2" ry="2"/><path d="M7 11v-3a5 5 0 0 1 10 0v3"/>
              </svg>
              <input
                type="password"
                id="password"
                name="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required minlength="6" />
            </div>
          </div>

          <button type="submit" className="btn-submit mt-4">Acceder</button>
        </form>
        </article>
      </section>

      {/* Sección derecha ––––––––––––––––––––––––––––––––––––––––––– */}
      <aside class="brand-section" aria-label="Información de marca">
        <div className='logo-box'>
          <LogoPrincipal className='logo' size={0.6} color='var(--color-bg-text)'/>
        </div>
        <div class="words" aria-hidden="true">
          <span>LLMs para detección automática</span>
          <span>de lenguaje sexista</span>
          <span>en redes sociales</span>
        </div>
      </aside>
    </div>
  );
};

export default LoginPage;
