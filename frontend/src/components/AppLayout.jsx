import React, { useState, useContext, useRef, useEffect, useCallback } from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import AuthContext from '../functions/AuthContext';
import '../styles/AppLayout.css';
import '../styles/Common.css';
import LogoPrincipal from '../components/logos';
import { getRoles } from '../functions/authUtils';
import { useToast } from './ToastProvider';

const AppLayout = () => {
    const { logout } = useContext(AuthContext);
    const navigate   = useNavigate();
    const [adminPrivileges, setAdminPrivileges] = useState(false);
    const [SexismDetectionPrivileges, setSexismDetectionPrivileges] = useState(false);
    const [open, setOpen] = useState(false);
    const menuRef = useRef(null);

    // Toast para notificaciones
    const { showError } = useToast();

    /* ── helpers ───────────────────────────────────────────────────────────── */
    const toggleMenu  = () => setOpen(!open);
    const closeMenu   = () => setOpen(false);
    const goProfile   = () => { closeMenu(); navigate('/perfil'); };
    const handleLogout = () => { closeMenu(); logout(); navigate('/'); };

    const fetchRoles = useCallback(async () => {
      try {
        const roles = getRoles();

        setAdminPrivileges(Array.isArray(roles) && roles.includes('admin'));
        setSexismDetectionPrivileges(Array.isArray(roles) && (roles.includes('admin') || roles.includes('sexism_detection')));

      } catch (error) {
          setAdminPrivileges(false);
          setSexismDetectionPrivileges(false);
          showError('Error al cargar los roles');
      }
    }, [setAdminPrivileges, setSexismDetectionPrivileges, showError]);

    useEffect(() => {
      fetchRoles();
      const handleClickOutside = (e) => {
          if (menuRef.current && !menuRef.current.contains(e.target)) closeMenu();
      };
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [fetchRoles]);

    const [showAdminSub, setShowAdminSub] = useState(false);

    /* helpers para los sub-menú */
    const openAdminSub  = () => setShowAdminSub(true);
    const closeAdminSub = () => setShowAdminSub(false);

    /* ── UI ─────────────────────────────────────────────────────────────────── */
    return (
    <>
        <div id="FullPageWrapper">
            <header className="app-header" id="TopNavigation">
                <nav className="nav-container">
                    {/* --------- LOGO --------- */}
                    <a href='/app'><LogoPrincipal size={0.25} color='var(--color-primary)'/></a>

                    {/* --------- ENLACES PRINCIPALES --------- */}
                    <ul className="nav-links">
                        {SexismDetectionPrivileges ? (
                          <li><NavLink to="/analiticas"  className="nav-link">Analíticas</NavLink></li>
                        ): null }

                        {adminPrivileges ? (
                          <li
                          className="nav-item nav-dropdown"
                          onMouseEnter={openAdminSub}
                          onMouseLeave={closeAdminSub}
                        >
                          <NavLink to="/admin"  className="nav-link">
                            Panel Administración ▾
                          </NavLink>

                          {showAdminSub && (
                            <ul className="sub-menu">
                              <li>
                                <NavLink
                                  to="/admin/roles"
                                  className="nav-sublink"
                                  onClick={closeAdminSub}
                                >
                                  Roles
                                </NavLink>
                              </li>
                            </ul>
                          )}
                          </li>
                        ): null}
                    </ul>

                    {/* --------- MENÚ DE USUARIO --------- */}
                    <div className="user-wrapper"  ref={menuRef}>
                        <button className="icon-btn" onClick={toggleMenu} aria-label="Menú de usuario">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                                strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="8" r="4" />
                            <path d="M4 20c0-4 4-6 8-6s8 2 8 6" />
                            </svg>
                        </button>

                        {open && (
                            <ul className="user-dropdown">
                            <li><button onClick={goProfile}>Perfil</button></li>
                            <li><button onClick={handleLogout}>Cerrar sesión</button></li>
                            </ul>
                        )}
                    </div>
                </nav>
            </header>

            {/* --------- CONTENIDO DINÁMICO --------- */}
            <main className="app-main">
                <Outlet />
            </main>
        </div>
    </>
    );
};

export default AppLayout;
