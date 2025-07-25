import {
  Routes,
  Route,
  Navigate,
} from 'react-router-dom';

import LoginPage from './pages/LoginPage';
import HomePage from './pages/HomePage';
import { ProtectedRoute } from './functions/ProtectedRoute';
import AppLayout from './components/AppLayout';
import AnalyticsPage  from './pages/AnalyticsPage';
import ProfilePage from './pages/ProfilePage';
import UsersManagement from './pages/UsersManagement';
import RolesManagement from './pages/RolesManagement';
import RoleGuard from './functions/RoleGuard';

export default function AppRoutes() {
    return (
    <Routes>
        {/* 1) Login fuera del layout  ---------------------------------------- */}
        <Route path="/login" element={<LoginPage />} />

        {/* 2) Zona protegida bajo /  (basename aporta “/app”)  --------------- */}
        <Route
            element={
                <ProtectedRoute>
                    <AppLayout />
                </ProtectedRoute>
            }
        >
            {/* Landing: /app */}
            <Route index element={<HomePage />} />

            {/* Páginas encuestas */}
            <Route element={<RoleGuard allowedRoles={['sexism_detection', 'admin']} />}>
                <Route path="analiticas" element={<AnalyticsPage />} />
            </Route>

            {/* Página perfil */}
            <Route path="perfil" element={<ProfilePage />} />

            {/* Sólo puede entrar si tiene el rol admin */}
            <Route element={<RoleGuard allowedRoles="admin" />}>
                <Route path="admin"           element={<UsersManagement />} />
                <Route path="admin/roles"     element={<RolesManagement />} />
            </Route>
        </Route>

        {/* Cualquier otra ruta: al login */}
        <Route path="*" element={<Navigate to="/login" replace />} />
    </Routes>
    );
}
