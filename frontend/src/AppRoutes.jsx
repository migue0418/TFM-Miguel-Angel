import {
  Routes,
  Route,
  Navigate,
} from 'react-router-dom';

import LoginPage from './pages/LoginPage';
import HomePage from './pages/HomePage';
import AnalyticsHomePage from './pages/AnalyticsHomePage';
import SexismHomePage from './pages/SexismHomePage';
import { ProtectedRoute } from './functions/ProtectedRoute';
import AppLayout from './components/AppLayout';
import DomainsPage from './pages/DomainsPage';
import DomainUrlsPage from './pages/DomainUrlsPage';
import UrlDetailPage from './pages/UrlDetailPage';
import ProfilePage from './pages/ProfilePage';
import UsersManagement from './pages/UsersManagement';
import RolesManagement from './pages/RolesManagement';
import RoleGuard from './functions/RoleGuard';
import TextSexismAnalyzer from './pages/TextSexismAnalyzerPage';
import UrlSexismAnalyzerPage from './pages/UrlSexismAnalyzerPage';
import DomainSexismAnalyzerPage from './pages/DomainSexismAnalyzerPage';
import Dashboard from './pages/Dashboard';

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
                <Route path="analiticas" element={<AnalyticsHomePage />} />
                <Route path="analiticas/global" element={<Dashboard />} />
                <Route path="analiticas/dominios" element={<DomainsPage />} />
                <Route path="analiticas/dominios/:id_dominio/urls" element={<DomainUrlsPage />} />
                <Route path="analiticas/dominios/:id_dominio/urls/:id_url" element={<UrlDetailPage />} />

                <Route path="detector-sexismo" element={<SexismHomePage />} />
                <Route path="detector-sexismo/textos" element={<TextSexismAnalyzer />} />
                <Route path="detector-sexismo/urls" element={<UrlSexismAnalyzerPage />} />
                <Route path="detector-sexismo/dominios" element={<DomainSexismAnalyzerPage />} />
            </Route>

            {/* Página perfil */}
            <Route path="perfil" element={<ProfilePage />} />

            {/* Sólo puede entrar si tiene el rol admin */}
            <Route element={<RoleGuard allowedRoles="admin" />}>
                <Route path="admin" element={<UsersManagement />} />
                <Route path="admin/roles" element={<RolesManagement />} />
            </Route>
        </Route>

        {/* Cualquier otra ruta: al login */}
        <Route path="*" element={<Navigate to="/login" replace />} />
    </Routes>
    );
}
