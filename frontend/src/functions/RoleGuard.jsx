// src/functions/RoleGuard.jsx
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { getRoles } from './authUtils';


/**
 * @param {string|string[]} allowedRoles – rol o array de roles permitidos
 */
export default function RoleGuard({ allowedRoles }) {
  const location = useLocation();
  const roles = getRoles();

  const hasAccess = roles.includes('admin') ||
                    (Array.isArray(allowedRoles)
                      ? allowedRoles.some(r => roles.includes(r))
                      : roles.includes(allowedRoles));

  if (!hasAccess) {
    return <Navigate to="/?error=noAccess" state={{ from: location }} replace />;
  }
  return <Outlet />;
}
