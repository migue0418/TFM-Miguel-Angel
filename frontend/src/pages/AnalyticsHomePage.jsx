// src/pages/AnalyticsHomePage.jsx
import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Row, Col } from 'react-bootstrap';
import { getRoles } from '../functions/authUtils';
import { useToast } from '../components/ToastProvider';
import '../styles/HomeDashboard.css';

export default function AnalyticsHomePage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { showError } = useToast();

  const roles = getRoles() ?? [];

  /** Tarjetas visibles según rol */
  const tiles = [
    {
        title: 'Global',
        text: 'Obtiene las analíticas globales de los dominios y URLs.',
        icon: 'bi-bar-chart-line',
        to: 'global',
        role: 'sexism_detection'
    },
    {
        title: 'Gestión de Dominios',
        text: 'Gestión de dominios y URLs.',
        icon: 'bi-globe',
        to: 'dominios',
        role: 'sexism_detection'
    }
  ];

  const visibles = tiles.filter(t =>
    roles.includes('admin') || t.role === 'all' || roles.includes(t.role)
  );

  /* mensaje “noAccess” */
  useEffect(() => {
    if (new URLSearchParams(location.search).get('error') === 'noAccess') {
      showError('No tienes acceso a esa página.');
    }
  }, [location, showError]);

  return (
    <section className="container py-4">
      <h1 className="title mb-4">Analíticas</h1>

      <Row xs={1} md={2} lg={3} className="g-4">
        {visibles.map(({ title, text, icon, to }) => (
          <Col key={title}>
            <button className="dash-card neutral" onClick={() => navigate(to)}>
              <div className="dash-icon-wrapper neutral">
                <i className={`bi ${icon}`} />
              </div>
              <div className="dash-content">
                <h3>{title}</h3>
                <p>{text}</p>
              </div>
            </button>
          </Col>
        ))}
      </Row>
    </section>
  );
}
