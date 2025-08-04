// src/pages/SexismHomePage.jsx
import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Row, Col } from 'react-bootstrap';
import { getRoles } from '../functions/authUtils';
import { useToast } from '../components/ToastProvider';
import '../styles/HomeDashboard.css';

export default function SexismHomePage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { showError } = useToast();

  const roles = getRoles() ?? [];

  /** Tarjetas visibles según rol */
  const tiles = [
    {
        title: 'Textos',
        text: 'Analiza un texto y detecta lenguaje sexista.',
        icon: 'bi-file-text',
        to: 'textos',
        role: 'sexism_detection'
    },
    {
        title: 'URLs',
        text: 'Analiza una URL y detecta lenguaje sexista.',
        icon: 'bi-filetype-html',
        to: 'urls',
        role: 'sexism_detection'
    },
    {
        title: 'Dominios',
        text: 'Analiza un dominio y detecta lenguaje sexista.',
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
      <h1 className="title mb-4">Detector de Sexismo</h1>

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
