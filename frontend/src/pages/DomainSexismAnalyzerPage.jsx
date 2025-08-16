import React, { useState } from 'react';
import { Form, Button, Card, Spinner, Alert, Row, Col } from 'react-bootstrap';
import api from '../functions/api';
import { useToast } from '../components/ToastProvider';
import '../styles/Common.css';


const DomainSexismAnalyzerPage = () => {
  const { showError } = useToast();
  const [domainAbs, setDomainAbs] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [collapsed, setCollapsed] = useState({}); // {index: boolean}

  const toggleCollapse = (idx) => {
    setCollapsed((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const token = localStorage.getItem('access_token');
      const response = await api.post(
        '/web-crawling/sitemap/get-urls',
        { domain: domainAbs }, // Ajusta la clave si tu backend espera otro nombre
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setResult(response.data);
      setCollapsed({}); // reset desplegables
    } catch (error) {
      console.error('Error analizando el dominio:', error);
      showError('No se pudo analizar el dominio');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="analyze-text-section">
      <h1 className="title">Analizador de Dominio</h1>

      <Form onSubmit={handleSubmit} className="mb-4">
        <Form.Group className="mb-3">
          <Form.Label>Introduce el dominio absoluto</Form.Label>
          <Form.Control
            type="url"
            placeholder="https://example.com"
            value={domainAbs}
            onChange={(e) => setDomainAbs(e.target.value)}
            required
          />
        </Form.Group>

        <Button type="submit" className="btn-dark" disabled={loading}>
          {loading ? <Spinner size="sm" animation="border" /> : 'Analizar Dominio'}
        </Button>
      </Form>

      {result && (
        <div className="analysis-results">
          {/* Mensaje de info + botón Ver resultados */}
          <Alert variant="info" className="d-flex justify-content-between align-items-center">
            <div>
              <strong>Analizando URLs del dominio</strong>
              <div className="small text-muted">{result.domain}</div>
            </div>
            <a
              href={`/app/analiticas/dominios/${result.id_domain}/urls`}
              className="btn-primary btn"
            >
              Ver resultados
            </a>
          </Alert>

          {/* Lista de sitemaps */}
          <Row>
            {result.sitemaps?.map((sm, idx) => {
              const isCollapsed = !!collapsed[idx];
              const urlCount = sm.urls?.length || 0;
              return (
                <Col xs={12} key={`${sm.url_sitemap}-${idx}`} className="mb-3">
                  <Card>
                    <Card.Header
                      as="div"
                      className="d-flex justify-content-between align-items-center cursor-pointer"
                      onClick={() => toggleCollapse(idx)}
                      style={{ userSelect: 'none' }}
                    >
                      <div className="d-flex flex-column">
                        <span className="fw-semibold">
                          Sitemap: <a href={sm.url_sitemap} target="_blank" rel="noopener noreferrer">{sm.url_sitemap}</a>
                        </span>
                        <small className="text-muted">{urlCount} URL{urlCount !== 1 ? 's' : ''}</small>
                      </div>
                      <Button
                        variant="link"
                        className="p-0 ms-2"
                        onClick={(e) => { e.stopPropagation(); toggleCollapse(idx); }}
                        aria-label={isCollapsed ? 'Mostrar URLs' : 'Ocultar URLs'}
                        title={isCollapsed ? 'Mostrar URLs' : 'Ocultar URLs'}
                      >
                        <i className={`bi ${isCollapsed ? 'bi-chevron-down' : 'bi-chevron-up'}`} />
                      </Button>
                    </Card.Header>

                    {!isCollapsed && (
                      <Card.Body style={{ maxHeight: 320, overflowY: 'auto' }}>
                        {urlCount === 0 ? (
                          <div className="text-muted">No se han encontrado URLs en este sitemap.</div>
                        ) : (
                          <ul className="list-group list-group-flush">
                            {sm.urls.map((u, i) => (
                              <li key={`${u}-${i}`} className="list-group-item d-flex justify-content-between align-items-center">
                                <a href={u} target="_blank" rel="noopener noreferrer" className="text-break">{u}</a>
                                <span className="badge bg-secondary">URL</span>
                              </li>
                            ))}
                          </ul>
                        )}
                      </Card.Body>
                    )}
                  </Card>
                </Col>
              );
            })}
          </Row>
        </div>
      )}
    </section>
  );
};

export default DomainSexismAnalyzerPage;
