import React, { useEffect, useState } from 'react';
import { Row, Col, Card, Spinner } from 'react-bootstrap';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import api from '../functions/api';
import { useToast } from '../components/ToastProvider';
import SexismPieChart from '../components/SexismPieChart';

const Dashboard = () => {
  const { showError } = useToast();

  const [loading, setLoading] = useState(true);
  const [overview, setOverview] = useState(null);
  const [topSentences, setTopSentences] = useState([]);
  const [severity, setSeverity] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const token = localStorage.getItem('access_token');
        const [ovRes, topRes, sevRes] = await Promise.all([
          api.get('/web-crawling/analytics/overview', { headers: { Authorization: `Bearer ${token}` } }),
          api.get('/web-crawling/analytics/top-sexist-sentences?limit=5', { headers: { Authorization: `Bearer ${token}` } }),
          api.get('/web-crawling/analytics/severity-distribution', { headers: { Authorization: `Bearer ${token}` } }),
        ]);

        setOverview(ovRes.data);
        setTopSentences(topRes.data.sentences);
        setSeverity(sevRes.data.bins);
      } catch (err) {
        console.error('Error cargando analíticas:', err);
        showError('No se pudieron cargar las analíticas globales');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [showError]);

  if (loading) {
    return (
      <div className="d-flex justify-content-center align-items-center" style={{ height: '60vh' }}>
        <Spinner animation="border" />
      </div>
    );
  }

  if (!overview) return null;

  return (
    <section className="dashboard-section">
      <h1 className="title">Dashboard de Analíticas</h1>

      {/* Métricas Globales */}
      <Card className="mb-4">
        <Card.Body style={{ minHeight: 220 }}>
          <Row className="h-100">
            <Col md={4} className="text-center border-end d-flex flex-column justify-content-center align-items-center">
              <h5 className="mb-2">URLs analizadas</h5>
              <p className="display-6 m-0">{overview.total_urls}</p>
            </Col>

            <Col md={4} className="text-center border-end d-flex flex-column justify-content-center align-items-center">
              <h5 className="mb-2">Frases analizadas</h5>
              <p className="display-6 m-0">{overview.total_sentences}</p>
              <small className="text-muted mt-2">Frases con sexismo</small>
              <div style={{ fontSize: 20, fontWeight: 600 }}>{overview.sexist_sentences}</div>
            </Col>

            <Col
              md={4}
              className="text-center d-flex flex-column justify-content-center align-items-center"
            >
              <h5 className="mb-3">% Sexismo Global</h5>
              <div className="w-100" style={{ height: 200, maxWidth: 260 }}>
                <SexismPieChart
                  percentage={overview.global_sexism_percentage} // 0.44 => 0.44%
                  isFraction={false}                              // <— clave
                  height={200}
                />
              </div>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      <Row>
        {/* Top 5 frases sexistas */}
        <Col lg={6} className="mb-4">
          <Card style={{ height: '100%' }}>
            <Card.Body className="d-flex flex-column">
              <h5 className="mb-3">Top 5 frases sexistas</h5>
              <ul className="list-group list-group-flush" style={{ maxHeight: 320, overflowY: 'auto' }}>
                {topSentences.map((s, idx) => (
                  <li key={idx} className="list-group-item">
                    <p className="mb-1"><strong>Score:</strong> {(s.score_sexist * 100).toFixed(2)}%</p>
                    <p className="mb-0">{s.text}</p>
                  </li>
                ))}
              </ul>
            </Card.Body>
          </Card>
        </Col>

        {/* Distribución de severidad */}
        <Col lg={6} className="mb-4">
          <Card style={{ height: '100%' }}>
            <Card.Body>
              <h5 className="mb-3">Distribución de Severidad</h5>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={severity} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                  <XAxis dataKey="range" />
                  <YAxis allowDecimals={false} />
                  <Tooltip formatter={(value) => [value, 'Frases']} />
                  <Bar dataKey="count" />
                </BarChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </section>
  );
};

export default Dashboard;
