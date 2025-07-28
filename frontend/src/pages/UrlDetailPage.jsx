import React, { useEffect, useState, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { Card, Form, Spinner, Dropdown } from 'react-bootstrap';
import { PieChart, Pie, Cell, ResponsiveContainer, Label } from 'recharts';
import api from '../functions/api';
import { useToast } from '../components/ToastProvider';
import '../styles/Common.css';

const UrlDetailPage = () => {
  const { id_url } = useParams();
  const { showError } = useToast();

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filter, setFilter] = useState('all');

  const fetchData = useCallback(async () => {
    try {
      const token = localStorage.getItem('access_token');
      const response = await api.get(`/web-crawling/urls/${id_url}/sexism`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setData(response.data);
    } catch (err) {
      console.error(err);
      showError('Error al cargar el contenido de la URL');
    } finally {
      setLoading(false);
    }
  }, [id_url, showError]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const getStatusStyle = (score) => {
    if (score > 0.65) return { color: 'red', fontWeight: 'bold' };
    if (score >= 0.35) return { color: '#555', fontWeight: 'bold' };
    return { color: 'green', fontWeight: 'bold' };
  };

  const getStatusText = (score) => (score > 0.5 ? 'Sexista' : 'No sexista');

  const filteredTexts =
    data?.texts.filter((t) => {
      const contentMatch = t.content.toLowerCase().includes(searchQuery.toLowerCase());
      if (filter === 'sexista') return contentMatch && t.sexist;
      if (filter === 'no-sexista') return contentMatch && !t.sexist;
      return contentMatch;
    }) || [];

  const renderSexismChart = (percentage) => {
    const chartData = [
      { name: 'Sexista', value: percentage },
      { name: 'No sexista', value: 100 - percentage },
    ];
    const COLORS = ['#dc3545', '#ccc'];

    return (
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie
            data={chartData}
            innerRadius={70}
            outerRadius={90}
            dataKey="value"
            startAngle={90}
            endAngle={-270}
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index]} />
            ))}
            <Label
              value={percentage >= 50 ? 'Sexista' : 'No Sexista'}
              position="center"
              style={{ fontSize: '1rem', fontWeight: 'bold' }}
            />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    );
  };

  return (
    <section className="url-detail-section">
      <h1 className="title">Análisis de la URL</h1>
      {loading ? (
        <p className="status-msg">
          <Spinner animation="border" size="sm" /> Cargando análisis…
        </p>
      ) : (
        data && (
          <>
            <Card className="mb-4">
              <Card.Body>
                <p>
                  <strong>URL:</strong>{' '}
                  <a href={data.absolute_url} target="_blank" rel="noopener noreferrer">
                    {data.absolute_url}
                  </a>
                </p>
                {renderSexismChart(data.global_score * 100)}
                <p className="mt-3">
                  Se ha detectado sexismo en{' '}
                  <strong>{(data.global_score * 100).toFixed(2)}%</strong> del contenido.
                  Esto corresponde a{' '}
                  <strong>{Math.round(data.global_score * data.total_sentences)}</strong> frases de un total de{' '}
                  <strong>{data.total_sentences}</strong>.
                </p>
              </Card.Body>
            </Card>

            <div className="d-flex gap-3 mb-3 align-items-center">
              <Form.Control
                type="text"
                placeholder="Buscar texto"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <Dropdown>
                <Dropdown.Toggle variant="outline-secondary" size="sm">
                  {filter === 'all' ? 'Todas' : filter === 'sexista' ? 'Solo Sexistas' : 'Solo No Sexistas'}
                </Dropdown.Toggle>
                <Dropdown.Menu>
                  <Dropdown.Item onClick={() => setFilter('all')}>Todas</Dropdown.Item>
                  <Dropdown.Item onClick={() => setFilter('sexista')}>Solo Sexistas</Dropdown.Item>
                  <Dropdown.Item onClick={() => setFilter('no-sexista')}>Solo No Sexistas</Dropdown.Item>
                </Dropdown.Menu>
              </Dropdown>
            </div>

            <Card>
              <Card.Body>
                {filteredTexts.length === 0 ? (
                  <p>No hay frases que coincidan.</p>
                ) : (
                  <ul className="list-group">
                    {filteredTexts.map((t, idx) => (
                      <li key={idx} className="list-group-item">
                        <p>
                          <strong>Texto:</strong> {t.content}
                        </p>
                        <p>
                          <strong>Predicción:</strong>{' '}
                          <span style={getStatusStyle(t.score_sexist)}>
                            {getStatusText(t.score_sexist)}
                          </span>
                        </p>
                        <p>
                          <strong>Probabilidad sexista:</strong>{' '}
                          {(t.score_sexist * 100).toFixed(2)}%
                        </p>
                        <p>
                          <strong>Probabilidad no sexista:</strong>{' '}
                          {(t.score_non_sexist * 100).toFixed(2)}%
                        </p>
                      </li>
                    ))}
                  </ul>
                )}
              </Card.Body>
            </Card>
          </>
        )
      )}
    </section>
  );
};

export default UrlDetailPage;
