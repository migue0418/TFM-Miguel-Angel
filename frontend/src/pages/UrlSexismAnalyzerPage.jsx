import React, { useState } from 'react';
import { Form, Button, Card, Spinner } from 'react-bootstrap';
import SexismPieChart from '../components/SexismPieChart';
import api from '../functions/api';
import { useToast } from '../components/ToastProvider';
import '../styles/Common.css';


const UrlSexismAnalyzerPage = () => {
  const { showError } = useToast();
  const [url, setUrl] = useState('');
  const [filterTag, setFilterTag] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const token = localStorage.getItem('access_token');
      const response = await api.post(
        '/web-crawling/url/check-sexism',
        {
          url,
          filter_tag: filterTag || null,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      setResult(response.data);
    } catch (error) {
      console.error('Error analizando la URL:', error);
      showError('No se pudo analizar la URL');
    } finally {
      setLoading(false);
    }
  };

  const getPredictionStyle = (pred, scoreSexist) => {
    if (pred === 'not sexist' && scoreSexist < 0.35) return { color: 'green', fontWeight: 'bold' };
    if (scoreSexist >= 0.35 && scoreSexist <= 0.65) return { color: '#555', fontWeight: 'bold' };
    if (pred === 'sexist' && scoreSexist > 0.65) return { color: 'red', fontWeight: 'bold' };
    return { fontWeight: 'bold' };
  };

  return (
    <section className="analyze-text-section">
      <h1 className="title">Detector de Sexismo en URL</h1>

      <Form onSubmit={handleSubmit} className="mb-4">
        <Form.Group className="mb-3">
          <Form.Label>Introduce la URL absoluta</Form.Label>
          <Form.Control
            type="url"
            placeholder="https://ejemplo.com/articulo"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            required
          />
        </Form.Group>

        <Form.Group className="mb-3">
          <Form.Label>Etiqueta HTML (opcional)</Form.Label>
          <Form.Control
            type="text"
            placeholder="article, p, div…"
            value={filterTag}
            onChange={(e) => setFilterTag(e.target.value)}
          />
        </Form.Group>

        <Button type="submit" className="btn-dark" disabled={loading}>
          {loading ? <Spinner size="sm" animation="border" /> : 'Analizar URL'}
        </Button>
      </Form>

      {result && (
        <div className="analysis-results">
          <Card className="mb-4">
            <Card.Body>
              <h5>Análisis Global</h5>
              {SexismPieChart(result.global.sexism_percentage)}
              <p className="mt-3">
                Se ha detectado <strong><span className='red'>sexismo</span></strong> en <strong>{result.global.sexism_percentage.toFixed(1)}%</strong> del contenido.
                Esto corresponde a <strong>{Math.round((result.global.sexism_percentage / 100) * result.global.total_texts)}</strong> frases de un total de <strong>{result.global.total_texts}</strong>.
              </p>
            </Card.Body>
          </Card>

          <Card>
            <Card.Body>
              <h5>Análisis por Frase</h5>
              <ul className="list-group">
                {result.texts.map((sentence, idx) => (
                  <li key={idx} className="list-group-item">
                    <p><strong>Texto:</strong> {sentence.text}</p>
                    <p>
                      <strong>Predicción:</strong>{' '}
                      <span style={getPredictionStyle(sentence.pred, sentence.score_sexist)}>
                        {sentence.pred}
                      </span>
                    </p>
                    <p><strong>Probabilidad sexista:</strong> {(sentence.score_sexist * 100).toFixed(2)}%</p>
                    <p><strong>Probabilidad no sexista:</strong> {(sentence.score_not_sexist * 100).toFixed(2)}%</p>
                  </li>
                ))}
              </ul>
            </Card.Body>
          </Card>
        </div>
      )}
    </section>
  );
};

export default UrlSexismAnalyzerPage;
