import React, { useState } from 'react';
import { Form, Button, Card, Spinner } from 'react-bootstrap';
import { PieChart, Pie, Cell, ResponsiveContainer, Label } from 'recharts';
import api from '../functions/api';
import { useToast } from '../components/ToastProvider';
import '../styles/Common.css';

const TextSexismAnalyzer = () => {
  const { showError } = useToast();
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const token = localStorage.getItem('access_token');
      const response = await api.post(
        '/web-crawling/text/check-sexism',
        { text },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      setResult(response.data);
    } catch (error) {
      console.error('Error analizando el texto:', error);
      showError('No se pudo analizar el texto');
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

  const renderSexismChart = (percentage) => {
    const data = [
      { name: 'Sexista', value: percentage },
      { name: 'No sexista', value: 100 - percentage },
    ];
    const COLORS = ['#dc3545', '#ccc'];

    return (
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie
            data={data}
            innerRadius={60}
            outerRadius={80}
            dataKey="value"
            startAngle={90}
            endAngle={-270}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index]} />
            ))}
            <Label
              value={percentage >= 50 ? 'Sexista' : 'No Sexista'}
              position="center"
              style={{ fontSize: '1.2rem', fontWeight: 'bold' }}
            />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    );
  };

  return (
    <section className="analyze-text-section">
      <h1 className="title">Detector de Sexismo en Texto</h1>

      <Form onSubmit={handleSubmit} className="mb-4">
        <Form.Group className="mb-3">
          <Form.Label>Introduce el texto completo</Form.Label>
          <Form.Control
            as="textarea"
            rows={6}
            value={text}
            onChange={(e) => setText(e.target.value)}
            required
          />
        </Form.Group>
        <Button type="submit" className="btn-dark" disabled={loading}>
          {loading ? <Spinner size="sm" animation="border" /> : 'Analizar Texto'}
        </Button>
      </Form>

      {result && (
        <div className="analysis-results">
          <Card className="mb-4">
            <Card.Body>
              <h5>Análisis Global</h5>
              {renderSexismChart(result.global.sexism_percentage)}
              <p className="mt-3">
                Se ha detectado sexismo en <strong>{result.global.sexism_percentage.toFixed(1)}%</strong> del texto.
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

export default TextSexismAnalyzer;
