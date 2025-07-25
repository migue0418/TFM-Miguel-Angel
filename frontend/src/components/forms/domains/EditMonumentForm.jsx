import React, { useEffect, useMemo, useState } from 'react';
import { Modal, Form, Button, Spinner, Row, Col } from 'react-bootstrap';
import api from '../../../functions/api';
import '../../../styles/Forms.css';
import { useToast } from '../../ToastProvider';


const EditMonumentForm = ({
  data,
  show,
  handleClose,
  onUpdated,
  cities = [],
  countries = [],
  loadingCatalog
}) => {
  const { showSuccess, showError } = useToast();

  const INITIAL_FORM = {
    id_monument: '',
    id_country: '',
    id_city: '',
    name: '',
    summary: '',
    url: '',
    url_maps: '',
    latitud: '',
    longitud: ''
  };

  const [formData, setFormData] = useState(INITIAL_FORM);
  const [submitting, setSubmitting] = useState(false);

  /* -------------------- cargar datos entrantes -------------------- */
  useEffect(() => {
    if (data && show) {
      // Aseguramos que id_country exista; si no viene en data, lo inferimos desde la ciudad.
      const countryFromCity = cities.find(c => c.id_city === data.id_city)?.id_country ?? '';
      setFormData({
        ...INITIAL_FORM,
        ...data,
        id_country: String(data.id_country ?? countryFromCity ?? '')
      });
    }
  }, [data, show, cities]);

  /* -------------------- selects dinámicos -------------------- */
  const filteredCities = useMemo(
    () => cities.filter(c => String(c.id_country) === formData.id_country),
    [cities, formData.id_country]
  );

  const handleChange = ({ target: { name, value } }) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  /* -------------------- validación de URLs -------------------- */
  const checkUrl = async url => {
    if (!url) return true;
    try {
      const res = await fetch(url, { method: 'HEAD', mode: 'no-cors' });
      return res.ok || res.type === 'opaque';
    } catch {
      return false;
    }
  };

  /* -------------------- submit -------------------- */
  const handleSubmit = async e => {
    e.preventDefault();

    for (const key of ['url', 'url_maps']) {
      if (!(await checkUrl(formData[key]))) {
        showError(`La dirección indicada en "${key}" no responde (código diferente de 200).`);
        return;
      }
    }

    setSubmitting(true);
    try {
      const token = localStorage.getItem('access_token');
      const { data: updated } = await api.post(
        '/tourist-guide/management/monuments/edit',
        formData,
        { headers: { Authorization: `Bearer ${token}` } }
      );

      onUpdated(updated);
      showSuccess('Monumento actualizado correctamente');
      handleClose();
    } catch (err) {
      console.error(err);
      showError('Error al actualizar el monumento');
    } finally {
      setSubmitting(false);
    }
  };

  /* -------------------- render -------------------- */
  return (
    <Modal
      show={show}
      onHide={handleClose}
      backdrop="static"
      centered
      dialogClassName="basic-modal"
    >
      <Modal.Header closeButton>
        <Modal.Title>Actualizar Información de Monumento</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Form onSubmit={handleSubmit}>
          {/* Fila 1 : name */}
          <Form.Group className="mb-3">
            <Form.Label>Nombre</Form.Label>
            <Form.Control
              type="text"
              name="name"
              value={formData.name}
              onChange={handleChange}
              required
            />
          </Form.Group>

          {/* Fila 2 : país + ciudad */}
          <Row className="mb-3">
            <Form.Group as={Col} md={6}>
              <Form.Label>País</Form.Label>
              <Form.Select
                name="id_country"
                value={formData.id_country}
                onChange={e =>
                  setFormData(prev => ({
                    ...prev,
                    id_country: e.target.value,
                    id_city: '' // reiniciar ciudad al cambiar país
                  }))
                }
                disabled={loadingCatalog}
                required
              >
                <option value="">Selecciona…</option>
                {countries.map(c => (
                  <option key={c.id_country} value={c.id_country}>
                    {c.name}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>

            <Form.Group as={Col} md={6}>
              <Form.Label>Ciudad</Form.Label>
              <Form.Select
                name="id_city"
                value={formData.id_city}
                onChange={handleChange}
                disabled={!formData.id_country || loadingCatalog}
                required
              >
                <option value="">Selecciona…</option>
                {filteredCities.map(c => (
                  <option key={c.id_city} value={c.id_city}>
                    {c.name}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
          </Row>

          {/* Fila 3 : summary */}
          <Form.Group className="mb-3">
            <Form.Label>Descripción</Form.Label>
            <Form.Control
              as="textarea"
              rows={2}
              name="summary"
              value={formData.summary}
              onChange={handleChange}
            />
          </Form.Group>

          {/* Fila 4 : url */}
          <Form.Group className="mb-3">
            <Form.Label>URL</Form.Label>
            <Form.Control
              type="url"
              name="url"
              value={formData.url}
              onChange={handleChange}
              placeholder="https://…"
            />
          </Form.Group>

          {/* Fila 5 : url_maps */}
          <Form.Group className="mb-3">
            <Form.Label>URL Google Maps</Form.Label>
            <Form.Control
              type="url"
              name="url_maps"
              value={formData.url_maps}
              onChange={handleChange}
              placeholder="https://maps.google.com/…"
            />
          </Form.Group>

          {/* Fila 6 : latitud + longitud */}
          <Row className="mb-4">
            <Form.Group as={Col} md={6}>
              <Form.Label>Latitud</Form.Label>
              <Form.Control
                type="number"
                step="any"
                name="latitud"
                value={formData.latitud}
                onChange={handleChange}
              />
            </Form.Group>
            <Form.Group as={Col} md={6}>
              <Form.Label>Longitud</Form.Label>
              <Form.Control
                type="number"
                step="any"
                name="longitud"
                value={formData.longitud}
                onChange={handleChange}
              />
            </Form.Group>
          </Row>

          {/* Botón */}
          <div className="d-grid">
            <Button
              variant="primary"
              type="submit"
              className="btn-dark"
              disabled={submitting}
            >
              {submitting ? (
                <>
                  <Spinner size="sm" className="me-2" /> Guardando…
                </>
              ) : (
                'Actualizar Monumento'
              )}
            </Button>
          </div>
        </Form>
      </Modal.Body>
    </Modal>
  );
};

export default EditMonumentForm;
