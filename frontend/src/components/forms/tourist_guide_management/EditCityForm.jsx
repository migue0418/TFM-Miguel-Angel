import React, { useState, useEffect } from 'react';
import { Modal, Form, Button, Spinner } from 'react-bootstrap';
import api from '../../../functions/api';
import '../../../styles/Forms.css'
import { useToast } from '../../ToastProvider';


const EditCityForm = ({ data, show, handleClose, onUpdated, countriesList = [], loadingCatalog }) => {
  // Formulario inicial
  const INITIAL_FORM = { name: '', id_city: '', id_country: ''};

  const [formData, setFormData] = useState(INITIAL_FORM);
  const [loadingCountries, setLoadingTypes]   = useState(false);

  const { showSuccess, showError, showInfo, showWarning } = useToast();
  useEffect(() => {
    if (data) {
      // Copiamos los campos de la licencia en el estado
      setFormData({ ...data });
    }
  }, [data]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const token = localStorage.getItem('access_token');
      const response = await api.post('/tourist-guide/management/cities/edit', formData, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      // Notificamos al padre que se creó el cliente
      onUpdated(response.data);
      // Cerramos el modal
      handleClose();
    } catch (error) {
      showError('Error al actualizar la ciudad:', error);
    }
  };

  return (
    <Modal show={show} onHide={handleClose} backdrop="static" keyboard={false} centered dialogClassName="basic-modal">
      <Modal.Header closeButton>
        <Modal.Title>Actualizar Información de Ciudad</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Form onSubmit={handleSubmit}>
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

          <Form.Group>
            <Form.Label>Tipo</Form.Label>
            {loadingCountries ? (
                <div className="d-flex align-items-center gap-2 mb-3">
                <Spinner size="sm" /> Cargando…
                </div>
            ) : (
                <Form.Select
                name="id_country"
                value={formData.id_country}
                onChange={e =>
                    setFormData(prev => ({ ...prev, id_country: e.target.value }))}
                disabled={loadingCatalog}
                required
                >
                <option value="">Selecciona…</option>
                {countriesList.map(t => (
                    <option key={t.id_country} value={t.id_country}>
                    {t.name}
                    </option>
                ))}
                </Form.Select>
            )}
            </Form.Group>

          <div className="d-grid">
            <Button variant="primary" type="submit" className='btn-dark'>
              Actualizar Ciudad
            </Button>
          </div>
        </Form>
      </Modal.Body>
    </Modal>
  );
};

export default EditCityForm;
