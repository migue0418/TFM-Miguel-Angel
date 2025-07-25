import React, { useState, useEffect } from 'react';
import { Modal, Form, Button } from 'react-bootstrap';
import api from '../../../functions/api';
import '../../../styles/Forms.css'
import { useToast } from '../../ToastProvider';


const EditCountryForm = ({ data, show, handleClose, onUpdated }) => {
  const [formData, setFormData] = useState({
    id_country: '',
    name: '',
  });

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
      const response = await api.post('/tourist-guide/management/countries/edit', formData, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      // Notificamos al padre que se creó el cliente
      onUpdated(response.data);
      // Cerramos el modal
      handleClose();
    } catch (error) {
      showError('Error al actualizar el pais:', error);
    }
  };

  return (
    <Modal show={show} onHide={handleClose} backdrop="static" keyboard={false} centered dialogClassName="basic-modal">
      <Modal.Header closeButton>
        <Modal.Title>Actualizar Información de Pais</Modal.Title>
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

          <div className="d-grid">
            <Button variant="primary" type="submit" className='btn-dark'>
              Actualizar Pais
            </Button>
          </div>
        </Form>
      </Modal.Body>
    </Modal>
  );
};

export default EditCountryForm;
