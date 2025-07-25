import React, { useEffect, useState } from 'react';
import { Modal, Form, Button, Spinner, Row, Col } from 'react-bootstrap';
import api from '../../../functions/api';
import '../../../styles/Forms.css';
import { useToast } from '../../ToastProvider';

const NewCountryForm = ({
  show, handleClose, onCreated,
}) => {
  const { showSuccess, showError, showInfo } = useToast();

  /* ---------- state ---------- */
  const [formData, setFormData] = useState({
    name: '',
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  /* ---------- submit ---------- */
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const token = localStorage.getItem('access_token');
      const response = await api.post('/tourist-guide/management/countries/create', formData, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      // Notificamos al padre que se creó el cliente
      onCreated(response.data);
      // Cerramos el modal
      handleClose();
    } catch (err) {
      console.error(err);
      showError('No se pudo crear el pais');
    }
  };

  /* ---------- render ---------- */
  return (
    <Modal
      show={show}
      onHide={handleClose}
      backdrop="static"
      centered
      dialogClassName="basic-modal"
    >
      <Modal.Header closeButton>
        <Modal.Title>Crear Nuevo Pais</Modal.Title>
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

          {/* ---------- botón ---------- */}
          <div className="d-grid">
            <Button variant="primary" type="submit" className="btn-dark">
              Crear Pais
            </Button>
          </div>
        </Form>
      </Modal.Body>
    </Modal>
  );
};

export default NewCountryForm;
