// src/components/forms/NewRolForm.jsx
import React, { useState } from 'react';
import { Modal, Form, Button } from 'react-bootstrap';
import api from '../../functions/api';
import '../../styles/Forms.css';
import { useToast } from '../ToastProvider';

const NewRolForm = ({
  show, handleClose, onCreated,
}) => {
  const { showError } = useToast();

  /* ---------- state ---------- */
  const [formData, setFormData] = useState({
    nombre: '',
    descripcion: '',
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
      const response = await api.post('/auth/roles/create', formData, {
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
      showError('No se pudo crear el rol');
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
        <Modal.Title>Subir nuevo rol</Modal.Title>
      </Modal.Header>

      <Modal.Body>
        <Form onSubmit={handleSubmit}>
          <Form.Group className="mb-3">
            <Form.Label>Nombre</Form.Label>
            <Form.Control
              type="text"
              name="nombre"
              value={formData.nombre}
              onChange={handleChange}
              required
            />
          </Form.Group>
          <Form.Group className="mb-3">
            <Form.Label>Descripción</Form.Label>
            <Form.Control
              type="text"
              name="descripcion"
              value={formData.descripcion}
              onChange={handleChange}
              required
            />
          </Form.Group>

          {/* ---------- botón ---------- */}
          <div className="d-grid">
            <Button variant="primary" type="submit" className="btn-dark">
              Crear Rol
            </Button>
          </div>
        </Form>
      </Modal.Body>
    </Modal>
  );
};

export default NewRolForm;
