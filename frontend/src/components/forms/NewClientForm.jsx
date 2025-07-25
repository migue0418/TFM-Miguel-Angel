import React, { useState } from 'react';
import { Modal, Form, Button } from 'react-bootstrap';
import api from '../../functions/api';
import '../../styles/Forms.css'


const NewClientForm = ({ show, handleClose, onCreated }) => {
  const [formData, setFormData] = useState({
    nombre: '',
    telefono: '',
    email: '',
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const token = localStorage.getItem('access_token');
      const response = await api.post('/clients-management/sign-up', formData, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      // Notificamos al padre que se creó el cliente
      onCreated(response.data);
      // Cerramos el modal
      handleClose();
    } catch (error) {
      console.error('Error al crear el cliente:', error);
      // Puedes mostrar un toast o algún mensaje de error
    }
  };

  return (
    <Modal show={show} onHide={handleClose} backdrop="static" keyboard={false} centered dialogClassName="basic-modal">
      <Modal.Header closeButton>
        <Modal.Title>Crear Nuevo Analítica</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Form onSubmit={handleSubmit}>
          <Form.Group className="mb-3">
            <Form.Label>Email</Form.Label>
            <Form.Control
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
            />
          </Form.Group>

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

          <Form.Group className="mb-4">
            <Form.Label>Teléfono de Contacto (opcional)</Form.Label>
            <Form.Control
              type="text"
              name="telefono"
              value={formData.telefono}
              onChange={handleChange}
            />
          </Form.Group>

          <div className="d-grid">
            <Button variant="primary" type="submit" className='btn-dark'>
              Crear Analítica
            </Button>
          </div>
        </Form>
      </Modal.Body>
    </Modal>
  );
};

export default NewClientForm;
