import React, { useState, useEffect } from 'react';
import { Modal, Form, Button } from 'react-bootstrap';
import api from '../../functions/api';
import '../../styles/Forms.css'
import { useToast } from '../../components/ToastProvider';


const EditClientForm = ({ clientData, show, handleClose, onUpdated }) => {
  const [formData, setFormData] = useState({
    id_cliente: '',
    nombre: '',
    telefono: '',
    email: '',
    usuario_modificacion: localStorage.getItem("username")
  });

  const { showError } = useToast();
  useEffect(() => {
    if (clientData) {
      // Copiamos los campos de la licencia en el estado
      setFormData({ ...clientData });
    }
  }, [clientData]);

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
      onUpdated(response.data);
      // Cerramos el modal
      handleClose();
    } catch (error) {
      showError('Error al actualizar el cliente:', error);
    }
  };

  return (
    <Modal show={show} onHide={handleClose} backdrop="static" keyboard={false} centered dialogClassName="basic-modal">
      <Modal.Header closeButton>
        <Modal.Title>Actualizar Analítica</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Form onSubmit={handleSubmit}>
          <Form.Group className="mb-3">
            <Form.Label>Nombre</Form.Label>
            <Form.Control
              type="text"
              name="nombre"
              value={formData.name}
              onChange={handleChange}
              required
            />
          </Form.Group>

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

          <Form.Group className="mb-4">
            <Form.Label>Teléfono de Contacto (opcional)</Form.Label>
            <Form.Control
              type="text"
              name="telefono"
              value={formData.telephone}
              onChange={handleChange}
            />
          </Form.Group>

          <div className="d-grid">
            <Button variant="primary" type="submit" className='btn-dark'>
              Actualizar Analítica
            </Button>
          </div>
        </Form>
      </Modal.Body>
    </Modal>
  );
};

export default EditClientForm;
