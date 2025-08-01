import React, { useState, useEffect } from 'react';
import { Modal, Form, Button } from 'react-bootstrap';
import api from '../../../functions/api';
import '../../../styles/Forms.css'
import { useToast } from '../../ToastProvider';


const EditDomainForm = ({ data, show, handleClose, onUpdated }) => {
  const [formData, setFormData] = useState({
    domain_url: '',
    absolute_url: '',
  });

  const { showSuccess, showError } = useToast();
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
      formData.absolute_url = formData.absolute_url.trim();
      // Obtenemos la url relativa del dominio a partir de la url absoluta
      const relativeUrl = new URL(formData.absolute_url).hostname;
      setFormData(prev => ({ ...prev, domain_url: relativeUrl }));
      // Enviamos la solicitud de actualización
      const token = localStorage.getItem('access_token');
      const response = await api.put(`/web-crawling/domain/${data.id_domain}`, formData, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      // Notificamos al padre que se creó el cliente
      onUpdated(response.data);
      showSuccess("Dominio actualizado correctamente");
      // Cerramos el modal
      handleClose();
    } catch (error) {
      showError('Error al actualizar el dominio:', error);
    }
  };

  return (
    <Modal show={show} onHide={handleClose} backdrop="static" keyboard={false} centered dialogClassName="basic-modal">
      <Modal.Header closeButton>
        <Modal.Title>Actualizar Información de Dominio</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Form onSubmit={handleSubmit}>
          <Form.Group className="mb-3">
            <Form.Label>URL absoluta</Form.Label>
            <Form.Control
              type="text"
              name="absolute_url"
              value={formData.absolute_url}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <div className="d-grid">
            <Button variant="primary" type="submit" className='btn-dark'>
              Actualizar Dominio
            </Button>
          </div>
        </Form>
      </Modal.Body>
    </Modal>
  );
};

export default EditDomainForm;
