import React, { useState } from 'react';
import { Modal, Form, Button } from 'react-bootstrap';
import api from '../../../functions/api';
import '../../../styles/Forms.css';
import { useToast } from '../../ToastProvider';

const NewDomainForm = ({
  show, handleClose, onCreated,
}) => {
  const { showSuccess, showError } = useToast();

  /* ---------- state ---------- */
  const [formData, setFormData] = useState({
    absolute_url: '',
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  /* ---------- submit ---------- */
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      formData.absolute_url = formData.absolute_url.trim();
      // Obtenemos la url relativa del dominio a partir de la url absoluta
      const relativeUrl = new URL(formData.absolute_url).hostname;
      // Enviamos la solicitud de creación
      const token = localStorage.getItem('access_token');
      const response = await api.post('/web-crawling/domains/', {domain_url: relativeUrl, absolute_url: formData.absolute_url}, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      // Notificamos al padre que se creó el cliente
      onCreated(response.data);
      showSuccess("Dominio creado correctamente");
      // Cerramos el modal
      handleClose();
    } catch (err) {
      console.error(err);
      showError('No se pudo crear el dominio');
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
        <Modal.Title>Crear Nuevo Dominio</Modal.Title>
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

          {/* ---------- botón ---------- */}
          <div className="d-grid">
            <Button variant="primary" type="submit" className="btn-dark">
              Crear Dominio
            </Button>
          </div>
        </Form>
      </Modal.Body>
    </Modal>
  );
};

export default NewDomainForm;
