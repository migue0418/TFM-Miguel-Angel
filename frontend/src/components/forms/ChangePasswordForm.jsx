import React, { useState } from 'react';
import { Modal, Form, Button, Spinner } from 'react-bootstrap';
import api from '../../functions/api';
import '../../styles/Forms.css'
import { useToast } from '../ToastProvider';

const ChangePasswordForm = ({ username, show, handleClose }) => {
  const { showSuccess, showError } = useToast();

  const [form, setForm]   = useState({ current: '', new1: '', new2: '' });
  const [loading, setLoading] = useState(false);

  const handleChange = (e) =>
    setForm(prev => ({ ...prev, [e.target.name]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (form.new1 !== form.new2) {
      showError('Las nuevas contraseñas no coinciden');
      return;
    }
    try {
      setLoading(true);
      const token = localStorage.getItem('access_token');
      await api.post(
        '/auth/users/change-password',
        { username, password: form.current, new_password: form.new1 },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      showSuccess('Contraseña actualizada');
      handleClose();
      setForm({ current: '', new1: '', new2: '' });
    } catch (err) {
      console.error(err);
      showError('No se pudo cambiar la contraseña');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal show={show} onHide={handleClose} centered backdrop="static" dialogClassName="basic-modal">
      <Modal.Header closeButton>
        <Modal.Title>Cambiar contraseña</Modal.Title>
      </Modal.Header>

      <Modal.Body>
        <Form onSubmit={handleSubmit}>
          <Form.Group className="mb-3">
            <Form.Label>Contraseña actual</Form.Label>
            <Form.Control
              type="password"
              name="current"
              value={form.current}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label>Nueva contraseña</Form.Label>
            <Form.Control
              type="password"
              name="new1"
              value={form.new1}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label>Repetir nueva contraseña</Form.Label>
            <Form.Control
              type="password"
              name="new2"
              value={form.new2}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <div className="d-grid">
            <Button variant="primary" type="submit" disabled={loading} className='btn-dark'>
              {loading ? <Spinner size="sm" /> : 'Guardar'}
            </Button>
          </div>
        </Form>
      </Modal.Body>
    </Modal>
  );
};

export default ChangePasswordForm;
