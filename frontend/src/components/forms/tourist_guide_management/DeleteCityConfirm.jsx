import React, { useState } from 'react';
import { Modal, Button, Spinner } from 'react-bootstrap';
import api from '../../../functions/api';
import { useToast } from '../../ToastProvider';

const DeleteCityConfirm = ({ data, show, handleClose, onDeleted }) => {
  const { showSuccess, showError } = useToast();
  const [loading, setLoading] = useState(false);

  if (!data) return null;               // nada que mostrar

  const handleDelete = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('access_token');
      await api.delete(`/tourist-guide/management/cities/${data.id_city}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      showSuccess('Ciudad eliminada');
      onDeleted?.(data.id_city);
      handleClose();
    } catch (err) {
      console.error(err);
      showError('No se pudo eliminar la ciudad');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal show={show} onHide={handleClose} centered dialogClassName="basic-modal">
      <Modal.Header closeButton>
        <Modal.Title>Eliminar Ciudad</Modal.Title>
      </Modal.Header>

      <Modal.Body className="text-center">
        <p>¿Seguro que deseas eliminar a <strong>{data.name}</strong>?</p>
      </Modal.Body>

      <Modal.Footer>
        <Button variant="secondary" onClick={handleClose}>
          No
        </Button>

        <Button variant="primary" className='btn-dark' onClick={handleDelete} disabled={loading}>
          {loading ? <Spinner size="sm" /> : 'Sí'}
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default DeleteCityConfirm;
