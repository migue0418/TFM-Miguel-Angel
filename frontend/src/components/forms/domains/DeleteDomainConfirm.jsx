import React, { useState } from 'react';
import { Modal, Button, Spinner } from 'react-bootstrap';
import api from '../../../functions/api';
import { useToast } from '../../ToastProvider';

const DeleteDomainConfirm = ({ data, show, handleClose, onDeleted }) => {
  const { showSuccess, showError } = useToast();
  const [loading, setLoading] = useState(false);

  if (!data) return null;               // nada que mostrar

  const handleDelete = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem('access_token');
      await api.delete(`/web-crawling/domain/${data.id_domain}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      showSuccess('Dominio eliminado');
      onDeleted?.(data.id_domain);
      handleClose();
    } catch (err) {
      console.error(err);
      showError('No se pudo eliminar el dominio');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal show={show} onHide={handleClose} centered dialogClassName="basic-modal">
      <Modal.Header closeButton>
        <Modal.Title>Eliminar Dominio</Modal.Title>
      </Modal.Header>

      <Modal.Body className="text-center">
        <p>¿Seguro que deseas eliminar a <strong>{data.domain_url}</strong>?</p>
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

export default DeleteDomainConfirm;
