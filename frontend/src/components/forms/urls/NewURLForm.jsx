import React, { useEffect, useState } from 'react';
import { Modal, Form, Button, Spinner, Row, Col } from 'react-bootstrap';
import api from '../../../functions/api';
import '../../../styles/Forms.css';
import { useToast } from '../../ToastProvider';

const NewURLForm = ({
  show, handleClose, onCreated, countriesList = [], loadingCatalog
}) => {
    const { showSuccess, showError, showInfo } = useToast();

    // Formulario inicial
    const INITIAL_FORM = { name: '', id_url: '', id_country: ''};

    const [formData, setFormData] = useState(INITIAL_FORM);
    const [loadingCountries, setLoadingTypes]   = useState(false);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    /* ---------- submit ---------- */
    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
        const token = localStorage.getItem('access_token');
        const response = await api.post('/web-crawling/urls/', {id_domain: id_domain, relative_url: relative_url, absolute_url: formData.absolute_url}, {
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
            showError('No se pudo crear la URL');
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
            <Modal.Title>Crear Nueva URL</Modal.Title>
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

            <Form.Group>
                <Form.Label>Tipo</Form.Label>
                {loadingCountries ? (
                    <div className="d-flex align-items-center gap-2">
                    <Spinner size="sm" /> Cargando…
                    </div>
                ) : (
                    <Form.Select
                    name="id_country"
                    value={formData.id_country}
                    onChange={e =>
                        setFormData(prev => ({ ...prev, id_country: e.target.value }))}
                    disabled={loadingCatalog}
                    required
                    >
                    <option value="">Selecciona…</option>
                    {countriesList.map(t => (
                        <option key={t.id_country} value={t.id_country}>
                        {t.name}
                        </option>
                    ))}
                    </Form.Select>
                )}
            </Form.Group>

            {/* ---------- botón ---------- */}
            <div className="d-grid">
                <Button variant="primary" type="submit" className="btn-dark">
                    Crear URL
                </Button>
            </div>
            </Form>
        </Modal.Body>
        </Modal>
    );
};

export default NewURLForm;
