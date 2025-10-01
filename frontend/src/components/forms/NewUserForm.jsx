import React, { useEffect, useState } from 'react';
import { Modal, Form, Button, Spinner, Row, Col } from 'react-bootstrap';
import api from '../../functions/api';
import '../../styles/Forms.css';
import { useToast } from '../ToastProvider';

const NewUserForm = ({
  show, handleClose, onCreated,
}) => {
  const { showSuccess, showError } = useToast();

  const [rolesCatalog, setRolesCatalog] = useState([]);
  const [loadingRoles, setLoadingRoles] = useState(false);
  const [formData, setFormData] = useState({
    nombre: '',
    username: '',
    email: '',
    password: '',
    roles: []
  });

  const toggleRole = (idRol) => {
    setFormData(prev => {
      const selected = prev.roles.includes(idRol)
        ? prev.roles.filter(r => r !== idRol)   // quita
        : [...prev.roles, idRol];               // añade
      return { ...prev, roles: selected };
    });
  };

  const handleChange = ({ target: { name, value } }) =>
    setFormData(prev => ({ ...prev, [name]: value }));

  useEffect(() => {
    if (!show) return;                   // evita peticiones si el modal está cerrado
    const fetchRoles = async () => {
      try {
        setLoadingRoles(true);
        const token = localStorage.getItem('access_token');
        const { data } = await api.get('/auth/roles', {
          headers: { Authorization: `Bearer ${token}` }
        });
        setRolesCatalog(data);           // [{ id_rol, nombre }, …]
      } catch (err) {
        console.error(err);
        showError('No se pudieron cargar los roles');
      } finally {
        setLoadingRoles(false);
      }
    };
    fetchRoles();
  }, [show, showError]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    /* Construimos el JSON final */
    const payload = {
      nombre: formData.nombre,
      username: formData.username,
      password: formData.password,
      email: formData.email,
      roles: formData.roles,
    };

    try {
      const token = localStorage.getItem('access_token');
      await api.post('/auth/users/create', payload, {
        headers: { Authorization: `Bearer ${token}` },
      });
      showSuccess('Encuesta creada con éxito');
      onCreated?.();
      handleClose();
    } catch (err) {
      showError('No se pudo crear la encuesta');
    }
  };

  return (
    <Modal
      show={show}
      onHide={handleClose}
      backdrop="static"
      centered
      dialogClassName="basic-modal"
    >
      <Modal.Header closeButton>
        <Modal.Title>Nuevo Usuario</Modal.Title>
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
            <Form.Label>Username</Form.Label>
            <Form.Control
              type="text"
              name="username"
              value={formData.username}
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

          <Form.Group className="mb-3">
            <Form.Label>Contraseña</Form.Label>
            <Form.Control
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
            />
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label>Roles</Form.Label>
            {loadingRoles ? (
              <div className="d-flex align-items-center gap-2">
                <Spinner size="sm" /> Cargando…
              </div>
            ) : (
              <Row xs={1} md={2} className="gy-1">
                {rolesCatalog.map(({ id_rol, nombre }) => (
                  <Col key={id_rol}>
                    <Form.Check
                      type="checkbox"
                      id={`rol-${id_rol}`}
                      label={nombre}
                      checked={formData.roles.includes(id_rol)}
                      onChange={() => toggleRole(id_rol)}
                    />
                  </Col>
                ))}
              </Row>
            )}
          </Form.Group>

          <div className="d-grid mt-4">
            <Button variant="primary" type="submit" className="btn-dark">
              Crear Usuario
            </Button>
          </div>
        </Form>
      </Modal.Body>
    </Modal>
  );
};

export default NewUserForm;
