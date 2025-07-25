import React, { useState, useEffect } from 'react';
import { Modal, Form, Button, Spinner, Row, Col } from 'react-bootstrap';
import api from '../../functions/api';
import '../../styles/Forms.css'
import { useToast } from '../ToastProvider';


const EditUserForm = ({ data, show, handleClose, onUpdated }) => {
  const [formData, setFormData] = useState({
      id_user: null,
      nombre: '',
      username: '',
      email: '',
      password: '',
      roles: []
  });

  const [rolesCatalog, setRolesCatalog] = useState([]);
  const [loadingRoles, setLoadingRoles] = useState(false);

  const { showError } = useToast();
  useEffect(() => {
    if (data) {
      // Copiamos los campos de la licencia en el estado
      setFormData({ ...data });

      // ⇢ extrae solo los id_rol
      const rolesIds = (data.roles || []).map(r => r.id_rol);
      setFormData({ ...data, roles: rolesIds });
    }
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
  }, [show, showError, data]);

  const toggleRole = (idRol) => {
    setFormData(prev => {
      const selected = prev.roles.includes(idRol)
        ? prev.roles.filter(r => r !== idRol)   // quita
        : [...prev.roles, idRol];               // añade
      return { ...prev, roles: selected };
    });
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const payload = {
        id_user: formData.id_user,
        nombre: formData.nombre,
        username: formData.username,
        password: formData.password ?? "",
        email: formData.email,
        roles: formData.roles,
      };

      const token = localStorage.getItem('access_token');
      const response = await api.put('/auth/users/edit', payload, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      // Notificamos al padre que se creó el cliente
      onUpdated(response.data);
      // Cerramos el modal
      handleClose();
    } catch (error) {
      showError('Error al actualizar:', error);
    }
  };

  return (
    <Modal show={show} onHide={handleClose} backdrop="static" keyboard={false} centered dialogClassName="basic-modal">
      <Modal.Header closeButton>
        <Modal.Title>Actualizar Usuario</Modal.Title>
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
              placeholder='Si se rellena sobreescribe la contraseña'
              value={formData.password}
              onChange={handleChange}
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

          <div className="d-grid">
            <Button variant="primary" type="submit" className='btn-dark'>
              Actualizar Usuario
            </Button>
          </div>
        </Form>
      </Modal.Body>
    </Modal>
  );
};

export default EditUserForm;
