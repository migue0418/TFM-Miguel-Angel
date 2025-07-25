import React, { useEffect, useState } from 'react';
import { Row, Spinner, Alert } from 'react-bootstrap';
import api from '../functions/api';
import '../styles/Forms.css'
import { useToast } from '../components/ToastProvider';
import ChangePasswordForm from '../components/forms/ChangePasswordForm';

const ProfilePage = () => {
  const { showError } = useToast();

  const [profile, setProfile]   = useState(null);
  const [loading, setLoading]   = useState(true);
  const [error,   setError]     = useState(false);
  const [showPwdModal, setShowPwdModal] = useState(false);

  /* ───── carga de datos ───── */
  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const token  = localStorage.getItem('access_token');
        const { data } = await api.get('/auth/me', {
          headers: { Authorization: `Bearer ${token}` },
        });
        setProfile(data);  // { username, nombre, email, created_at }
      } catch (err) {
        console.error(err);
        setError(true);
        showError('No se pudo obtener el perfil');
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, [showError]);

  /* ───── render ───── */
  if (loading) return <Spinner animation="border" className="mt-4" />;
  if (error)   return <Alert variant="danger" className="mt-4">Error al cargar perfil</Alert>;

  return (
    <>
      <h1 className="title mb-4">Perfil de Usuario</h1>

      <section className="mb-4 ms-3">
        <Row className="mb-3">
          <Row className="fw-semibold color-primary-dark fs-5">Username</Row>
          <Row>{profile.username}</Row>
        </Row>

        <Row className="mb-3">
          <Row className="fw-semibold color-primary-dark fs-5">Nombre</Row>
          <Row>{profile.nombre}</Row>
        </Row>

        <Row className="mb-3">
          <Row className="fw-semibold color-primary-dark fs-5">Email</Row>
          <Row>{profile.email}</Row>
        </Row>
      </section>

      <button
          className="btn-primary"
          aria-label="Cambiar contraseña"
          onClick={() => setShowPwdModal(true)}
        >Cambiar contraseña</button>

      {/* modal */}
      <ChangePasswordForm
        show={showPwdModal}
        handleClose={() => setShowPwdModal(false)}
        username={profile.username}
      />
    </>
  );
};

export default ProfilePage;
