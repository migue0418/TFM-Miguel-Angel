import api from '../functions/api';
import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Button } from 'react-bootstrap';
import { useToast } from '../components/ToastProvider';
import '../styles/Common.css';
import NewUserForm from '../components/forms/NewUserForm';
import SearchBar from '../components/SearchBar';
import EditUserForm from '../components/forms/EditUserForm';
import DeleteUserConfirm from '../components/forms/DeleteUserConfirm';
import RowActionsMenu from '../components/RowActionsMenu';

const UsersManagement = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [openRow, setOpenRow] = useState(null);
  const btnRefs = useRef({});
  const menuRef = useRef(null);
  const [showCreateModal, setShowCreateModal] = useState(false);

  // Toast para notificaciones
  const { showSuccess, showError } = useToast();

  // Editar Usuario
  const [dataToEdit, setdataToEdit] = useState(null);
  const [showEditModal, setShowEditModal] = useState(false);

  // Eliminar Usuario
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [dataToDelete, setdataToDelete] = useState(null);

  // Cargar la lista de users
  const fetchUsers = useCallback(async () => {
    try {
      const token = localStorage.getItem("access_token");
      const response = await api.get('/auth/users', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setUsers(response.data);
    } catch (error) {
      console.error('Error al cargar la lista de usuarios:', error);
      showError("Error al cargar la lista de usuarios");
      setError(error);
    } finally {
        setLoading(false);
    }
  }, [showError]);

  /* ───── Fetch inicial ───── */
  useEffect(() => {
    fetchUsers();
  }, [fetchUsers]);

  // Editar Licencia
  const openEditModal = (users) => {
    setdataToEdit(users);
    setShowEditModal(true);
  };
  const closeEditModal = () => {
    setShowEditModal(false);
    setdataToEdit(null);
  };

  const handleUpdated = (updatedData) => {
    setUsers(prev =>
      prev.map(c => (c.id_user === updatedData.id_user ? updatedData : c))
    );
    // Refrescar la lista de users
    fetchUsers();
    showSuccess('Usuario actualizado correctamente');
    closeEditModal();
  };

  // Eliminar usuario
  const openDeleteModal = (users) => {
    setdataToDelete(users);
    setShowDeleteModal(true);
  };
  const closeDeleteModal = () => {
    setShowDeleteModal(false);
    setdataToDelete(null);
  };

  const handleDeleted = (deletedData) => {
    // Refrescar la lista de users
    fetchUsers();
    showSuccess('Usuario eliminado correctamente');
    closeDeleteModal();
  };

  /* ───── Cerrar menú si se hace clic fuera ───── */
  useEffect(() => {
    const handleClickOutside = e => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setOpenRow(null);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);


  /*  BÚSQUEDA Y PAGINACIÓN  */
  // Filtrar users
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 10;
  const filteredData = users.filter((user) => {
    const query = searchQuery.toLowerCase();
    return (
      user.email?.toLowerCase().includes(query) ||
      user.nombre?.toLowerCase().includes(query) ||
      user.username?.toLowerCase().includes(query)
    );
  });
  const indexOfLastRow = currentPage * rowsPerPage;
  const indexOfFirstRow = indexOfLastRow - rowsPerPage;
  const currentUsers = filteredData.slice(indexOfFirstRow, indexOfLastRow);
  const totalPages = Math.ceil(filteredData.length / rowsPerPage);

  /* MODALES */
  const openCreateModal = () => {
    setShowCreateModal(true);
  };
  const closeCreateModal = () => {
    setShowCreateModal(false);
  };

  const handleCreated = (newData) => {
    showSuccess('Nuevo usuario creado con éxito');
    // Refrescar la lista de users
    fetchUsers();
  };

  /* ───── Render ───── */
  return (
    <section className="users-section">
      <h1 className="title">Panel de Administración</h1>
      <div className='grid-header'>
        <SearchBar
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          setCurrentPage={setCurrentPage}
          placeholder="Buscar por nombre o email"
        />

        <button
          className="btn-primary"
          aria-label="Nuevo Usuario"
          onClick={openCreateModal}
        >Nuevo Usuario</button>
      </div>


      {loading && <p className="status-msg">Cargando …</p>}

      {!loading && !error && (
        <div className="table-wrapper">
          <table className="basic-table">
            <thead>
              <tr>
                <th>Nombre</th>
                <th>Username</th>
                <th>Email</th>
                <th>Roles</th>
                <th aria-label="Acciones" />
              </tr>
            </thead>
            <tbody>
              {currentUsers.map(({ id_user, username, nombre, email, roles }) => (
                <React.Fragment key={id_user}>
                <tr>
                  <td>{nombre}</td>
                  <td>{username}</td>
                  <td>{email}</td>
                  {/* ► ROLES */}
                  <td className="pills-cell">
                    {roles.map(({ id_rol, nombre: rolNombre }) => (
                      <span key={id_rol} className="format-pill">
                        {rolNombre}
                      </span>
                    ))}
                  </td>
                  <td className="actions-cell">
                    <button
                      ref={(el) => (btnRefs.current[id_user] = el)}
                      className="icon-btn dots-btn"
                      aria-label="Acciones"
                      onClick={() => setOpenRow(openRow === id_user ? null : id_user)}
                    >
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                           strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="5"  r="1.5" />
                        <circle cx="12" cy="12" r="1.5" />
                        <circle cx="12" cy="19" r="1.5" />
                      </svg>
                    </button>
                  </td>
                </tr>
                {/* menú — solo para la fila activa */}
                {openRow === id_user && (
                  <RowActionsMenu
                    anchorEl={btnRefs.current[id_user]}
                    onEdit={() => openEditModal({ id_user, nombre, username, email, roles })}
                    onDelete={() => openDeleteModal({ id_user, nombre, username, email, roles })}
                    onClose={() => setOpenRow(null)}
                  />
                )}
              </React.Fragment>
              ))}
              {currentUsers.length === 0 && (
                <tr>
                  <td colSpan="6" className="text-center">
                    No se encontraron usuarios.
                  </td>
                </tr>
              )}
            </tbody>
          </table>

          {/* Paginación */}
          {totalPages > 1 && (
            <div className="d-flex justify-content-center pagination">
              {[...Array(totalPages)].map((_, index) => {
                const page = index + 1;
                return (
                  <Button
                    key={page}
                    variant={page === currentPage ? 'primary' : 'outline-primary'}
                    onClick={() => setCurrentPage(page)}
                    className="mx-1"
                  >
                    {page}
                  </Button>
                );
              })}
            </div>
          )}
        </div>
      )}

      <NewUserForm
        show={showCreateModal}
        handleClose={closeCreateModal}
        onCreated={handleCreated}
      />

      <EditUserForm
        show={showEditModal}
        data={dataToEdit}
        handleClose={closeEditModal}
        onUpdated={handleUpdated}
      />

      <DeleteUserConfirm
        show={showDeleteModal}
        data={dataToDelete}
        handleClose={closeDeleteModal}
        onDeleted={handleDeleted}
      />
    </section>
  );
};

export default UsersManagement;
