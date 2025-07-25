import api from '../functions/api';
import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Button, Alert } from 'react-bootstrap';
import { useToast } from '../components/ToastProvider';
import '../styles/Common.css';
import NewRolForm from '../components/forms/NewRolForm';
import SearchBar from '../components/SearchBar';
import EditRolForm from '../components/forms/EditRolForm';
import RowActionsMenu from '../components/RowActionsMenu';

const RolesManagement = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [openRow, setOpenRow] = useState(null);   // id_rol de la fila cuyo menú está abierto
  const btnRefs = useRef({});
  const menuRef = useRef(null);
  const [showCreateModal, setshowCreateModal] = useState(false);
  // Toast para notificaciones
  const { showSuccess, showError } = useToast();
  // Editar Rol
  const [dataToEdit, setdataToEdit] = useState(null);
  const [showEditModal, setshowEditModal] = useState(false);

  // Cargar la lista de roles
  const fetchRoles = useCallback(async () => {
    try {
      const token = localStorage.getItem("access_token");
      const response = await api.get('/auth/roles', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setData(response.data);
    } catch (error) {
      console.error('Error al cargar la lista de roles:', error);
      showError("Error al cargar la lista de roles");
      setError(error);
    } finally {
        setLoading(false);
    }
  }, [showError]);

  /* ───── Fetch inicial ───── */
  useEffect(() => {
    fetchRoles();
  }, [fetchRoles]);

  // Editar Licencia
  const openEditModal = (data) => {
    setdataToEdit(data);
    setshowEditModal(true);
  };
  const closeEditModal = () => {
    setshowEditModal(false);
    setdataToEdit(null);
  };

  const handleUpdated = (updatedData) => {
    setData(prev =>
      prev.map(c => (c.id_rol === updatedData.id_rol ? updatedData : c))
    );
    // Refrescar la lista de roles
    fetchRoles();
    showSuccess('Rol actualizado correctamente');
    closeEditModal();
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
  // Filtrar roles types
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 10;
  const filteredData = data.filter((item) => {
    const query = searchQuery.toLowerCase();
    return (
      item.nombre.toLowerCase().includes(query)
    );
  });
  const indexOfLastRow = currentPage * rowsPerPage;
  const indexOfFirstRow = indexOfLastRow - rowsPerPage;
  const currentData = filteredData.slice(indexOfFirstRow, indexOfLastRow);
  const totalPages = Math.ceil(filteredData.length / rowsPerPage);

  /* MODALES */
  // Crear nuevo
  const openCreateModal = () => {
    setshowCreateModal(true);
  };
  const closeCreateModal = () => {
    setshowCreateModal(false);
  };

  const handleCreated = (newData) => {
    showSuccess('Nuevo rol creado');
    // Refrescar la lista de roles
    fetchRoles();
  };

  const protectedRoles = [1, 2, 3];   // IDs que NO se pueden borrar

  /* ───── Render ───── */
  return (
    <section className="roles-section">
      <h1 className="title">Roles</h1>

      <Alert
        variant="warning"
        className="mb-3"
      >
        La creación de roles distintos a los existentes
        <strong> no tendrá ningún efecto</strong> salvo que se adapte la parte
        programática en el código del servidor.
      </Alert>

      <div className='grid-header'>
        <SearchBar
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          setCurrentPage={setCurrentPage}
          placeholder="Buscar por nombre del rol"
        />

        <button
          className="btn-primary"
          aria-label="Nuevo Rol"
          onClick={openCreateModal}
        >Nuevo Rol</button>
      </div>


      {loading && <p className="status-msg">Cargando …</p>}

      {!loading && !error && (
        <div className="table-wrapper">
          <table className="basic-table">
            <thead>
              <tr>
                <th>Nombre</th>
                <th>Descripción</th>
                <th aria-label="Acciones" />
              </tr>
            </thead>
            <tbody>
              {currentData.map(({ id_rol, nombre, descripcion }) => (
                <React.Fragment key={id_rol}>
                <tr>
                  <td>{nombre}</td>
                  <td>{descripcion}</td>
                  <td className="actions-cell">
                    <button
                      ref={(el) => (btnRefs.current[id_rol] = el)}
                      className="icon-btn dots-btn"
                      aria-label="Acciones"
                      onClick={() => setOpenRow(openRow === id_rol ? null : id_rol)}
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
                {openRow === id_rol && (
                  <RowActionsMenu
                    anchorEl={btnRefs.current[id_rol]}
                    onEdit={() => openEditModal({ id_rol, nombre, descripcion })}
                    {...(!protectedRoles.includes(id_rol) && { onDelete: () => alert(`Borrar ${nombre}`) })}
                    onClose={() => setOpenRow(null)}
                  />
                )}
              </React.Fragment>
              ))}
              {currentData.length === 0 && (
                <tr>
                  <td colSpan="6" className="text-center">
                    No se encontraron roles.
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

      <NewRolForm
        show={showCreateModal}
        handleClose={closeCreateModal}
        onCreated={handleCreated}
      />

      <EditRolForm
        show={showEditModal}
        data={dataToEdit}
        handleClose={closeEditModal}
        onUpdated={handleUpdated}
      />
    </section>
  );
};

export default RolesManagement;
