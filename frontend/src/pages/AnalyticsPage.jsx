// src/pages/AnalyticsPage.jsx
import api from '../functions/api';
import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Button } from 'react-bootstrap';
import { useToast } from '../components/ToastProvider';
import '../styles/AnalyticsPage.css';
import '../styles/Common.css';
import NewClientForm from '../components/forms/NewClientForm';
import SearchBar from '../components/SearchBar';
import EditClientForm from '../components/forms/EditClientForm';

const AnalyticsPage = () => {
  const [clients, setClients] = useState([]);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState(null);
  const [openRow,   setOpenRow]   = useState(null);   // id de la fila cuyo menú está abierto
  const menuRef = useRef(null);
  const [showClientModal, setShowClientModal] = useState(false);
  // Toast para notificaciones
  const { showSuccess, showError } = useToast();
  // Editar Analítica
  const [clientToEdit, setClientToEdit] = useState(null);
  const [showEditClientModal, setShowEditClientModal] = useState(false);

  // Cargar la lista de analíticas
  const fetchClients = useCallback(async () => {
    try {
      const token = localStorage.getItem("access_token");
      const response = await api.get('/clients/', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setClients(response.data);
    } catch (error) {
      console.error('Error al cargar la lista de analíticas:', error);
      showError("Error al cargar la lista de analíticas");
      setError(error);
    } finally {
        setLoading(false);
    }
  }, [showError]);

  /* ───── Fetch inicial ───── */
  useEffect(() => {
    fetchClients();
  }, [fetchClients]);

  // Editar Licencia
  const openEditClientModal = (client) => {
    setClientToEdit(client);
    setShowEditClientModal(true);
  };
  const closeEditClientModal = () => {
    setShowEditClientModal(false);
    setClientToEdit(null);
  };

  const handleClientUpdated = (updatedClient) => {
    setClients(prev =>
      prev.map(c => (c.id === updatedClient.id ? updatedClient : c))
    );
    showSuccess('Analítica actualizado correctamente');
    closeEditClientModal();
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
  // Filtrar analíticas
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 10;
  const filteredClients = clients.filter((client) => {
    const query = searchQuery.toLowerCase();
    return (
      client.telephone.toLowerCase().includes(query) ||
      client.name.toLowerCase().includes(query) ||
      client.email.toLowerCase().includes(query)
    );
  });
  const indexOfLastRow = currentPage * rowsPerPage;
  const indexOfFirstRow = indexOfLastRow - rowsPerPage;
  const currentClients = filteredClients.slice(indexOfFirstRow, indexOfLastRow);
  const totalPages = Math.ceil(filteredClients.length / rowsPerPage);

  /* MODALES */
  // Crear cliente nuevo
  const openClientModal = () => {
    setShowClientModal(true);
  };
  const closeClientModal = () => {
    setShowClientModal(false);
  };

  const handleClientCreated = (newClientData) => {
    // Refrescar la lista de analíticas
    fetchClients();
  };

  /* ───── Render ───── */
  return (
    <section className="clients-section">
      <h1 className="title">Analíticas</h1>
      <div className='grid-header'>
        <SearchBar
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          setCurrentPage={setCurrentPage}
          placeholder="Buscar por nombre, contacto o email"
        />

        <button
          className="btn-primary"
          aria-label="Nuevo Analítica"
          onClick={openClientModal}
        >Nuevo Analítica</button>
      </div>


      {loading && <p className="status-msg">Cargando …</p>}

      {!loading && !error && (
        <div className="table-wrapper">
          <table className="clients-table">
            <thead>
              <tr>
                <th>Nombre</th>
                <th>Email</th>
                <th>Teléfono</th>
                <th aria-label="Acciones" />
              </tr>
            </thead>
            <tbody>
              {currentClients.map(({ id, name, email, telephone }) => (
                <tr key={id}>
                  <td>{name}</td>
                  <td>{email}</td>
                  <td>{telephone}</td>
                  <td className="actions-cell" ref={id === openRow ? menuRef : null}>
                    <button
                      className="icon-btn dots-btn"
                      aria-label="Acciones"
                      onClick={() => setOpenRow(openRow === id ? null : id)}
                    >
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                           strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="5"  r="1.5" />
                        <circle cx="12" cy="12" r="1.5" />
                        <circle cx="12" cy="19" r="1.5" />
                      </svg>
                    </button>

                    {openRow === id && (
                      <ul className="row-menu">
                        <li><button onClick={() => openEditClientModal({ id, name, email, telephone })}>Editar</button></li>
                        <li><button onClick={() => alert(`Borrar ${name}`)}>Borrar</button></li>
                      </ul>
                    )}
                  </td>
                </tr>
              ))}
              {currentClients.length === 0 && (
                <tr>
                  <td colSpan="6" className="text-center">
                    No se encontraron analíticas.
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

      <NewClientForm
        show={showClientModal}
        handleClose={closeClientModal}
        onCreated={handleClientCreated}
      />

      <EditClientForm
        show={showEditClientModal}
        clientData={clientToEdit}
        handleClose={closeEditClientModal}
        onCreated={handleClientUpdated}
      />
    </section>
  );
};

export default AnalyticsPage;
