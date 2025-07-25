// src/pages/AnalyticsPage.jsx
import api from '../functions/api';
import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Button } from 'react-bootstrap';
import { useToast } from '../components/ToastProvider';
import SearchBar from '../components/SearchBar';
import RowActionsMenu from '../components/RowActionsMenu';
import NewDomainForm from '../components/forms/domains/NewDomainForm';
import EditDomainForm from '../components/forms/domains/EditDomainForm';
import DeleteDomainConfirm from '../components/forms/domains/DeleteDomainConfirm';

const DomainsPage = () => {
  const [domains, setDomains] = useState([]);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState(null);
  const [openRow,   setOpenRow]   = useState(null);   // id_domain de la fila cuyo menú está abierto
  const btnRefs = useRef({});
  const menuRef = useRef(null);
  const [showCreateModal, setShowCreateModal] = useState(false);

  // Toast para notificaciones
  const { showSuccess, showError } = useToast();

  // Editar Dominio
  const [dataToEdit, setdataToEdit] = useState(null);
  const [showEditModal, setShowEditModal] = useState(false);

  // Eliminar Dominio
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [dataToDelete, setdataToDelete] = useState(null);

  // Cargar la lista de dominios
  const fetchDomains = useCallback(async () => {
    try {
      const token = localStorage.getItem("access_token");
      const response = await api.get('/web-crawling/domains/', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setDomains(response.data);
    } catch (error) {
      console.error('Error al cargar la lista de dominios:', error);
      showError("Error al cargar la lista de dominios");
      setError(error);
    } finally {
        setLoading(false);
    }
  }, [showError]);

  /* ───── Fetch inicial ───── */
  useEffect(() => {
    fetchDomains();
  }, [fetchDomains]);

  /* MODALES */
  const openCreateModal = () => {
    setShowCreateModal(true);
  };
  const closeCreateModal = () => {
    setShowCreateModal(false);
  };

  const handleCreated = (newData) => {
    showSuccess('Nuevo usuario creado con éxito');
    // Refrescar la lista de domains
    fetchDomains();
  };

  // Editar Licencia
  const openEditModal = (domains) => {
    setdataToEdit(domains);
    setShowEditModal(true);
  };
  const closeEditModal = () => {
    setShowEditModal(false);
    setdataToEdit(null);
  };

  const handleUpdated = (updatedData) => {
    setDomains(prev =>
      prev.map(c => (c.id_domain === updatedData.id_domain ? updatedData : c))
    );
    // Refrescar la lista de domains
    fetchDomains();
    showSuccess('Dominio actualizado correctamente');
    closeEditModal();
  };

  // Eliminar usuario
  const openDeleteModal = (domains) => {
    setdataToDelete(domains);
    setShowDeleteModal(true);
  };
  const closeDeleteModal = () => {
    setShowDeleteModal(false);
    setdataToDelete(null);
  };

  const handleDeleted = (deletedData) => {
    // Refrescar la lista de domains
    fetchDomains();
    showSuccess('Dominio eliminado correctamente');
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
  // Filtrar dominios
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 10;
  const filteredDomains = domains.filter((domain) => {
    const query = searchQuery.toLowerCase();
    return (
      domain.domain_url.toLowerCase().includes(query) ||
      domain.absolute_url.toLowerCase().includes(query)
    );
  });
  const indexOfLastRow = currentPage * rowsPerPage;
  const indexOfFirstRow = indexOfLastRow - rowsPerPage;
  const currentDomains = filteredDomains.slice(indexOfFirstRow, indexOfLastRow);
  const totalPages = Math.ceil(filteredDomains.length / rowsPerPage);


  /* ───── Render ───── */
  return (
    <section className="domains-section">
      <h1 className="title">Dominios Web</h1>
      <div className='grid-header'>
        <SearchBar
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          setCurrentPage={setCurrentPage}
          placeholder="Buscar por URL del dominio"
        />

        <button
          className="btn-primary"
          aria-label="Nuevo Dominio"
          onClick={openCreateModal}
        >Nuevo Dominio</button>
      </div>


      {loading && <p className="status-msg">Cargando …</p>}

      {!loading && !error && (
        <div className="table-wrapper">
          <table className="basic-table">
            <thead>
              <tr>
                <th>Dominio</th>
                <th>URL Absoluta</th>
                <th aria-label="Acciones" />
              </tr>
            </thead>
            <tbody>
              {currentDomains.map(({ id_domain, domain_url, absolute_url }) => (
                <tr key={id_domain}>
                  <td>{domain_url}</td>
                  <td>{absolute_url}</td>
                  <td className="actions-cell">
                    <button
                      ref={(el) => (btnRefs.current[id_domain] = el)}
                      className="icon-btn dots-btn"
                      aria-label="Acciones"
                      onClick={() => setOpenRow(openRow === id_domain ? null : id_domain)}
                    >
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                           strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="5"  r="1.5" />
                        <circle cx="12" cy="12" r="1.5" />
                        <circle cx="12" cy="19" r="1.5" />
                      </svg>
                    </button>

                    {openRow === id_domain && (
                      <RowActionsMenu
                        anchorEl={btnRefs.current[id_domain]}
                        onEdit={() => openEditModal({ id_domain, domain_url, absolute_url })}
                        onDelete={() => openDeleteModal({ id_domain, domain_url, absolute_url })}
                        onClose={() => setOpenRow(null)}
                      />
                    )}
                  </td>
                </tr>
              ))}
              {currentDomains.length === 0 && (
                <tr>
                  <td colSpan="6" className="text-center">
                    No se encontraron dominios.
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

      <NewDomainForm
        show={showCreateModal}
        handleClose={closeCreateModal}
        onCreated={handleCreated}
      />

      <EditDomainForm
        show={showEditModal}
        data={dataToEdit}
        handleClose={closeEditModal}
        onUpdated={handleUpdated}
      />

      <DeleteDomainConfirm
        show={showDeleteModal}
        data={dataToDelete}
        handleClose={closeDeleteModal}
        onDeleted={handleDeleted}
      />
    </section>
  );
};

export default DomainsPage;
