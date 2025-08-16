import React, { useEffect, useState, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Button, ButtonGroup } from 'react-bootstrap';
import api from '../functions/api';
import { useToast } from '../components/ToastProvider';
import SearchBar from '../components/SearchBar';
import '../styles/Common.css';

const DomainUrlsPage = () => {
  const { id_dominio } = useParams();
  const { showError } = useToast();

  const [urls, setUrls] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterClass, setFilterClass] = useState('all');
  const [filterParts, setFilterParts] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);

  const rowsPerPage = 10;

  /* ---------- Fetch ---------- */
  const fetchUrls = useCallback(async () => {
    try {
      const token = localStorage.getItem('access_token');
      const response = await api.get(`/web-crawling/domain/${id_dominio}/urls`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setUrls(response.data);
    } catch (error) {
      console.error('Error al cargar URLs del dominio:', error);
      showError('Error al cargar URLs del dominio');
    } finally {
      setLoading(false);
    }
  }, [id_dominio, showError]);

  useEffect(() => {
    fetchUrls();
  }, [fetchUrls]);

  /* ---------- Helpers ---------- */
  const getStatusText = (score) => (score > 50 ? 'Sexista' : 'No sexista');
  const getStatusStyle = (score) => {
    if (score > 65) return { color: 'red', fontWeight: 'bold' };
    if (score >= 35) return { color: '#555', fontWeight: 'bold' };
    return { color: 'green', fontWeight: 'bold' };
  };

  /* ---------- Filtros ---------- */
  const filteredUrls = urls.filter((url) => {
    const { score_sexist_global, has_sexist_parts } = url;
    const matchesQuery = url.relative_url.toLowerCase().includes(searchQuery.toLowerCase());

    // clasificación global
    const classMatch =
      filterClass === 'all' ||
      (filterClass === 'sexista' && score_sexist_global > 50) ||
      (filterClass === 'no-sexista' && score_sexist_global <= 50);

    // partes sexistas (flag backend o fallback a score)
    const hasSexist = typeof has_sexist_parts === 'boolean' ? has_sexist_parts : score_sexist_global > 0;
    const partsMatch =
      filterParts === 'all' ||
      (filterParts === 'con-partes' && hasSexist) ||
      (filterParts === 'sin-partes' && !hasSexist);

    return matchesQuery && classMatch && partsMatch;
  });

  /* ---------- Paginación ---------- */
  const indexOfLastRow = currentPage * rowsPerPage;
  const currentUrls = filteredUrls.slice(indexOfLastRow - rowsPerPage, indexOfLastRow);
  const totalPages = Math.ceil(filteredUrls.length / rowsPerPage);

  const getPagination = () => {
    const pages = [];
    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) pages.push(i);
    } else {
      pages.push(1);
      const start = Math.max(2, currentPage - 2);
      const end = Math.min(totalPages - 1, currentPage + 2);
      if (start > 2) pages.push('ellipsis-start');
      for (let i = start; i <= end; i++) pages.push(i);
      if (end < totalPages - 1) pages.push('ellipsis-end');
      pages.push(totalPages);
    }
    return [...new Set(pages)];
  };

  /* ---------- Render ---------- */
  return (
    <section className="urls-section">
      <h1 className="title">Listado de URLs del Dominio</h1>
      <div className="d-flex gap-3 align-items-center mb-3 flex-wrap">
        <SearchBar
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          setCurrentPage={setCurrentPage}
          placeholder="Buscar por URL relativa"
        />

        {/* Filtro clasificación */}
        <ButtonGroup size="sm" className="filter-group me-2">
          <Button variant={filterClass === 'all' ? 'dark' : 'outline-secondary'} onClick={() => setFilterClass('all')}>
            Todas
          </Button>
          <Button variant={filterClass === 'sexista' ? 'danger' : 'outline-danger'} onClick={() => setFilterClass('sexista')}>
            Sexistas
          </Button>
          <Button variant={filterClass === 'no-sexista' ? 'success' : 'outline-success'} onClick={() => setFilterClass('no-sexista')}>
            No Sexistas
          </Button>
        </ButtonGroup>

        {/* Filtro partes sexistas */}
        <ButtonGroup size="sm" className="filter-group">
          <Button variant={filterParts === 'all' ? 'dark' : 'outline-secondary'} onClick={() => setFilterParts('all')}>
            Todas las partes
          </Button>
          <Button variant={filterParts === 'con-partes' ? 'danger' : 'outline-danger'} onClick={() => setFilterParts('con-partes')}>
            Con partes sexistas
          </Button>
          <Button variant={filterParts === 'sin-partes' ? 'success' : 'outline-success'} onClick={() => setFilterParts('sin-partes')}>
            Sin partes sexistas
          </Button>
        </ButtonGroup>
      </div>

      {loading ? (
        <p className="status-msg">Cargando …</p>
      ) : (
        <div className="table-wrapper">
          <table className="basic-table">
            <thead>
              <tr>
                <th>URL Relativa</th>
                <th>% Sexismo</th>
                <th>Clasificación</th>
              </tr>
            </thead>
            <tbody>
              {currentUrls.map(({ id_url, relative_url, score_sexist_global }, idx) => (
                <tr key={idx}>
                  <td>
                    <Link to={`/analiticas/dominios/${id_dominio}/urls/${id_url}`} className="link-dark">
                      {relative_url}
                    </Link>
                  </td>
                  <td>{score_sexist_global.toFixed(2)}%</td>
                  <td style={getStatusStyle(score_sexist_global)}>{getStatusText(score_sexist_global)}</td>
                </tr>
              ))}
              {currentUrls.length === 0 && (
                <tr>
                  <td colSpan="3" className="text-center">No se encontraron URLs.</td>
                </tr>
              )}
            </tbody>
          </table>

          {totalPages > 1 && (
            <div className="d-flex flex-wrap justify-content-center pagination">
              {getPagination().map((page, index) =>
                page === 'ellipsis-start' || page === 'ellipsis-end' ? (
                  <span key={`ellipsis-${index}`} className="mx-1 my-1">…</span>
                ) : (
                  <Button
                    key={`page-${page}-${index}`}
                    variant={page === currentPage ? 'primary' : 'outline-primary'}
                    onClick={() => setCurrentPage(page)}
                    className="mx-1 my-1"
                  >
                    {page}
                  </Button>
                )
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
};

export default DomainUrlsPage;
