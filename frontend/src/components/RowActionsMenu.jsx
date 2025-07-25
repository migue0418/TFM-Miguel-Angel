// src/components/RowActionsMenu.jsx
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

const MENU_W = 140;              // ancho aprox. del menú – ajústalo si cambias CSS

export default function RowActionsMenu({
  anchorEl,          // ⬇️  botón ⋯ que sirve de ancla
  onEdit,
  onDelete,
  onClose,
  extraOptions = '',
}) {
  const menuRef = useRef(null);
  const [pos, setPos] = useState({ top: 0, left: 0 });

  const handleEdit = () => {
    onEdit?.();          // acción real (abrir modal, etc.)
    onClose?.();         // cierra el menú
  };

  const handleDelete = () => {
    onDelete?.();
    onClose?.();
  };

  /* ── calcula posición ───────────────────────────────────── */
  useEffect(() => {
    if (!anchorEl) return;
    const { bottom, right } = anchorEl.getBoundingClientRect();
    setPos({ top: bottom + 4, left: right - MENU_W });   // 4 px de separación
  }, [anchorEl]);

  /* ── click-fuera para cerrar ────────────────────────────── */
  useEffect(() => {
    const handler = (e) => {
      if (
        !menuRef.current?.contains(e.target) &&
        !anchorEl?.contains(e.target)
      ) {
        onClose?.();
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [anchorEl, onClose]);

  /* ── render en portal ───────────────────────────────────── */
  return createPortal(
    <ul
      ref={menuRef}
      className="row-menu"
      style={{
        position: 'fixed',
        top: pos.top,
        left: pos.left,
        zIndex: 10_000,
        margin: 0,
        padding: 0,
        listStyle: 'none',
        width: MENU_W,
        background: '#fff',
        borderRadius: 6,
        boxShadow: '0 4px 12px rgba(0,0,0,.15)',
      }}
    >
      <li>
        <button onClick={handleEdit}>Editar</button>
      </li>
      {onDelete && (
        <li>
          <button onClick={handleDelete}>Borrar</button>
        </li>
      )}
      {extraOptions}
    </ul>,
    document.body
  );
}
