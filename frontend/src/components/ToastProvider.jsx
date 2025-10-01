import React, {
  createContext,
  useContext,
  useState,
  useCallback,
} from 'react';
import { createPortal } from 'react-dom';
import '../styles/Toast.css';

const ToastContext = createContext();

export const ToastProvider = ({ children }) => {
/* estado único (puedes convertirlo en array si quieres cola) */
  const [toast, setToast] = useState(null); // { variant, message }

  /* helpers visibles desde el hook */
  const open = useCallback((msg, variant = 'success') => {
    setToast({ message: msg, variant });
    /* autohide tras 5 s */
    setTimeout(() => setToast(null), 5000);
  }, []);

  const api = {
    showSuccess: msg => open(msg, 'success'),
    showError:   msg => open(msg, 'error'),
    showWarning: msg => open(msg, 'warning'),
    showInfo:    msg => open(msg, 'info'),
  };

  return (
    <ToastContext.Provider value={api}>
      {children}

      {/* portal para sacar el toast fuera del flujo */}
      {toast &&
        createPortal(
          <ToastCard
            variant={toast.variant}
            message={toast.message}
            onClose={() => setToast(null)}
          />,
          document.body
        )}
    </ToastContext.Provider>
  );
};

const Icon = ({ children }) => (
  <svg
    width="20" height="20" viewBox="0 0 24 24"
    fill="none" stroke="currentColor" strokeWidth="2"
    strokeLinecap="round" strokeLinejoin="round"
  >
    {children}
  </svg>
);

const icons = {
  error: (
    <Icon>
      <circle cx="12" cy="12" r="10" />
      <line x1="15" y1="9" x2="9" y2="15" />
      <line x1="9" y1="9" x2="15" y2="15" />
    </Icon>
  ),
  warning: (
    <Icon>
      <polygon points="12 4 2 20 22 20" />
      <line x1="12" y1="8" x2="12" y2="13" />
      <circle cx="12" cy="17" r="1" />
    </Icon>
  ),
  info: (
    <Icon>
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8"  x2="12" y2="12" />
      <circle cx="12" cy="16" r="1" />
    </Icon>
  ),
  success: (
    <Icon>
      <circle cx="12" cy="12" r="10" />
      <polyline points="16 8 11 13 8 10" />
    </Icon>
  ),
};

const titles = {
  success: 'Success',
  error:   'Error',
  warning: 'Warning',
  info:    'Information',
};

function ToastCard({ variant, message, onClose }) {
  return (
    <div className={`toast-card ${variant}`} data-type={variant} data-theme="light">
      <button className="toast-close" onClick={onClose}>×</button>

      <span className="toast-icon">{icons[variant]}</span>

      <div>
        <div className="toast-title">{titles[variant]}</div>
        <div className="toast-message">{message}</div>
      </div>
    </div>
  );
}

export const useToast = () => {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used inside <ToastProvider>');
  return ctx;
};
