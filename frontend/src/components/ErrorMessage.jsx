import PropTypes from 'prop-types';
import { AlertOctagon, RefreshCw, Terminal } from 'lucide-react';

export default function ErrorMessage({ title, message, onRetry, showBackendGuide }) {
  return (
    <div style={{
      background: '#fef2f2',
      border: '1px solid #fecaca',
      borderRadius: '14px',
      padding: '1.5rem',
      margin: '1.5rem 0',
      boxShadow: '0 4px 12px rgba(220, 38, 38, 0.05)',
    }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.85rem' }}>
        <div style={{
          background: '#dc2626',
          color: '#ffffff',
          padding: '0.45rem',
          borderRadius: '10px',
          display: 'flex',
          marginTop: '2px',
        }}>
          <AlertOctagon size={22} />
        </div>
        <div style={{ flex: 1 }}>
          <h4 style={{ margin: '0 0 0.35rem 0', color: '#991b1b', fontSize: '1.05rem', fontWeight: 800 }}>
            {title || 'Enhancement Request Failed'}
          </h4>
          <p style={{ margin: 0, color: '#7f1d1d', fontSize: '0.9rem', lineHeight: 1.5 }}>
            {message}
          </p>

          {showBackendGuide && (
            <div style={{
              marginTop: '1rem',
              background: '#ffffff',
              border: '1px solid #fca5a5',
              borderRadius: '8px',
              padding: '0.85rem',
              fontSize: '0.82rem',
              color: '#1e293b',
            }}>
              <div style={{ fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.4rem', color: '#991b1b' }}>
                <Terminal size={14} /> Run backend server in Powershell:
              </div>
              <pre style={{
                background: '#0f172a',
                color: '#38bdf8',
                padding: '0.65rem 0.85rem',
                borderRadius: '6px',
                fontFamily: 'monospace',
                fontSize: '0.78rem',
                overflowX: 'auto',
                margin: 0,
              }}>
{`cd /d "D:\\MINI PROJECT-ImageSuperResolution\\ImageSuperResolution\\backend"
.\\venv\\Scripts\\activate
python -m uvicorn main:app --host 127.0.0.1 --port 8000`}
              </pre>
            </div>
          )}

          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="btn"
              style={{
                marginTop: '1rem',
                background: '#dc2626',
                color: '#ffffff',
                fontSize: '0.85rem',
                padding: '0.45rem 1rem',
              }}
            >
              <RefreshCw size={14} /> Try Again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

ErrorMessage.propTypes = {
  title: PropTypes.string,
  message: PropTypes.string.isRequired,
  onRetry: PropTypes.func,
  showBackendGuide: PropTypes.bool,
};
