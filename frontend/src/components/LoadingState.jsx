import PropTypes from 'prop-types';
import { Loader2, Sparkles } from 'lucide-react';

export default function LoadingState({ scale, modelName }) {
  return (
    <div style={{
      background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
      border: '2px solid #3b82f6',
      borderRadius: '16px',
      padding: '2.5rem 1.5rem',
      textAlign: 'center',
      margin: '1.75rem 0',
      boxShadow: '0 8px 24px rgba(37, 99, 235, 0.12)',
    }}>
      <div style={{
        position: 'relative',
        width: '64px',
        height: '64px',
        margin: '0 auto 1.25rem auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <Loader2
          size={56}
          color="#2563eb"
          className="animate-spin"
          style={{ position: 'absolute' }}
        />
        <Sparkles size={24} color="#06b6d4" />
      </div>

      <h3 style={{ margin: '0 0 0.5rem 0', color: '#0f172a', fontSize: '1.25rem', fontWeight: 800 }}>
        Enhancing Image at {scale}× Resolution
      </h3>

      <p style={{ margin: '0 0 1rem 0', color: '#64748b', fontSize: '0.92rem' }}>
        Processing forward pass with <strong>{modelName}</strong> PyTorch engine...
      </p>

      <div style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.5rem',
        background: '#eff6ff',
        border: '1px solid #bfdbfe',
        color: '#1d4ed8',
        padding: '0.4rem 1rem',
        borderRadius: '20px',
        fontSize: '0.82rem',
        fontWeight: 700,
      }}>
        <span>⚡ Knowledge Distillation & Neural Reconstruction</span>
      </div>
    </div>
  );
}

LoadingState.propTypes = {
  scale: PropTypes.number.isRequired,
  modelName: PropTypes.string.isRequired,
};
