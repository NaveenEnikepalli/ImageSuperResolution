import PropTypes from 'prop-types';
import { Sparkles, Image as ImageIcon } from 'lucide-react';

export default function ImageComparison({
  originalUrl,
  enhancedUrl,
  inputResolution,
  outputResolution,
  scale,
}) {
  return (
    <div style={{ marginTop: '2rem' }}>
      <h3 style={{ fontSize: '1.15rem', color: '#0f172a', fontWeight: 800, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Sparkles size={20} color="#2563eb" /> Enhancement Result Comparison
      </h3>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '1.25rem',
      }}>
        {/* Original Image Card */}
        <div className="card-container" style={{ padding: '1rem', display: 'flex', flexDirection: 'column' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: '#f8fafc',
            border: '1px solid #e2e8f0',
            borderRadius: '8px',
            padding: '0.5rem 0.85rem',
            marginBottom: '0.85rem',
          }}>
            <span style={{ fontWeight: 700, color: '#475569', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <ImageIcon size={16} /> Original Input
            </span>
            <span style={{ color: '#64748b', fontSize: '0.82rem', fontWeight: 600 }}>
              {inputResolution}
            </span>
          </div>

          <div style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#0f172a08',
            borderRadius: '10px',
            padding: '0.5rem',
            minHeight: '260px',
          }}>
            <img
              src={originalUrl}
              alt="Original Input"
              style={{
                maxWidth: '100%',
                maxHeight: '450px',
                objectFit: 'contain',
                borderRadius: '6px',
              }}
            />
          </div>
          <div style={{ textAlign: 'center', marginTop: '0.5rem', fontSize: '0.8rem', color: '#64748b' }}>
            Input Resolution ({inputResolution})
          </div>
        </div>

        {/* AI Enhanced Image Card */}
        <div className="card-container" style={{
          padding: '1rem',
          display: 'flex',
          flexDirection: 'column',
          border: '2px solid #bfdbfe',
          boxShadow: '0 6px 20px rgba(37, 99, 235, 0.08)',
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: '#eff6ff',
            border: '1px solid #bfdbfe',
            borderRadius: '8px',
            padding: '0.5rem 0.85rem',
            marginBottom: '0.85rem',
          }}>
            <span style={{ fontWeight: 800, color: '#1d4ed8', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Sparkles size={16} color="#06b6d4" /> AI Enhanced ({scale}×)
            </span>
            <span style={{ color: '#0891b2', fontWeight: 800, fontSize: '0.85rem' }}>
              {outputResolution}
            </span>
          </div>

          <div style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#0f172a08',
            borderRadius: '10px',
            padding: '0.5rem',
            minHeight: '260px',
          }}>
            <img
              src={enhancedUrl}
              alt="AI Enhanced Result"
              style={{
                maxWidth: '100%',
                maxHeight: '450px',
                objectFit: 'contain',
                borderRadius: '6px',
                boxShadow: '0 4px 14px rgba(0,0,0,0.1)',
              }}
            />
          </div>
          <div style={{ textAlign: 'center', marginTop: '0.5rem', fontSize: '0.8rem', color: '#1d4ed8', fontWeight: 600 }}>
            Reconstructed Output ({outputResolution})
          </div>
        </div>
      </div>
    </div>
  );
}

ImageComparison.propTypes = {
  originalUrl: PropTypes.string.isRequired,
  enhancedUrl: PropTypes.string.isRequired,
  inputResolution: PropTypes.string.isRequired,
  outputResolution: PropTypes.string.isRequired,
  scale: PropTypes.number.isRequired,
};
