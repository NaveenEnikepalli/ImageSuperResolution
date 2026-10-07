import PropTypes from 'prop-types';
import { Download, BarChart2, Cpu, Maximize2, Clock, Check } from 'lucide-react';

export default function ResultDetails({
  resultData,
  onDownload,
}) {
  const {
    modelName,
    scale,
    inputResolution,
    outputResolution,
    inferenceTime,
    filename,
  } = resultData;

  const handleDownloadClick = () => {
    let saveName = filename ? `enhanced_${filename}` : `enhanced_x${scale}.png`;
    if (!saveName.match(/\.(png|jpe?g|webp)$/i)) {
      saveName = `${saveName}.png`;
    }
    onDownload(saveName);
  };

  return (
    <div style={{ marginTop: '1.75rem' }}>
      {/* Metadata Summary Grid */}
      <div style={{
        background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
        border: '1px solid #cbd5e1',
        borderRadius: '16px',
        padding: '1.5rem',
        marginBottom: '1.5rem',
        boxShadow: '0 4px 14px rgba(0,0,0,0.03)',
      }}>
        <h4 style={{ margin: '0 0 1.25rem 0', color: '#0f172a', fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 800 }}>
          <BarChart2 size={20} color="#2563eb" /> Performance & Model Metadata Summary
        </h4>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
          gap: '1rem',
        }}>
          <div style={{ background: '#ffffff', padding: '1rem', borderRadius: '12px', border: '1px solid #e2e8f0', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.3rem', textTransform: 'uppercase' }}>
              <Cpu size={14} color="#2563eb" /> Model Architecture
            </div>
            <div style={{ fontSize: '1rem', color: '#2563eb', fontWeight: 800, marginTop: '0.35rem' }}>
              {modelName}
            </div>
          </div>

          <div style={{ background: '#ffffff', padding: '1rem', borderRadius: '12px', border: '1px solid #e2e8f0', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.3rem', textTransform: 'uppercase' }}>
              <Check size={14} color="#059669" /> Scale Factor
            </div>
            <div style={{ fontSize: '1.1rem', color: '#059669', fontWeight: 800, marginTop: '0.35rem' }}>
              {scale}×
            </div>
          </div>

          <div style={{ background: '#ffffff', padding: '1rem', borderRadius: '12px', border: '1px solid #e2e8f0', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.3rem', textTransform: 'uppercase' }}>
              <Maximize2 size={14} color="#0f172a" /> Input Resolution
            </div>
            <div style={{ fontSize: '0.95rem', color: '#0f172a', fontWeight: 700, marginTop: '0.35rem' }}>
              {inputResolution}
            </div>
          </div>

          <div style={{ background: '#ffffff', padding: '1rem', borderRadius: '12px', border: '1px solid #e2e8f0', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.3rem', textTransform: 'uppercase' }}>
              <Maximize2 size={14} color="#7c3aed" /> Output Resolution
            </div>
            <div style={{ fontSize: '0.95rem', color: '#7c3aed', fontWeight: 800, marginTop: '0.35rem' }}>
              {outputResolution}
            </div>
          </div>

          <div style={{ background: '#ffffff', padding: '1rem', borderRadius: '12px', border: '1px solid #e2e8f0', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.3rem', textTransform: 'uppercase' }}>
              <Clock size={14} color="#ea580c" /> Inference Time
            </div>
            <div style={{ fontSize: '1rem', color: '#ea580c', fontWeight: 800, marginTop: '0.35rem' }}>
              {inferenceTime.toFixed(3)} s
            </div>
          </div>
        </div>
      </div>

      {/* Download Action Section */}
      <div style={{
        background: '#eff6ff',
        border: '1px solid #bfdbfe',
        borderRadius: '16px',
        padding: '1.25rem 1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem',
      }}>
        <div>
          <h4 style={{ margin: 0, color: '#1e293b', fontSize: '1.05rem', fontWeight: 800 }}>
            💾 Save High-Resolution Image
          </h4>
          <p style={{ margin: 0, color: '#475569', fontSize: '0.85rem' }}>
            Download original lossless PNG reconstructed at {outputResolution}
          </p>
        </div>

        <button
          type="button"
          onClick={handleDownloadClick}
          className="btn btn-primary"
          style={{ padding: '0.75rem 1.75rem', fontSize: '0.95rem' }}
        >
          <Download size={18} /> Download Enhanced Image ({outputResolution} PNG)
        </button>
      </div>
    </div>
  );
}

ResultDetails.propTypes = {
  resultData: PropTypes.shape({
    modelName: PropTypes.string.isRequired,
    scale: PropTypes.number.isRequired,
    inputResolution: PropTypes.string.isRequired,
    outputResolution: PropTypes.string.isRequired,
    inferenceTime: PropTypes.number.isRequired,
    filename: PropTypes.string,
  }).isRequired,
  onDownload: PropTypes.func.isRequired,
};
