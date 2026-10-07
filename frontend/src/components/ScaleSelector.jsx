import PropTypes from 'prop-types';
import { Zap, Layers, Cpu } from 'lucide-react';

const SCALE_CONFIGS = [
  {
    scale: 2,
    title: '2× Enhancement',
    model: 'Student RFDB',
    params: '~186K Params',
    description: 'Fast lightweight model optimized for edge devices.',
    icon: Zap,
    color: '#2563eb',
    bgColor: '#eff6ff',
    borderColor: '#3b82f6',
  },
  {
    scale: 4,
    title: '4× Enhancement',
    model: 'Student RFDB',
    params: '~436K Params',
    description: 'Fine detail synthesis via knowledge distillation.',
    icon: Layers,
    color: '#7c3aed',
    bgColor: '#f5f3ff',
    borderColor: '#8b5cf6',
  },
  {
    scale: 8,
    title: '8× Enhancement',
    model: 'SwinIR-M Transformer',
    params: '~11.8M Params',
    description: 'High-capacity Swin Vision Transformer engine.',
    icon: Cpu,
    color: '#0891b2',
    bgColor: '#ecfeff',
    borderColor: '#06b6d4',
  },
];

export default function ScaleSelector({ selectedScale, onSelectScale }) {
  return (
    <div style={{ marginBottom: '1.75rem' }}>
      <h3 style={{ fontSize: '1.1rem', marginBottom: '0.85rem', color: '#0f172a', fontWeight: 700 }}>
        ⚙️ Choose Enhancement Scale & Model Routing
      </h3>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: '1rem',
      }}>
        {SCALE_CONFIGS.map((cfg) => {
          const isSelected = selectedScale === cfg.scale;
          const IconComp = cfg.icon;

          return (
            <div
              key={cfg.scale}
              onClick={() => onSelectScale(cfg.scale)}
              style={{
                background: isSelected ? '#ffffff' : '#f8fafc',
                border: `2px solid ${isSelected ? cfg.borderColor : '#e2e8f0'}`,
                borderRadius: '14px',
                padding: '1.25rem',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: isSelected ? `0 6px 20px ${cfg.color}20` : 'none',
                position: 'relative',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
              }}
            >
              {isSelected && (
                <div style={{
                  position: 'absolute',
                  top: '10px',
                  right: '12px',
                  background: cfg.color,
                  color: '#ffffff',
                  fontSize: '0.7rem',
                  fontWeight: 800,
                  padding: '2px 8px',
                  borderRadius: '10px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                }}>
                  Selected
                </div>
              )}

              <div>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  marginBottom: '0.75rem',
                }}>
                  <div style={{
                    width: '40px',
                    height: '40px',
                    borderRadius: '10px',
                    background: cfg.bgColor,
                    color: cfg.color,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 800,
                    fontSize: '1.1rem',
                  }}>
                    {cfg.scale}×
                  </div>
                  <div>
                    <h4 style={{ margin: 0, fontSize: '1rem', color: '#0f172a', fontWeight: 700 }}>
                      {cfg.title}
                    </h4>
                    <span style={{ fontSize: '0.8rem', color: cfg.color, fontWeight: 700 }}>
                      {cfg.model}
                    </span>
                  </div>
                </div>

                <p style={{ margin: '0 0 0.75rem 0', fontSize: '0.84rem', color: '#64748b', lineHeight: 1.45 }}>
                  {cfg.description}
                </p>
              </div>

              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontSize: '0.78rem',
                fontWeight: 700,
                color: cfg.color,
                background: cfg.bgColor,
                padding: '0.35rem 0.65rem',
                borderRadius: '8px',
                width: 'fit-content',
              }}>
                <IconComp size={14} />
                <span>{cfg.params}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

ScaleSelector.propTypes = {
  selectedScale: PropTypes.number.isRequired,
  onSelectScale: PropTypes.func.isRequired,
};
