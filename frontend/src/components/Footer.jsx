import { Cpu, Zap } from 'lucide-react';

export default function Footer() {
  return (
    <footer style={{
      marginTop: '4rem',
      paddingTop: '1.75rem',
      borderTop: '1px solid #e2e8f0',
      textAlign: 'center',
      color: '#64748b',
      fontSize: '0.85rem',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem', marginBottom: '0.35rem' }}>
        <Zap size={16} color="#2563eb" />
        <span style={{ fontWeight: 700, color: '#0f172a' }}>
          PixelLift: Lightweight AI-Based Image Enhancement using Knowledge Distillation
        </span>
      </div>
      <p style={{ margin: '0 0 0.5rem 0', color: '#64748b', fontSize: '0.82rem' }}>
        React + Vite Frontend & FastAPI Backend | Student RFDB Distillation (2×, 4×) & SwinIR Transformer (8×)
      </p>
      <div style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.6rem',
        background: '#ffffff',
        border: '1px solid #e2e8f0',
        padding: '0.25rem 0.75rem',
        borderRadius: '20px',
        fontSize: '0.75rem',
        color: '#475569',
        fontWeight: 600,
      }}>
        <Cpu size={12} color="#06b6d4" />
        Edge-Device Optimized Inference Engine
      </div>
    </footer>
  );
}
