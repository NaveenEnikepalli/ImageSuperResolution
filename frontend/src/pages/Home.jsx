import PropTypes from 'prop-types';
import { ArrowRight, Sparkles, Brain, Zap, Smartphone } from 'lucide-react';

export default function Home({ onStartEnhancing }) {
  return (
    <div>
      {/* HERO SECTION */}
      <section style={{
        background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)',
        borderRadius: '24px',
        padding: '3.5rem 2rem',
        color: '#ffffff',
        textAlign: 'center',
        marginBottom: '2.5rem',
        boxShadow: '0 20px 40px -15px rgba(15, 23, 42, 0.5)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        position: 'relative',
        overflow: 'hidden',
      }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.5rem',
          background: 'rgba(56, 189, 248, 0.15)',
          border: '1px solid rgba(56, 189, 248, 0.3)',
          color: '#38bdf8',
          padding: '0.35rem 1rem',
          borderRadius: '20px',
          fontSize: '0.82rem',
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
          marginBottom: '1.25rem',
        }}>
          <Brain size={16} /> Deep Learning Vision Suite
        </div>

        <h1 style={{
          fontSize: '3rem',
          fontWeight: 800,
          marginBottom: '1rem',
          lineHeight: 1.2,
          letterSpacing: '-1px',
        }} className="hero-gradient-text">
          AI Image Super Resolution
        </h1>

        <p style={{
          fontSize: '1.15rem',
          maxWidth: '740px',
          margin: '0 auto 2.25rem auto',
          color: '#94a3b8',
          lineHeight: 1.6,
        }}>
          Lightweight Image Enhancement using Knowledge Distillation. Reconstruct fine textures and clarity from low-resolution images with real-time neural network architectures.
        </p>

        {/* Visual Pipeline Banner */}
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '0.85rem',
          flexWrap: 'wrap',
          background: 'rgba(255, 255, 255, 0.05)',
          padding: '0.85rem 1.5rem',
          borderRadius: '14px',
          maxWidth: '680px',
          marginBottom: '2.25rem',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}>
          <span style={{ color: '#cbd5e1', fontWeight: 600, fontSize: '0.9rem' }}>📷 Low Res Input</span>
          <span style={{ color: '#38bdf8', fontWeight: 800 }}>──►</span>
          <span style={{ color: '#38bdf8', fontWeight: 700, fontSize: '0.88rem', background: 'rgba(56, 189, 248, 0.15)', padding: '0.25rem 0.65rem', borderRadius: '6px' }}>
            ⚡ AI Neural Engine
          </span>
          <span style={{ color: '#38bdf8', fontWeight: 800 }}>──►</span>
          <span style={{ color: '#4ade80', fontWeight: 700, fontSize: '0.9rem' }}>✨ High Res Output</span>
        </div>

        {/* Hero CTA Button */}
        <div>
          <button
            type="button"
            onClick={onStartEnhancing}
            className="btn btn-primary"
            style={{
              padding: '0.9rem 2.25rem',
              fontSize: '1.05rem',
              borderRadius: '12px',
            }}
          >
            <Sparkles size={20} /> Start Enhancing Now <ArrowRight size={18} />
          </button>
        </div>
      </section>

      {/* MODEL ARCHITECTURE SECTION */}
      <section style={{ marginBottom: '3rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0f172a', marginBottom: '0.4rem' }}>
            🔍 Model Architecture & Enhancement Scales
          </h2>
          <p style={{ color: '#64748b', fontSize: '0.95rem' }}>
            Automatic routing selects optimized student distillation models or Swin Transformer baselines
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1.5rem',
        }}>
          {/* 2x Card */}
          <div className="card-container" style={{
            padding: '1.75rem',
            textAlign: 'center',
            border: '2px solid #3b82f6',
            background: 'linear-gradient(180deg, #ffffff 0%, #eff6ff 100%)',
          }}>
            <div style={{
              background: '#2563eb',
              color: '#ffffff',
              width: '52px',
              height: '52px',
              borderRadius: '14px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 800,
              fontSize: '1.35rem',
              margin: '0 auto 1rem auto',
              boxShadow: '0 6px 16px rgba(37, 99, 235, 0.3)',
            }}>
              2×
            </div>
            <h3 style={{ margin: '0 0 0.35rem 0', color: '#0f172a', fontWeight: 800 }}>Student RFDB</h3>
            <div style={{ color: '#2563eb', fontWeight: 700, fontSize: '0.88rem', marginBottom: '0.85rem' }}>
              ~186,603 Parameters
            </div>
            <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b', lineHeight: 1.5 }}>
              Ultrafast Residual Feature Distillation Block network tailored for instant 2× resolution upscaling on edge devices.
            </p>
          </div>

          {/* 4x Card */}
          <div className="card-container" style={{
            padding: '1.75rem',
            textAlign: 'center',
            border: '2px solid #8b5cf6',
            background: 'linear-gradient(180deg, #ffffff 0%, #f5f3ff 100%)',
          }}>
            <div style={{
              background: '#7c3aed',
              color: '#ffffff',
              width: '52px',
              height: '52px',
              borderRadius: '14px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 800,
              fontSize: '1.35rem',
              margin: '0 auto 1rem auto',
              boxShadow: '0 6px 16px rgba(124, 58, 237, 0.3)',
            }}>
              4×
            </div>
            <h3 style={{ margin: '0 0 0.35rem 0', color: '#0f172a', fontWeight: 800 }}>Student RFDB</h3>
            <div style={{ color: '#7c3aed', fontWeight: 700, fontSize: '0.88rem', marginBottom: '0.85rem' }}>
              ~436,011 Parameters
            </div>
            <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b', lineHeight: 1.5 }}>
              Deep resolution synthesis model trained via feature map distillation to recover crisp edges and textures.
            </p>
          </div>

          {/* 8x Card */}
          <div className="card-container" style={{
            padding: '1.75rem',
            textAlign: 'center',
            border: '2px solid #06b6d4',
            background: 'linear-gradient(180deg, #ffffff 0%, #ecfeff 100%)',
          }}>
            <div style={{
              background: '#0891b2',
              color: '#ffffff',
              width: '52px',
              height: '52px',
              borderRadius: '14px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 800,
              fontSize: '1.35rem',
              margin: '0 auto 1rem auto',
              boxShadow: '0 6px 16px rgba(6, 182, 212, 0.3)',
            }}>
              8×
            </div>
            <h3 style={{ margin: '0 0 0.35rem 0', color: '#0f172a', fontWeight: 800 }}>SwinIR-M Transformer</h3>
            <div style={{ color: '#0891b2', fontWeight: 700, fontSize: '0.88rem', marginBottom: '0.85rem' }}>
              ~11.8M Parameters
            </div>
            <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b', lineHeight: 1.5 }}>
              High-capacity Shifted-Window Vision Transformer baseline delivering high-definition synthesis at 8× magnification.
            </p>
          </div>
        </div>
      </section>

      {/* TECHNICAL HIGHLIGHTS SECTION */}
      <section style={{ marginBottom: '1.5rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0f172a', marginBottom: '0.4rem' }}>
            ✨ Key Technical Features
          </h2>
          <p style={{ color: '#64748b', fontSize: '0.95rem' }}>
            Built with modern computer vision research and clean software design
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1.25rem',
        }}>
          <div className="card-container" style={{ padding: '1.5rem' }}>
            <div style={{
              background: '#eff6ff',
              color: '#2563eb',
              width: '42px',
              height: '42px',
              borderRadius: '10px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '1rem',
            }}>
              <Brain size={22} />
            </div>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '0.4rem', color: '#0f172a' }}>Knowledge Distillation</h3>
            <p style={{ margin: 0, fontSize: '0.88rem', color: '#64748b', lineHeight: 1.5 }}>
              Transfers complex high-frequency representation capabilities from Teacher models into compact Student RFDB blocks without quality loss.
            </p>
          </div>

          <div className="card-container" style={{ padding: '1.5rem' }}>
            <div style={{
              background: '#ecfeff',
              color: '#0891b2',
              width: '42px',
              height: '42px',
              borderRadius: '10px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '1rem',
            }}>
              <Zap size={22} />
            </div>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '0.4rem', color: '#0f172a' }}>Fast PyTorch Caching</h3>
            <p style={{ margin: 0, fontSize: '0.88rem', color: '#64748b', lineHeight: 1.5 }}>
              Persistent backend caching keeps model weights loaded in memory for instantaneous sub-second inference execution.
            </p>
          </div>

          <div className="card-container" style={{ padding: '1.5rem' }}>
            <div style={{
              background: '#f5f3ff',
              color: '#7c3aed',
              width: '42px',
              height: '42px',
              borderRadius: '10px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '1rem',
            }}>
              <Smartphone size={22} />
            </div>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '0.4rem', color: '#0f172a' }}>Responsive React UI</h3>
            <p style={{ margin: 0, fontSize: '0.88rem', color: '#64748b', lineHeight: 1.5 }}>
              Built with React + Vite for lightning-fast UI updates, side-by-side desktop image inspection, and mobile vertical stacking.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

Home.propTypes = {
  onStartEnhancing: PropTypes.func.isRequired,
};
