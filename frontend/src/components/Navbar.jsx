import { useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import { Sparkles, Activity, CheckCircle2, AlertCircle } from 'lucide-react';
import { checkBackendHealth } from '../services/api';

export default function Navbar({ activePage, setActivePage }) {
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    let isMounted = true;
    async function verifyHealth() {
      const res = await checkBackendHealth();
      if (isMounted) {
        setBackendStatus(res.isHealthy ? 'online' : 'offline');
      }
    }
    verifyHealth();
    const interval = setInterval(verifyHealth, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header style={{
      background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
      border: '1px solid #e2e8f0',
      borderRadius: '16px',
      padding: '1.1rem 1.5rem',
      marginBottom: '2rem',
      boxShadow: '0 4px 14px rgba(0, 0, 0, 0.03)',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem',
      }}>
        {/* Brand Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            background: 'linear-gradient(135deg, #2563eb 0%, #06b6d4 100%)',
            width: '48px',
            height: '48px',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#ffffff',
            boxShadow: '0 4px 12px rgba(37, 99, 235, 0.3)',
          }}>
            <Sparkles size={26} />
          </div>
          <div>
            <h1 style={{
              fontSize: '1.5rem',
              fontWeight: 800,
              letterSpacing: '-0.5px',
              margin: 0,
              background: 'linear-gradient(135deg, #1e293b 0%, #2563eb 50%, #06b6d4 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              AI Image Super Resolution
            </h1>
            <p style={{ margin: 0, fontSize: '0.85rem', color: '#64748b', fontWeight: 500 }}>
              Lightweight Deep Learning Enhancement via Knowledge Distillation
            </p>
          </div>
        </div>

        {/* Navigation & Status Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', flexWrap: 'wrap' }}>
          {/* Page Switcher */}
          <nav style={{
            display: 'flex',
            background: '#f1f5f9',
            padding: '4px',
            borderRadius: '10px',
            border: '1px solid #e2e8f0',
          }}>
            <button
              type="button"
              onClick={() => setActivePage('Home')}
              style={{
                border: 'none',
                background: activePage === 'Home' ? '#ffffff' : 'transparent',
                color: activePage === 'Home' ? '#2563eb' : '#64748b',
                fontWeight: activePage === 'Home' ? 700 : 600,
                fontSize: '0.88rem',
                padding: '0.45rem 1.1rem',
                borderRadius: '8px',
                cursor: 'pointer',
                boxShadow: activePage === 'Home' ? '0 2px 6px rgba(0,0,0,0.06)' : 'none',
                transition: 'all 0.2s ease',
              }}
            >
              Home
            </button>
            <button
              type="button"
              onClick={() => setActivePage('Enhance')}
              style={{
                border: 'none',
                background: activePage === 'Enhance' ? '#ffffff' : 'transparent',
                color: activePage === 'Enhance' ? '#2563eb' : '#64748b',
                fontWeight: activePage === 'Enhance' ? 700 : 600,
                fontSize: '0.88rem',
                padding: '0.45rem 1.1rem',
                borderRadius: '8px',
                cursor: 'pointer',
                boxShadow: activePage === 'Enhance' ? '0 2px 6px rgba(0,0,0,0.06)' : 'none',
                transition: 'all 0.2s ease',
              }}
            >
              Enhance Image
            </button>
          </nav>

          {/* Backend Status Pill */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            padding: '0.4rem 0.85rem',
            borderRadius: '20px',
            fontSize: '0.78rem',
            fontWeight: 700,
            background: backendStatus === 'online' ? '#ecfdf5' : backendStatus === 'offline' ? '#fef2f2' : '#f8fafc',
            color: backendStatus === 'online' ? '#059669' : backendStatus === 'offline' ? '#dc2626' : '#64748b',
            border: `1px solid ${backendStatus === 'online' ? '#a7f3d0' : backendStatus === 'offline' ? '#fecaca' : '#cbd5e1'}`,
          }}>
            {backendStatus === 'online' && <CheckCircle2 size={14} />}
            {backendStatus === 'offline' && <AlertCircle size={14} />}
            {backendStatus === 'checking' && <Activity size={14} className="animate-spin" />}
            <span>{backendStatus === 'online' ? 'API Online' : backendStatus === 'offline' ? 'API Offline' : 'Checking API'}</span>
          </div>
        </div>
      </div>
    </header>
  );
}

Navbar.propTypes = {
  activePage: PropTypes.string.isRequired,
  setActivePage: PropTypes.func.isRequired,
};
