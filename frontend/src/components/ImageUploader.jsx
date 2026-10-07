import { useState, useRef } from 'react';
import PropTypes from 'prop-types';
import { UploadCloud, Image as ImageIcon, X, FileImage, AlertTriangle } from 'lucide-react';

const MAX_SIZE_MB = 10;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;
const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];

export default function ImageUploader({ selectedFile, fileMeta, previewUrl, onSelectFile, onClearFile }) {
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const fileInputRef = useRef(null);

  const processFile = (file) => {
    setErrorMessage(null);
    if (!file) return;

    if (!ALLOWED_TYPES.includes(file.type) && !file.name.match(/\.(png|jpe?g|webp)$/i)) {
      setErrorMessage('Unsupported file format. Please upload a PNG, JPG, JPEG, or WEBP image.');
      return;
    }

    if (file.size > MAX_SIZE_BYTES) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
      setErrorMessage(`File size (${sizeMb} MB) exceeds maximum allowed limit of ${MAX_SIZE_MB} MB.`);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      onSelectFile(file, {
        width: img.width,
        height: img.height,
        size: file.size,
        formattedSize: file.size < 1024 * 1024
          ? `${(file.size / 1024).toFixed(1)} KB`
          : `${(file.size / (1024 * 1024)).toFixed(2)} MB`,
      }, objectUrl);
    };
    img.onerror = () => {
      setErrorMessage('Failed to read image dimensions. File may be corrupted.');
    };
    img.src = objectUrl;
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  return (
    <div style={{ marginBottom: '1.75rem' }}>
      {!selectedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          style={{
            border: `2px dashed ${isDragging ? '#2563eb' : '#94a3b8'}`,
            borderRadius: '16px',
            padding: '2.5rem 1.5rem',
            background: isDragging
              ? 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)'
              : 'linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%)',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'all 0.25s ease',
            boxShadow: isDragging ? '0 0 20px rgba(37, 99, 235, 0.2)' : 'inset 0 2px 4px rgba(0,0,0,0.02)',
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png, image/jpeg, image/jpg, image/webp"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />

          <div style={{
            width: '64px',
            height: '64px',
            borderRadius: '16px',
            background: 'linear-gradient(135deg, #2563eb 0%, #06b6d4 100%)',
            color: '#ffffff',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1rem auto',
            boxShadow: '0 6px 16px rgba(37, 99, 235, 0.25)',
          }}>
            <UploadCloud size={32} />
          </div>

          <h3 style={{ margin: '0 0 0.35rem 0', color: '#0f172a', fontSize: '1.15rem', fontWeight: 700 }}>
            Drag & drop your image here
          </h3>
          <p style={{ margin: '0 0 1rem 0', fontSize: '0.9rem', color: '#64748b' }}>
            or click to browse files from your device
          </p>

          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
            background: '#ffffff',
            border: '1px solid #cbd5e1',
            padding: '0.35rem 0.85rem',
            borderRadius: '20px',
            fontSize: '0.8rem',
            color: '#475569',
            fontWeight: 600,
          }}>
            <FileImage size={14} color="#2563eb" />
            <span>Supported: PNG, JPG, JPEG, WEBP (Max {MAX_SIZE_MB} MB)</span>
          </div>
        </div>
      ) : (
        <div className="card-container" style={{ padding: '1.25rem' }}>
          {/* Metadata banner */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: '#eff6ff',
            border: '1px solid #bfdbfe',
            borderRadius: '10px',
            padding: '0.75rem 1rem',
            marginBottom: '1.25rem',
            flexWrap: 'wrap',
            gap: '0.75rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{
                background: '#2563eb',
                color: '#ffffff',
                padding: '0.4rem',
                borderRadius: '8px',
                display: 'flex',
              }}>
                <ImageIcon size={20} />
              </div>
              <div>
                <div style={{ fontWeight: 700, color: '#0f172a', fontSize: '0.92rem' }}>
                  {selectedFile.name}
                </div>
                <div style={{ fontSize: '0.8rem', color: '#475569' }}>
                  Resolution: <strong>{fileMeta?.width} × {fileMeta?.height}</strong> &nbsp;|&nbsp; Size: <strong>{fileMeta?.formattedSize}</strong>
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={onClearFile}
              className="btn btn-secondary"
              style={{ padding: '0.4rem 0.85rem', fontSize: '0.82rem', color: '#dc2626' }}
            >
              <X size={15} /> Remove File
            </button>
          </div>

          {/* Preview Image */}
          <div style={{
            textAlign: 'center',
            background: '#f8fafc',
            border: '1px solid #e2e8f0',
            borderRadius: '12px',
            padding: '1rem',
            overflow: 'hidden',
          }}>
            <img
              src={previewUrl}
              alt="Original Upload Preview"
              style={{
                maxWidth: '100%',
                maxHeight: '400px',
                borderRadius: '8px',
                objectFit: 'contain',
                boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
              }}
            />
            <div style={{ marginTop: '0.5rem', fontSize: '0.82rem', color: '#64748b', fontWeight: 600 }}>
              Original Input Image ({fileMeta?.width} × {fileMeta?.height})
            </div>
          </div>
        </div>
      )}

      {errorMessage && (
        <div style={{
          marginTop: '0.75rem',
          background: '#fef2f2',
          border: '1px solid #fecaca',
          color: '#991b1b',
          padding: '0.75rem 1rem',
          borderRadius: '10px',
          fontSize: '0.88rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.6rem',
        }}>
          <AlertTriangle size={18} color="#dc2626" />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
}

ImageUploader.propTypes = {
  selectedFile: PropTypes.object,
  fileMeta: PropTypes.object,
  previewUrl: PropTypes.string,
  onSelectFile: PropTypes.func.isRequired,
  onClearFile: PropTypes.func.isRequired,
};
