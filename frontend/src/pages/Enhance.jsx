import { useState, useEffect } from 'react';
import { Sparkles } from 'lucide-react';
import ImageUploader from '../components/ImageUploader';
import ScaleSelector from '../components/ScaleSelector';
import ImageComparison from '../components/ImageComparison';
import ResultDetails from '../components/ResultDetails';
import LoadingState from '../components/LoadingState';
import ErrorMessage from '../components/ErrorMessage';
import { checkBackendHealth, enhanceImageViaApi } from '../services/api';

const SCALE_MODELS = {
  2: 'Student RFDB',
  4: 'Student RFDB',
  8: 'SwinIR-M Transformer',
};

export default function Enhance() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileMeta, setFileMeta] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [scale, setScale] = useState(4);
  const [backendStatus, setBackendStatus] = useState({ isHealthy: true, error: null });

  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [resultData, setResultData] = useState(null);

  useEffect(() => {
    let isMounted = true;
    async function verifyHealth() {
      const res = await checkBackendHealth();
      if (isMounted) {
        setBackendStatus(res);
      }
    }
    verifyHealth();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleSelectFile = (file, meta, objectUrl) => {
    setSelectedFile(file);
    setFileMeta(meta);
    setPreviewUrl(objectUrl);
    setResultData(null);
    setErrorMessage(null);
  };

  const handleClearFile = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setFileMeta(null);
    setPreviewUrl(null);
    setResultData(null);
    setErrorMessage(null);
  };

  const handleEnhanceClick = async () => {
    if (!selectedFile) return;

    // Verify backend health first
    const health = await checkBackendHealth();
    setBackendStatus(health);

    if (!health.isHealthy) {
      setErrorMessage(health.error || 'FastAPI backend server is not reachable.');
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const result = await enhanceImageViaApi(selectedFile, scale);
      setResultData(result);
    } catch (err) {
      setErrorMessage(err.message || 'Image enhancement failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = (filename) => {
    if (!resultData || !resultData.blob) return;
    const a = document.createElement('a');
    a.href = resultData.imageUrl;
    a.download = filename || `enhanced_${scale}x_${selectedFile?.name || 'image.png'}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div>
      <div style={{ marginBottom: '1.5rem' }}>
        <h2 style={{ fontSize: '1.75rem', color: '#0f172a', fontWeight: 800, margin: '0 0 0.35rem 0' }}>
          🔍 Enhance Your Image
        </h2>
        <p style={{ color: '#64748b', fontSize: '0.95rem', margin: 0 }}>
          Upload a low-resolution image, select target scale factor (2×, 4×, or 8×), and reconstruct fine high-definition details using deep neural networks.
        </p>
      </div>

      {!backendStatus.isHealthy && (
        <ErrorMessage
          title="FastAPI Backend Unavailable"
          message={backendStatus.error || 'Cannot communicate with the FastAPI backend server.'}
          showBackendGuide={true}
          onRetry={async () => {
            const h = await checkBackendHealth();
            setBackendStatus(h);
          }}
        />
      )}

      {/* Upload Component */}
      <ImageUploader
        selectedFile={selectedFile}
        fileMeta={fileMeta}
        previewUrl={previewUrl}
        onSelectFile={handleSelectFile}
        onClearFile={handleClearFile}
      />

      {/* Scale Selection */}
      {selectedFile && (
        <ScaleSelector
          selectedScale={scale}
          onSelectScale={(s) => {
            setScale(s);
            if (resultData) setResultData(null);
          }}
        />
      )}

      {/* Enhance Action Button */}
      {selectedFile && !isLoading && (
        <div style={{ textAlign: 'center', margin: '1.5rem 0' }}>
          <button
            type="button"
            onClick={handleEnhanceClick}
            disabled={!backendStatus.isHealthy}
            className="btn btn-primary"
            style={{
              padding: '0.85rem 2.25rem',
              fontSize: '1.05rem',
              borderRadius: '12px',
              width: '100%',
              maxWidth: '480px',
            }}
          >
            <Sparkles size={20} /> Enhance Image ({scale}× Resolution)
          </button>
        </div>
      )}

      {/* Loading Indicator */}
      {isLoading && (
        <LoadingState
          scale={scale}
          modelName={SCALE_MODELS[scale] || 'Student RFDB'}
        />
      )}

      {/* Enhancement Error Message */}
      {errorMessage && backendStatus.isHealthy && (
        <ErrorMessage
          title="Enhancement Failure"
          message={errorMessage}
          onRetry={handleEnhanceClick}
        />
      )}

      {/* Enhancement Result Section */}
      {resultData && previewUrl && (
        <div>
          <ImageComparison
            originalUrl={previewUrl}
            enhancedUrl={resultData.imageUrl}
            inputResolution={resultData.inputResolution}
            outputResolution={resultData.outputResolution}
            scale={resultData.scale}
          />

          <ResultDetails
            resultData={resultData}
            onDownload={handleDownload}
          />
        </div>
      )}
    </div>
  );
}
