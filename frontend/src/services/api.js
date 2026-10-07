/**
 * API Service Client for Image Super-Resolution Backend.
 *
 * Interfacing React frontend with FastAPI backend endpoints:
 * - GET /api/health
 * - GET /api/models
 * - POST /api/enhance
 */

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

/**
 * Check backend server health status.
 * @returns {Promise<{ isHealthy: boolean, data: object|null, error: string|null }>}
 */
export async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    if (response.ok) {
      const data = await response.json();
      return { isHealthy: true, data, error: null };
    }
    return { isHealthy: false, data: null, error: `Backend status ${response.status}` };
  } catch {
    return {
      isHealthy: false,
      data: null,
      error: `Cannot connect to FastAPI backend at ${API_BASE_URL}. Ensure the backend server is running.`,
    };
  }
}

/**
 * Fetch available model metadata.
 * @returns {Promise<{ success: boolean, data: object|null }>}
 */
export async function getModelsInfo() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/models`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    if (response.ok) {
      const data = await response.json();
      return { success: true, data };
    }
    return { success: false, data: null };
  } catch {
    return { success: false, data: null };
  }
}

/**
 * Send image to FastAPI backend for AI super-resolution enhancement.
 *
 * @param {File} file - Low resolution image file object.
 * @param {number} scale - Desired scale factor (2, 4, or 8).
 * @returns {Promise<{
 *   success: boolean,
 *   blob: Blob,
 *   imageUrl: string,
 *   inferenceTime: number,
 *   modelName: string,
 *   inputResolution: string,
 *   outputResolution: string,
 *   scale: number,
 *   filename: string
 * }>}
 */
export async function enhanceImageViaApi(file, scale) {
  const formData = new FormData();
  formData.append('image', file, file.name);
  formData.append('scale', String(scale));

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120000); // 120 seconds timeout

  try {
    const response = await fetch(`${API_BASE_URL}/api/enhance`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let errorMessage = `Server error (Status ${response.status})`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          errorMessage = typeof errorJson.detail === 'string' 
            ? errorJson.detail 
            : JSON.stringify(errorJson.detail);
        }
      } catch {
        // Response was not JSON
      }
      throw new Error(errorMessage);
    }

    const blob = await response.blob();
    const imageUrl = URL.createObjectURL(blob);

    const headers = response.headers;
    const inferenceTime = parseFloat(headers.get('X-Inference-Time') || '0.0');
    const modelName = headers.get('X-Model-Used') || (scale === 8 ? 'SwinIR-M Transformer' : `Student RFDB (x${scale})`);
    const inputResolution = headers.get('X-Input-Resolution') || 'N/A';
    const outputResolution = headers.get('X-Output-Resolution') || 'N/A';
    const scaleFactor = parseInt(headers.get('X-Scale-Factor') || String(scale), 10);

    return {
      success: true,
      blob,
      imageUrl,
      inferenceTime,
      modelName,
      inputResolution,
      outputResolution,
      scale: scaleFactor,
      filename: file.name,
    };
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('Image processing timed out after 120 seconds. Please try a smaller image.', { cause: err });
    }
    if (err.message && err.message.includes('Failed to fetch')) {
      throw new Error(`Cannot connect to FastAPI backend at ${API_BASE_URL}. Please verify the server is running.`, { cause: err });
    }
    throw err;
  }
}
