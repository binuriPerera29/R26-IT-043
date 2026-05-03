// frontend/src/services/api.js
// Central API service — all backend calls go through here.

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5002";

/**
 * Sends a fundus image to the VDS (Visibility Degradation Score) endpoint
 * and returns the VDS result.
 *
 * @param {File} imageFile  - image selected by the user
 * @returns {Promise<object>} - { success, result: { vds, grade, predicted_class, ... } }
 */
export async function analyzeVDS(imageFile) {
  const formData = new FormData();
  formData.append("image", imageFile);

  const response = await fetch(`${BASE_URL}/api/vds/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `Server error ${response.status}`);
  }

  return response.json();
}

/**
 * Health-check: confirms the VDS model is loaded on the backend.
 * @returns {Promise<object>}
 */
export async function checkVDSHealth() {
  const response = await fetch(`${BASE_URL}/api/vds/health`);
  return response.json();
}
