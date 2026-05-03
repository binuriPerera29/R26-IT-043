/**
 * api.js — Service for the EYE OCT backend
 * Calls Flask directly on port 5000
 */

const BASE_URL = 'http://localhost:5000/api'

/**
 * Predict OCT disease + Grad-CAM + medical explanation
 * @param {File} file - Image file
 * @returns {Promise<Object>} prediction result
 */
export async function predictOCT(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const err = await response.json().catch(() => ({ error: `HTTP ${response.status}` }))
    throw { response: { data: err } }
  }

  return response.json()
}

/**
 * Health check
 */
export async function healthCheck() {
  const response = await fetch(`${BASE_URL}/health`)
  return response.json()
}