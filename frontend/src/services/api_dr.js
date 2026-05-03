import axios from 'axios'

// If VITE_API_URL env var is set use it, otherwise call Flask directly on 5000
// This avoids any Vite proxy issues
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5001/api'

export const analyzeRetina = async (imageFile) => {
  const formData = new FormData()
  formData.append('image', imageFile)

  const response = await axios.post(`${API_BASE}/analyze`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  })
  return response.data
}

export const checkHealth = async () => {
  const response = await axios.get(`${API_BASE}/health`, { timeout: 5000 })
  return response.data
}