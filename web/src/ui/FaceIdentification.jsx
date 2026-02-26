import React, { useRef, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const API_BASE = import.meta.env.VITE_API_BASE || ''

export function FaceIdentification({ onIdentification, maxCandidates = 5 }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const wsRef = useRef(null)

  const [isIdentifying, setIsIdentifying] = useState(false)
  const [identificationResult, setIdentificationResult] = useState(null)
  const [qualityMetrics, setQualityMetrics] = useState(null)
  const [error, setError] = useState(null)
  const [isCapturing, setIsCapturing] = useState(false)

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
      }
    } catch (err) {
      setError(`Camera access failed: ${err.message}`)
    }
  }, [])

  // Capture single frame and identify
  const captureAndIdentify = useCallback(async () => {
    if (!videoRef.current) return

    setIsCapturing(true)
    setError(null)

    try {
      const video = videoRef.current
      const canvas = canvasRef.current

      if (canvas && video.videoWidth > 0 && video.videoHeight > 0) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight

        const ctx = canvas.getContext('2d')
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

        const imageData = canvas.toDataURL('image/jpeg', 0.8)

        // Send to identification endpoint
        const response = await fetch(`${API_BASE}/identify`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: imageData,
            max_candidates: maxCandidates
          })
        })

        const result = await response.json()

        if (result.error) {
          setError(result.error)
        } else {
          setIdentificationResult(result)
          setQualityMetrics(result.quality_metrics)

          if (onIdentification) {
            onIdentification(result)
          }
        }
      }
    } catch (err) {
      setError(`Identification failed: ${err.message}`)
    } finally {
      setIsCapturing(false)
    }
  }, [maxCandidates, onIdentification])

  // Start identification mode
  const startIdentification = useCallback(async () => {
    setError(null)
    setIdentificationResult(null)
    await startCamera()
    setIsIdentifying(true)
  }, [startCamera])

  // Stop identification
  const stopIdentification = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    setIsIdentifying(false)
    setIdentificationResult(null)
    setQualityMetrics(null)
  }, [])

  return (
    <div className="w-full max-w-4xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-800 mb-2">
          Face Identification
        </h2>
        <p className="text-gray-600">
          Identify unknown faces against the entire database
        </p>
      </div>

      {/* Camera View */}
      <div className="relative bg-black rounded-lg overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-auto"
          style={{ display: isIdentifying ? 'block' : 'none' }}
        />
        <canvas
          ref={canvasRef}
          className="hidden"
        />

        {!isIdentifying && (
          <div className="flex items-center justify-center h-96 text-white">
            <div className="text-center">
              <div className="text-6xl mb-4">🔍</div>
              <p className="text-xl">Face Identification</p>
              <p className="text-sm opacity-75">Click start to identify faces</p>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex justify-center space-x-4">
        <button
          onClick={startIdentification}
          disabled={isIdentifying}
          className="px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 text-white font-semibold rounded-lg hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isIdentifying ? 'Camera Active' : 'Start Camera'}
        </button>

        <button
          onClick={captureAndIdentify}
          disabled={!isIdentifying || isCapturing}
          className="px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold rounded-lg hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isCapturing ? 'Identifying...' : 'Identify Face'}
        </button>

        <button
          onClick={stopIdentification}
          disabled={!isIdentifying}
          className="px-6 py-3 bg-gradient-to-r from-red-500 to-pink-600 text-white font-semibold rounded-lg hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Stop
        </button>
      </div>

      {/* Error Display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg"
          >
            <div className="flex items-center">
              <span className="text-2xl mr-3">❌</span>
              <div>
                <p className="font-semibold">Error</p>
                <p>{error}</p>
              </div>
            </div>
          </motion.div>
        )}

        {identificationResult && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="space-y-4"
          >
            {/* Best Match */}
            {identificationResult.best_match && (
              <div className={`p-6 rounded-lg border-2 ${
                identificationResult.identified
                  ? 'bg-green-50 border-green-300 text-green-800'
                  : 'bg-yellow-50 border-yellow-300 text-yellow-800'
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    <span className="text-3xl mr-3">
                      {identificationResult.identified ? '✅' : '⚠️'}
                    </span>
                    <div>
                      <h3 className="text-xl font-bold">
                        {identificationResult.identified ? 'IDENTIFIED' : 'POSSIBLE MATCH'}
                      </h3>
                      <p className="text-lg">{identificationResult.best_match.user_id}</p>
                    </div>
                  </div>

                  <div className="text-right">
                    <p className="text-2xl font-bold">
                      {(identificationResult.best_match.confidence * 100).toFixed(1)}%
                    </p>
                    <p className="text-sm">Confidence</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <p className="text-2xl font-bold">{identificationResult.best_match.similarity.toFixed(3)}</p>
                    <p className="text-sm">Similarity</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold">
                      {qualityMetrics?.quality_score?.toFixed(0) || 'N/A'}%
                    </p>
                    <p className="text-sm">Image Quality</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold">{identificationResult.candidates.length}</p>
                    <p className="text-sm">Candidates</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold">{identificationResult.datastore_stats.total_users}</p>
                    <p className="text-sm">Total Users</p>
                  </div>
                </div>
              </div>
            )}

            {/* All Candidates */}
            {identificationResult.candidates && identificationResult.candidates.length > 1 && (
              <div className="bg-gray-50 p-6 rounded-lg">
                <h4 className="text-lg font-semibold mb-4 text-gray-800">All Candidates (Ranked)</h4>
                <div className="space-y-3">
                  {identificationResult.candidates.map((candidate, index) => (
                    <div
                      key={candidate.user_id}
                      className={`flex items-center justify-between p-3 rounded-lg ${
                        index === 0 ? 'bg-blue-100 border border-blue-300' : 'bg-white'
                      }`}
                    >
                      <div className="flex items-center">
                        <span className="text-lg mr-3">#{index + 1}</span>
                        <div>
                          <p className="font-semibold">{candidate.user_id}</p>
                          <p className="text-sm text-gray-600">
                            Similarity: {candidate.similarity.toFixed(3)} |
                            Confidence: {(candidate.confidence * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>

                      <div className={`px-3 py-1 rounded-full text-sm font-semibold ${
                        candidate.verified
                          ? 'bg-green-100 text-green-800'
                          : 'bg-gray-100 text-gray-600'
                      }`}>
                        {candidate.verified ? 'VERIFIED' : 'CANDIDATE'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Quality Metrics */}
            {qualityMetrics && (
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-800 mb-2">Image Quality Analysis</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-blue-600">Brightness</p>
                    <p className="font-semibold">{qualityMetrics.brightness?.toFixed(1)}</p>
                  </div>
                  <div>
                    <p className="text-blue-600">Contrast</p>
                    <p className="font-semibold">{qualityMetrics.contrast?.toFixed(1)}</p>
                  </div>
                  <div>
                    <p className="text-blue-600">Sharpness</p>
                    <p className="font-semibold">{qualityMetrics.sharpness?.toFixed(1)}</p>
                  </div>
                  <div>
                    <p className="text-blue-600">Quality Score</p>
                    <p className="font-semibold">{qualityMetrics.quality_score?.toFixed(1)}%</p>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Instructions */}
      {!isIdentifying && (
        <div className="text-center text-gray-600">
          <p className="mb-2">📋 How to use:</p>
          <ul className="text-sm space-y-1">
            <li>• Start the camera to begin</li>
            <li>• Position the unknown face clearly</li>
            <li>• Click "Identify Face" to search the database</li>
            <li>• Results show the best matches ranked by similarity</li>
          </ul>
        </div>
      )}
    </div>
  )
}

export default FaceIdentification
