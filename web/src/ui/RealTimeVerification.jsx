import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const API_BASE = import.meta.env.VITE_API_BASE || ''

export function RealTimeVerification({ userId }) {
  const videoRef = useRef(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [result, setResult] = useState(null)
  const [isVerifying, setIsVerifying] = useState(false)
  const [quality, setQuality] = useState(null)
  const [stats, setStats] = useState(null)
  const [history, setHistory] = useState([])

  // Start camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsStreaming(true)
      }
    } catch (err) {
      console.error('Camera access denied:', err)
      alert('Unable to access camera')
    }
  }

  // Stop camera
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
      setIsStreaming(false)
    }
  }

  // Capture frame and verify
  const verifyFrame = async () => {
    if (!videoRef.current || !isStreaming) return

    setIsVerifying(true)

    try {
      const canvas = document.createElement('canvas')
      canvas.width = videoRef.current.videoWidth || 640
      canvas.height = videoRef.current.videoHeight || 480
      const ctx = canvas.getContext('2d')
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
      const imageData = canvas.toDataURL('image/jpeg', 0.95)

      const res = await fetch(`${API_BASE}/verify-realtime`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          image: imageData,
        }),
      })

      if (!res.ok) {
        const error = await res.text()
        throw new Error(error)
      }

      const data = await res.json()
      setResult(data)
      setQuality(data.quality_metrics)

      // Add to history
      setHistory((prev) => [
        {
          timestamp: new Date().toLocaleTimeString(),
          verified: data.verified,
          confidence: data.confidence,
          similarity: data.similarity,
        },
        ...prev.slice(0, 9), // Keep last 10
      ])
    } catch (err) {
      console.error('Verification error:', err)
      alert(`Error: ${err.message}`)
    } finally {
      setIsVerifying(false)
    }
  }

  // Fetch stats
  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/performance-stats`)
      if (res.ok) {
        const data = await res.json()
        setStats(data)
      }
    } catch (err) {
      console.error('Failed to fetch stats:', err)
    }
  }

  useEffect(() => {
    const interval = setInterval(fetchStats, 5000) // Update every 5s
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="w-full max-w-4xl mx-auto p-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg shadow-lg">
      <h2 className="text-3xl font-bold mb-6 text-center text-gray-800">
        🎥 Real-time Face Verification
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video Feed */}
        <div className="lg:col-span-2">
          <div className="bg-black rounded-lg overflow-hidden shadow-lg">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full aspect-video"
            />
          </div>

          <div className="mt-4 flex gap-2 justify-center">
            {!isStreaming ? (
              <button
                onClick={startCamera}
                className="px-6 py-2 bg-green-500 text-white font-semibold rounded-lg hover:bg-green-600"
              >
                Start Camera
              </button>
            ) : (
              <>
                <button
                  onClick={stopCamera}
                  className="px-6 py-2 bg-red-500 text-white font-semibold rounded-lg hover:bg-red-600"
                >
                  Stop Camera
                </button>
                <button
                  onClick={verifyFrame}
                  disabled={isVerifying}
                  className="px-6 py-2 bg-blue-500 text-white font-semibold rounded-lg hover:bg-blue-600 disabled:opacity-50"
                >
                  {isVerifying ? 'Verifying...' : 'Verify Face'}
                </button>
              </>
            )}
          </div>

          {/* Quality Metrics */}
          <AnimatePresence>
            {quality && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="mt-4 p-4 bg-white rounded-lg shadow-md"
              >
                <h4 className="font-semibold mb-3 text-gray-800">Image Quality</h4>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <div className="text-gray-600">Quality Score</div>
                    <div className="font-bold text-lg text-blue-600">
                      {quality.quality_score?.toFixed(0)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-600">Blur Status</div>
                    <div
                      className={`font-bold ${
                        quality.is_blurry ? 'text-red-600' : 'text-green-600'
                      }`}
                    >
                      {quality.is_blurry ? '❌ Blurry' : '✓ Sharp'}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-600">Brightness</div>
                    <div className="font-bold">{quality.brightness?.toFixed(0)}</div>
                  </div>
                  <div>
                    <div className="text-gray-600">Contrast</div>
                    <div className="font-bold">{quality.contrast?.toFixed(0)}</div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Results Panel */}
        <div className="space-y-4">
          {/* Main Result */}
          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className={`p-6 rounded-lg shadow-lg ${
                  result.verified
                    ? 'bg-gradient-to-br from-green-100 to-emerald-100 border-2 border-green-500'
                    : 'bg-gradient-to-br from-red-100 to-pink-100 border-2 border-red-500'
                }`}
              >
                <div className="text-center">
                  <div className="text-5xl mb-2">
                    {result.verified ? '✅' : '❌'}
                  </div>
                  <h3 className="text-2xl font-bold text-gray-800 mb-3">
                    {result.verified ? 'VERIFIED' : 'NOT VERIFIED'}
                  </h3>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-700">Similarity:</span>
                    <span className="font-bold">{result.similarity.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-700">Confidence:</span>
                    <span className="font-bold">
                      {(result.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-700">Threshold:</span>
                    <span className="font-bold">{result.threshold.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-700">Margin:</span>
                    <span className="font-bold">{result.margin.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-700">Time:</span>
                    <span className="font-bold">{result.processing_time_ms.toFixed(0)}ms</span>
                  </div>
                </div>

                {/* Confidence Bar */}
                <div className="mt-4">
                  <div className="text-xs text-gray-600 mb-1">Confidence Level</div>
                  <div className="w-full bg-gray-300 rounded-full h-3 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${result.confidence * 100}%` }}
                      className={`h-full ${
                        result.verified
                          ? 'bg-gradient-to-r from-green-400 to-emerald-500'
                          : 'bg-gradient-to-r from-red-400 to-pink-500'
                      }`}
                    />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Performance Stats */}
          {stats && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="p-4 bg-white rounded-lg shadow-md"
            >
              <h4 className="font-semibold mb-2 text-gray-800">System Stats</h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span>Verifications:</span>
                  <span className="font-bold">{stats.total_verifications}</span>
                </div>
                <div className="flex justify-between">
                  <span>Success Rate:</span>
                  <span className="font-bold">
                    {(stats.success_rate * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Avg Time:</span>
                  <span className="font-bold">
                    {stats.average_processing_time_ms.toFixed(0)}ms
                  </span>
                </div>
                {stats.cache_stats && (
                  <div className="flex justify-between">
                    <span>Cache Hit Rate:</span>
                    <span className="font-bold">
                      {(stats.cache_stats.hit_rate * 100).toFixed(0)}%
                    </span>
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {/* Recent History */}
          {history.length > 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="p-4 bg-white rounded-lg shadow-md"
            >
              <h4 className="font-semibold mb-2 text-gray-800">Recent</h4>
              <div className="space-y-1 max-h-40 overflow-y-auto text-xs">
                {history.map((item, idx) => (
                  <div
                    key={idx}
                    className="flex justify-between items-center p-1 rounded hover:bg-gray-100"
                  >
                    <span className="text-gray-600">{item.timestamp}</span>
                    <div className="flex items-center gap-2">
                      <span className={item.verified ? 'text-green-600' : 'text-red-600'}>
                        {item.verified ? '✓' : '✗'}
                      </span>
                      <span className="font-bold">
                        {(item.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}

export default RealTimeVerification
