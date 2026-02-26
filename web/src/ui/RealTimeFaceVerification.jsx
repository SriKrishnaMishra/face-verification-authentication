import React, { useEffect, useRef, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const API_BASE = import.meta.env.VITE_API_BASE || ''

export function RealTimeFaceVerification({ userId, onVerification, onError }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const overlayCanvasRef = useRef(null)
  const streamRef = useRef(null)
  const wsRef = useRef(null)
  const intervalRef = useRef(null)
  const animationFrameRef = useRef(null)

  const [isStreaming, setIsStreaming] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [verificationResult, setVerificationResult] = useState(null)
  const [qualityMetrics, setQualityMetrics] = useState(null)
  const [processingTime, setProcessingTime] = useState(0)
  const [error, setError] = useState(null)
  const [frameCount, setFrameCount] = useState(0)
  const [faceDetected, setFaceDetected] = useState(false)
  const [faceBounds, setFaceBounds] = useState(null)
  const [systemStats, setSystemStats] = useState(null)
  const [mlFeatures, setMlFeatures] = useState({
    model: 'Loading...',
    detector: 'Loading...',
    vectorDb: 'Loading...',
    confidence: 0
  })

  // Load system information
  const loadSystemInfo = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/health`)
      const data = await response.json()
      setSystemStats(data)
      setMlFeatures({
        model: data.model || 'Unknown',
        detector: data.detector || 'Unknown',
        vectorDb: data.faiss_available ? 'FAISS (Optimized)' : 'Linear Search',
        confidence: 0
      })
    } catch (err) {
      console.error('Failed to load system info:', err)
    }
  }, [])

  // Start verification session
  const startSession = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/stream/start-session${userId ? `?user_id=${userId}` : ''}`)
      const session = await response.json()
      setSessionId(session.session_id)
      return session.session_id
    } catch (err) {
      setError(`Failed to start session: ${err.message}`)
      if (onError) onError(err)
      return null
    }
  }, [userId, onError])

  // Start WebSocket connection
  const startWebSocket = useCallback((sessionId) => {
    const wsUrl = `ws://localhost:8000/ws/verify/${sessionId}`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket connected for real-time verification')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.type === 'verification_result') {
        setVerificationResult(data)
        setQualityMetrics(data.quality_metrics)
        setProcessingTime(data.processing_time_ms || 0)
        setFrameCount(prev => prev + 1)

        // Update ML features with real-time data
        setMlFeatures(prev => ({
          ...prev,
          confidence: data.confidence || 0
        }))

        // Face detection visualization
        if (data.face_bounds) {
          setFaceBounds(data.face_bounds)
          setFaceDetected(true)
        } else {
          setFaceDetected(false)
        }

        if (onVerification) {
          onVerification(data)
        }
      } else if (data.type === 'face_detected') {
        setFaceDetected(true)
        setFaceBounds(data.bounds)
      } else if (data.type === 'no_face') {
        setFaceDetected(false)
        setFaceBounds(null)
      }
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setError('Connection lost - please restart verification')
    }

    return ws
  }, [onVerification])

  // Draw face detection overlay
  const drawFaceOverlay = useCallback(() => {
    const video = videoRef.current
    const overlayCanvas = overlayCanvasRef.current

    if (!video || !overlayCanvas || !faceDetected || !faceBounds) return

    const ctx = overlayCanvas.getContext('2d')
    const rect = video.getBoundingClientRect()

    overlayCanvas.width = rect.width
    overlayCanvas.height = rect.height

    // Clear previous drawings
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height)

    // Draw face bounding box
    const scaleX = rect.width / video.videoWidth
    const scaleY = rect.height / video.videoHeight

    ctx.strokeStyle = verificationResult?.verified ? '#10B981' : '#EF4444'
    ctx.lineWidth = 3
    ctx.strokeRect(
      faceBounds.x * scaleX,
      faceBounds.y * scaleY,
      faceBounds.width * scaleX,
      faceBounds.height * scaleY
    )

    // Draw face label
    ctx.fillStyle = verificationResult?.verified ? '#10B981' : '#EF4444'
    ctx.font = '16px Arial'
    ctx.fillText(
      verificationResult?.verified ? '✓ VERIFIED' : '✗ NOT VERIFIED',
      faceBounds.x * scaleX,
      faceBounds.y * scaleY - 10
    )
  }, [faceDetected, faceBounds, verificationResult])

  // Animation loop for overlay
  useEffect(() => {
    const animate = () => {
      drawFaceOverlay()
      animationFrameRef.current = requestAnimationFrame(animate)
    }

    if (isStreaming) {
      animate()
    } else {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      // Clear overlay when not streaming
      const overlayCanvas = overlayCanvasRef.current
      if (overlayCanvas) {
        const ctx = overlayCanvas.getContext('2d')
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height)
      }
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [isStreaming, drawFaceOverlay])

  // Start camera stream
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280, min: 640 },
          height: { ideal: 720, min: 480 },
          frameRate: { ideal: 30, min: 15 },
          facingMode: 'user'
        },
        audio: false
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setIsStreaming(true)
      }
    } catch (err) {
      setError(`Camera access failed: ${err.message}`)
      if (onError) onError(err)
    }
  }, [onError])

  // Capture and send frame
  const captureAndSendFrame = useCallback(() => {
    if (!videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return
    }

    const video = videoRef.current
    const canvas = canvasRef.current

    if (canvas && video.videoWidth > 0 && video.videoHeight > 0) {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      const ctx = canvas.getContext('2d')
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      const imageData = canvas.toDataURL('image/jpeg', 0.8)

      // Send frame via WebSocket
      wsRef.current.send(JSON.stringify({
        type: 'frame',
        image: imageData,
        timestamp: Date.now()
      }))
    }
  }, [])

  // Start real-time verification
  const startVerification = useCallback(async () => {
    setError(null)
    setVerificationResult(null)
    setFrameCount(0)
    setFaceDetected(false)
    setFaceBounds(null)

    // Load system info
    await loadSystemInfo()

    // Start session
    const session = await startSession()
    if (!session) return

    // Start WebSocket
    const ws = startWebSocket(session)

    // Start camera
    await startCamera()

    // Start frame capture interval (15 FPS for better performance)
    intervalRef.current = setInterval(captureAndSendFrame, 66)
  }, [startSession, startWebSocket, startCamera, captureAndSendFrame, loadSystemInfo])

  // Stop verification
  const stopVerification = useCallback(async () => {
    // Clear interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    // Stop WebSocket
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: 'end_session' }))
      wsRef.current.close()
      wsRef.current = null
    }

    // Stop camera
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    // End session
    if (sessionId) {
      try {
        await fetch(`${API_BASE}/stream/end-session/${sessionId}`, { method: 'DELETE' })
      } catch (err) {
        console.error('Error ending session:', err)
      }
    }

    setIsStreaming(false)
    setSessionId(null)
    setVerificationResult(null)
    setQualityMetrics(null)
    setFaceDetected(false)
    setFaceBounds(null)
  }, [sessionId])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopVerification()
    }
  }, [stopVerification])

  return (
    <div className="w-full max-w-6xl mx-auto p-6 space-y-6">
      {/* Enhanced Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 rounded-2xl p-8 text-white"
      >
        <div className="flex items-center justify-center mb-4">
          <motion.div
            animate={{ rotate: isStreaming ? 360 : 0 }}
            transition={{ duration: 2, repeat: isStreaming ? Infinity : 0, ease: "linear" }}
            className="text-5xl mr-4"
          >
            {isStreaming ? '🔄' : '📹'}
          </motion.div>
          <div>
            <h1 className="text-4xl font-bold mb-2">
              Advanced Real-Time Face Verification
            </h1>
            <p className="text-xl opacity-90">
              AI-Powered Authentication with Live Processing
            </p>
          </div>
        </div>

        {userId && (
          <div className="bg-white/10 rounded-lg p-4 inline-block">
            <p className="text-lg">Verifying: <span className="font-semibold">{userId}</span></p>
          </div>
        )}
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Video Stream */}
        <div className="lg:col-span-2">
          <div className="relative bg-gradient-to-br from-gray-900 to-black rounded-2xl overflow-hidden shadow-2xl">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-auto rounded-2xl"
              style={{ display: isStreaming ? 'block' : 'none' }}
            />

            {/* Face Detection Overlay */}
            <canvas
              ref={overlayCanvasRef}
              className="absolute top-0 left-0 w-full h-full pointer-events-none"
              style={{ display: isStreaming ? 'block' : 'none' }}
            />

            <canvas
              ref={canvasRef}
              className="hidden"
            />

            {!isStreaming && (
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="flex items-center justify-center h-96 text-white"
              >
                <div className="text-center">
                  <motion.div
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                    className="text-8xl mb-6"
                  >
                    🎯
                  </motion.div>
                  <h3 className="text-2xl font-bold mb-2">Ready for Verification</h3>
                  <p className="text-lg opacity-75 mb-4">Advanced AI Face Recognition System</p>
                  <div className="flex justify-center space-x-4 text-sm">
                    <span className="bg-blue-500/20 px-3 py-1 rounded-full">HD Camera</span>
                    <span className="bg-green-500/20 px-3 py-1 rounded-full">Real-time</span>
                    <span className="bg-purple-500/20 px-3 py-1 rounded-full">AI Powered</span>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Live Status Indicator */}
            {isStreaming && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="absolute top-4 left-4 flex items-center space-x-2"
              >
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                  className="w-3 h-3 bg-red-500 rounded-full"
                />
                <span className="text-white font-semibold text-sm">LIVE</span>
              </motion.div>
            )}
          </div>

          {/* Enhanced Controls */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex justify-center space-x-4 mt-6"
          >
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={startVerification}
              disabled={isStreaming}
              className="px-8 py-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-bold rounded-xl hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              <span>🚀</span>
              <span>{isStreaming ? 'Verifying...' : 'Start AI Verification'}</span>
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={stopVerification}
              disabled={!isStreaming}
              className="px-8 py-4 bg-gradient-to-r from-red-500 to-pink-600 text-white font-bold rounded-xl hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              <span>⏹️</span>
              <span>Stop</span>
            </motion.button>
          </motion.div>
        </div>

        {/* Advanced Metrics Panel */}
        <div className="space-y-4">
          {/* ML Features Card */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl p-6 border border-indigo-200"
          >
            <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center">
              <span className="text-2xl mr-2">🤖</span>
              AI/ML Engine
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Model:</span>
                <span className="font-semibold text-indigo-600">{mlFeatures.model}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Detector:</span>
                <span className="font-semibold text-indigo-600">{mlFeatures.detector}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Vector DB:</span>
                <span className="font-semibold text-indigo-600">{mlFeatures.vectorDb}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Confidence:</span>
                <span className={`font-bold ${mlFeatures.confidence > 0.8 ? 'text-green-600' : mlFeatures.confidence > 0.5 ? 'text-yellow-600' : 'text-red-600'}`}>
                  {(mlFeatures.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </motion.div>

          {/* Real-time Stats Card */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border border-green-200"
          >
            <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center">
              <span className="text-2xl mr-2">📊</span>
              Real-time Stats
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <motion.p
                  key={processingTime}
                  initial={{ scale: 1.2 }}
                  animate={{ scale: 1 }}
                  className="text-2xl font-bold text-green-600"
                >
                  {processingTime.toFixed(0)}ms
                </motion.p>
                <p className="text-xs text-gray-600">Processing</p>
              </div>
              <div className="text-center">
                <motion.p
                  key={frameCount}
                  initial={{ scale: 1.2 }}
                  animate={{ scale: 1 }}
                  className="text-2xl font-bold text-green-600"
                >
                  {frameCount}
                </motion.p>
                <p className="text-xs text-gray-600">Frames</p>
              </div>
              <div className="text-center">
                <p className={`text-2xl font-bold ${faceDetected ? 'text-green-600' : 'text-red-600'}`}>
                  {faceDetected ? '✓' : '✗'}
                </p>
                <p className="text-xs text-gray-600">Face</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-green-600">
                  {isStreaming ? 'ON' : 'OFF'}
                </p>
                <p className="text-xs text-gray-600">Stream</p>
              </div>
            </div>
          </motion.div>

          {/* System Health Card */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl p-6 border border-blue-200"
          >
            <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center">
              <span className="text-2xl mr-2">💻</span>
              System Health
            </h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Status:</span>
                <span className="flex items-center">
                  <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                  <span className="text-sm font-semibold text-green-600">Online</span>
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">API:</span>
                <span className="text-sm font-semibold text-blue-600">FastAPI</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">WebSocket:</span>
                <span className={`text-sm font-semibold ${wsRef.current?.readyState === 1 ? 'text-green-600' : 'text-red-600'}`}>
                  {wsRef.current?.readyState === 1 ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Enhanced Status Display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            className="p-6 bg-gradient-to-r from-red-50 to-pink-50 border-2 border-red-300 rounded-xl shadow-lg"
          >
            <div className="flex items-center">
              <motion.span
                animate={{ rotate: [0, 10, -10, 0] }}
                transition={{ duration: 0.5, repeat: Infinity }}
                className="text-4xl mr-4"
              >
                ❌
              </motion.span>
              <div>
                <h3 className="text-xl font-bold text-red-800">Verification Error</h3>
                <p className="text-red-700 mt-1">{error}</p>
              </div>
            </div>
          </motion.div>
        )}

        {verificationResult && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className={`p-8 rounded-2xl border-4 shadow-2xl ${
              verificationResult.verified
                ? 'bg-gradient-to-r from-green-50 to-emerald-50 border-green-400'
                : 'bg-gradient-to-r from-red-50 to-pink-50 border-red-400'
            }`}
          >
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center">
                <motion.span
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 0.5 }}
                  className="text-5xl mr-4"
                >
                  {verificationResult.verified ? '✅' : '❌'}
                </motion.span>
                <div>
                  <h2 className={`text-3xl font-bold ${
                    verificationResult.verified ? 'text-green-800' : 'text-red-800'
                  }`}>
                    {verificationResult.verified ? 'IDENTITY VERIFIED' : 'VERIFICATION FAILED'}
                  </h2>
                  <p className={`text-lg ${
                    verificationResult.verified ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {verificationResult.user_id || 'Unknown user'}
                  </p>
                </div>
              </div>

              <div className="text-right">
                <motion.p
                  key={verificationResult.confidence}
                  initial={{ scale: 1.5 }}
                  animate={{ scale: 1 }}
                  className={`text-5xl font-bold ${
                    verificationResult.confidence > 0.8 ? 'text-green-600' :
                    verificationResult.confidence > 0.5 ? 'text-yellow-600' : 'text-red-600'
                  }`}
                >
                  {verificationResult.confidence ? `${(verificationResult.confidence * 100).toFixed(1)}%` : 'N/A'}
                </motion.p>
                <p className="text-sm text-gray-600">AI Confidence</p>
              </div>
            </div>

            {/* Advanced Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-6 mb-6">
              <div className="text-center bg-white/50 rounded-lg p-4">
                <p className="text-2xl font-bold text-blue-600">
                  {verificationResult.similarity?.toFixed(3) || 'N/A'}
                </p>
                <p className="text-sm text-gray-600">Similarity Score</p>
              </div>
              <div className="text-center bg-white/50 rounded-lg p-4">
                <p className="text-2xl font-bold text-green-600">{processingTime.toFixed(0)}ms</p>
                <p className="text-sm text-gray-600">Processing Time</p>
              </div>
              <div className="text-center bg-white/50 rounded-lg p-4">
                <p className="text-2xl font-bold text-purple-600">{frameCount}</p>
                <p className="text-sm text-gray-600">Frames Processed</p>
              </div>
              <div className="text-center bg-white/50 rounded-lg p-4">
                <p className="text-2xl font-bold text-indigo-600">
                  {qualityMetrics?.quality_score?.toFixed(0) || 'N/A'}%
                </p>
                <p className="text-sm text-gray-600">Image Quality</p>
              </div>
              <div className="text-center bg-white/50 rounded-lg p-4">
                <p className={`text-2xl font-bold ${faceDetected ? 'text-green-600' : 'text-red-600'}`}>
                  {faceDetected ? '✓' : '✗'}
                </p>
                <p className="text-sm text-gray-600">Face Detected</p>
              </div>
            </div>

            {/* Quality Details */}
            {qualityMetrics && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-white/30 rounded-lg p-4 mb-4"
              >
                <h4 className="font-semibold text-gray-800 mb-2">Image Analysis:</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>Brightness: <span className="font-semibold">{qualityMetrics.brightness?.toFixed(1)}</span></div>
                  <div>Contrast: <span className="font-semibold">{qualityMetrics.contrast?.toFixed(1)}</span></div>
                  <div>Sharpness: <span className="font-semibold">{qualityMetrics.sharpness?.toFixed(1)}</span></div>
                  <div>Faces: <span className="font-semibold">{qualityMetrics.face_count || 0}</span></div>
                </div>
              </motion.div>
            )}

            {/* Vector Database Info */}
            <div className="bg-gradient-to-r from-indigo-100 to-purple-100 rounded-lg p-4">
              <h4 className="font-semibold text-gray-800 mb-2 flex items-center">
                <span className="text-lg mr-2">🗄️</span>
                Vector Database Status
              </h4>
              <p className="text-sm text-gray-700">
                Using <span className="font-semibold text-indigo-600">{mlFeatures.vectorDb}</span> for fast similarity search.
                Embeddings stored locally with advanced indexing for real-time matching.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Enhanced Instructions */}
      {!isStreaming && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-8 border border-blue-200"
        >
          <h3 className="text-xl font-bold text-gray-800 mb-4">🚀 How It Works</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-left">
            <div className="text-center">
              <div className="text-4xl mb-2">📹</div>
              <h4 className="font-semibold text-gray-800 mb-1">1. Camera Capture</h4>
              <p className="text-sm text-gray-600">HD video stream with WebRTC for real-time processing</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-2">🤖</div>
              <h4 className="font-semibold text-gray-800 mb-1">2. AI Processing</h4>
              <p className="text-sm text-gray-600">DeepFace neural networks analyze facial features</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-2">✅</div>
              <h4 className="font-semibold text-gray-800 mb-1">3. Instant Verification</h4>
              <p className="text-sm text-gray-600">Vector similarity matching against stored embeddings</p>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

export default RealTimeFaceVerification
        setProcessingTime(data.processing_time_ms)
        setFrameCount(prev => prev + 1)

        if (onVerification) {
          onVerification(data)
        }
      } else if (data.type === 'error') {
        setError(data.message)
        if (onError) onError(new Error(data.message))
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setError('WebSocket connection failed')
      if (onError) onError(error)
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
    }

    return ws
  }, [onVerification, onError])

  // Start camera stream
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 10, max: 15 }
        },
        audio: false
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setIsStreaming(true)
      }
    } catch (err) {
      setError(`Camera access failed: ${err.message}`)
      if (onError) onError(err)
    }
  }, [onError])

  // Capture and send frame
  const captureAndSendFrame = useCallback(() => {
    if (!videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return
    }

    const video = videoRef.current
    const canvas = canvasRef.current

    if (canvas && video.videoWidth > 0 && video.videoHeight > 0) {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      const ctx = canvas.getContext('2d')
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      const imageData = canvas.toDataURL('image/jpeg', 0.8)

      // Send frame via WebSocket
      wsRef.current.send(JSON.stringify({
        type: 'frame',
        image: imageData,
        timestamp: Date.now()
      }))
    }
  }, [])

  // Start real-time verification
  const startVerification = useCallback(async () => {
    setError(null)
    setVerificationResult(null)
    setFrameCount(0)

    // Start session
    const session = await startSession()
    if (!session) return

    // Start WebSocket
    const ws = startWebSocket(session)

    // Start camera
    await startCamera()

    // Start frame capture interval (10 FPS)
    intervalRef.current = setInterval(captureAndSendFrame, 100)
  }, [startSession, startWebSocket, startCamera, captureAndSendFrame])

  // Stop verification
  const stopVerification = useCallback(async () => {
    // Clear interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    // Stop WebSocket
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: 'end_session' }))
      wsRef.current.close()
      wsRef.current = null
    }

    // Stop camera
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    // End session
    if (sessionId) {
      try {
        await fetch(`${API_BASE}/stream/end-session/${sessionId}`, { method: 'DELETE' })
      } catch (err) {
        console.error('Error ending session:', err)
      }
    }

    setIsStreaming(false)
    setSessionId(null)
    setVerificationResult(null)
    setQualityMetrics(null)
  }, [sessionId])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopVerification()
    }
  }, [stopVerification])

  return (
    <div className="w-full max-w-4xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-800 mb-2">
          Real-time Face Verification
        </h2>
        <p className="text-gray-600">
          {userId ? `Verifying against user: ${userId}` : 'Real-time face verification'}
        </p>
      </div>

      {/* Video Stream */}
      <div className="relative bg-black rounded-lg overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-auto"
          style={{ display: isStreaming ? 'block' : 'none' }}
        />
        <canvas
          ref={canvasRef}
          className="hidden"
        />

        {!isStreaming && (
          <div className="flex items-center justify-center h-96 text-white">
            <div className="text-center">
              <div className="text-6xl mb-4">📹</div>
              <p className="text-xl">Camera not active</p>
              <p className="text-sm opacity-75">Click start to begin verification</p>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex justify-center space-x-4">
        <button
          onClick={startVerification}
          disabled={isStreaming}
          className="px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold rounded-lg hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isStreaming ? 'Verifying...' : 'Start Verification'}
        </button>

        <button
          onClick={stopVerification}
          disabled={!isStreaming}
          className="px-6 py-3 bg-gradient-to-r from-red-500 to-pink-600 text-white font-semibold rounded-lg hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Stop
        </button>
      </div>

      {/* Status Display */}
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

        {verificationResult && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className={`p-6 rounded-lg border-2 ${
              verificationResult.verified
                ? 'bg-green-50 border-green-300 text-green-800'
                : 'bg-red-50 border-red-300 text-red-800'
            }`}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <span className="text-3xl mr-3">
                  {verificationResult.verified ? '✅' : '❌'}
                </span>
                <div>
                  <h3 className="text-xl font-bold">
                    {verificationResult.verified ? 'VERIFIED' : 'NOT VERIFIED'}
                  </h3>
                  <p className="text-sm opacity-75">
                    {verificationResult.user_id || 'Unknown user'}
                  </p>
                </div>
              </div>

              <div className="text-right">
                <p className="text-2xl font-bold">
                  {verificationResult.confidence ? `${(verificationResult.confidence * 100).toFixed(1)}%` : 'N/A'}
                </p>
                <p className="text-sm">Confidence</p>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div className="text-center">
                <p className="text-2xl font-bold">{verificationResult.similarity?.toFixed(3) || 'N/A'}</p>
                <p className="text-sm">Similarity</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold">{processingTime.toFixed(0)}ms</p>
                <p className="text-sm">Processing</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold">{frameCount}</p>
                <p className="text-sm">Frames</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold">
                  {qualityMetrics?.quality_score?.toFixed(0) || 'N/A'}%
                </p>
                <p className="text-sm">Quality</p>
              </div>
            </div>

            {/* Quality Details */}
            {qualityMetrics && (
              <div className="text-sm space-y-1 opacity-75">
                <p>Brightness: {qualityMetrics.brightness?.toFixed(1)} | Contrast: {qualityMetrics.contrast?.toFixed(1)}</p>
                <p>Sharpness: {qualityMetrics.sharpness?.toFixed(1)} | Faces: {qualityMetrics.face_count || 0}</p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Instructions */}
      {!isStreaming && (
        <div className="text-center text-gray-600">
          <p className="mb-2">📋 Instructions:</p>
          <ul className="text-sm space-y-1">
            <li>• Click "Start Verification" to begin</li>
            <li>• Position your face clearly in the camera</li>
            <li>• Ensure good lighting for best results</li>
            <li>• The system will verify in real-time</li>
          </ul>
        </div>
      )}
    </div>
  )
}

export default RealTimeFaceVerification
