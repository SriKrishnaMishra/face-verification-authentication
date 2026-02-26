import React, { useEffect, useMemo, useRef, useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import RealTimeFaceVerification from './RealTimeFaceVerification'
import FaceIdentification from './FaceIdentification'
import SystemDashboard from './SystemDashboard'
import AuthenticationFlow from './AuthenticationFlow'

const API_BASE = import.meta.env.VITE_API_BASE || ''

function useCamera() {
  const videoRef = useRef(null)
  const [streaming, setStreaming] = useState(false)

  const start = async () => {
    if (streaming) return
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
    if (videoRef.current) videoRef.current.srcObject = stream
    setStreaming(true)
  }
  const stop = () => {
    const video = videoRef.current
    if (video && video.srcObject) {
      video.srcObject.getTracks().forEach(t => t.stop())
      video.srcObject = null
    }
    setStreaming(false)
  }
  const grab = () => {
    const video = videoRef.current
    if (!video) return null
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    return canvas.toDataURL('image/jpeg', 0.92)
  }

  return { videoRef, streaming, start, stop, grab }
}

async function postJSON(path, payload) {
  const url = `${API_BASE}${path}`
  const token = localStorage.getItem('fv_token')
  const headers = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload) })
  if (!res.ok) {
    let body = await res.text()
    body = body || `${res.status} ${res.statusText}`
    throw new Error(body)
  }
  return res.json()
}

async function postForm(path, form) {
  const url = `${API_BASE}${path}`
  const token = localStorage.getItem('fv_token')
  const headers = {}
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(url, { method: 'POST', headers, body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

function AnimatedHeader() {
  return (
    <div className="relative overflow-hidden rounded-xl p-6 bg-gradient-to-r from-indigo-600/30 via-purple-600/30 to-pink-600/30">
      <div className="absolute -top-10 -left-10 w-40 h-40 bg-pink-500/30 blur-3xl rounded-full animate-pulse"></div>
      <div className="absolute -bottom-10 -right-10 w-40 h-40 bg-indigo-500/30 blur-3xl rounded-full animate-ping"></div>
      <h1 className="text-3xl font-bold">Real-time Face Verification</h1>
      <p className="opacity-80 mt-1">Webcam capture, registration and live verification using FastAPI + DeepFace</p>
    </div>
  )
}

function Navigation() {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Home', icon: '🏠' },
    { path: '/realtime', label: 'Real-time Verify', icon: '📹' },
    { path: '/identify', label: 'Identify Face', icon: '🔍' },
    { path: '/dashboard', label: 'Dashboard', icon: '📊' }
  ]

  return (
    <nav className="bg-white/10 backdrop-blur-sm rounded-xl p-4 mb-6">
      <div className="flex flex-wrap gap-2">
        {navItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={`px-4 py-2 rounded-lg transition-all flex items-center gap-2 ${
              location.pathname === item.path
                ? 'bg-white/20 text-white shadow-lg'
                : 'text-white/80 hover:bg-white/10 hover:text-white'
            }`}
          >
            <span>{item.icon}</span>
            <span className="hidden sm:inline">{item.label}</span>
          </Link>
        ))}
      </div>
    </nav>
  )
}

function HomePage() {
  const { videoRef, streaming, start, stop, grab } = useCamera()
  const [userId, setUserId] = useState('alice')
  const [authUser, setAuthUser] = useState(null)
  const [authStatus, setAuthStatus] = useState('')
  const [verifying, setVerifying] = useState(false)
  const [status, setStatus] = useState('Idle')
  const [score, setScore] = useState('-')

  useEffect(() => {
    let timer
    if (verifying) {
      timer = setInterval(async () => {
        try {
          const image = grab()
          if (!image) return
          setStatus('Verifying...')
          const data = await postJSON('/verify', { user_id: userId, image })
          setStatus(data.verified ? 'Verified ✅' : 'Not Verified ❌')
          setScore(`${data.score} (thr=${data.threshold})`)
        } catch (e) {
          setStatus('Verify error')
        }
      }, 1200)
    }
    return () => timer && clearInterval(timer)
  }, [verifying, userId])

  const onRegister = async () => {
    try {
      const image = grab()
      if (!image) return
      setStatus('Registering...')
      const data = await postJSON('/register', { user_id: userId, image })
      setStatus(`Registered: ${data.user_id}`)
    } catch (e) {
      setStatus('Register error')
    }
  }

  // Auth handlers
  const onAuthRegister = async (username, password) => {
    try {
      setAuthStatus('Creating...')
      const res = await postJSON('/auth/register', { username, password })
      setAuthStatus(`Created ${res.username}`)
    } catch (e) {
      setAuthStatus('Create error')
    }
  }

  const onLogin = async (username, password) => {
    try {
      setAuthStatus('Signing in...')
      const form = new URLSearchParams()
      form.append('username', username)
      form.append('password', password)
      form.append('grant_type', '')
      const res = await fetch(`${API_BASE}/auth/token`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      localStorage.setItem('fv_token', data.access_token)
      setAuthUser(username)
      setUserId(username)
      setAuthStatus('Signed in')
    } catch (e) {
      setAuthStatus('Sign in error')
    }
  }

  const onLogout = () => {
    localStorage.removeItem('fv_token')
    setAuthUser(null)
    setAuthStatus('')
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <AnimatedHeader />

      <div className="grid md:grid-cols-2 gap-6">
        <div className="card space-y-4">
          <div className="flex items-center gap-3">
            <input
              className="w-full px-3 py-2 rounded-lg bg-white/5 outline-none border border-white/10"
              placeholder="Enter User ID"
              value={userId}
              onChange={e => setUserId(e.target.value)}
            />
            <div className="flex flex-col gap-2">
              {authUser ? (
                <div className="flex items-center gap-2">
                  <div className="mono">{authUser}</div>
                  <button className="btn" onClick={onLogout}>Logout</button>
                </div>
              ) : (
                <AuthInline onRegister={onAuthRegister} onLogin={onLogin} authStatus={authStatus} />
              )}
            </div>
            {!streaming ? (
              <button className="btn" onClick={start}>Start Camera</button>
            ) : (
              <button className="btn bg-red-600" onClick={stop}>Stop</button>
            )}
          </div>

          <div className="relative rounded-xl overflow-hidden">
            <video ref={videoRef} autoPlay playsInline className="w-full aspect-video bg-black/50" />
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute inset-6 rounded-xl border border-white/20"></div>
            </div>
          </div>

          <div className="flex gap-3">
            <button className="btn" onClick={onRegister} disabled={!streaming}>Register Face</button>
            {!verifying ? (
              <button className="btn" onClick={() => setVerifying(true)} disabled={!streaming}>Start Verify</button>
            ) : (
              <button className="btn bg-amber-600" onClick={() => setVerifying(false)}>Stop Verify</button>
            )}
          </div>
        </div>

        <div className="card space-y-4">
          <div className="title">Status</div>
          <div className="mono">{status}</div>
          <div className="title">Score</div>
          <div className="mono text-lg">{score}</div>

          <div className="title">System</div>
          <HealthPanel />
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  // Check authentication on app load
  useEffect(() => {
    const token = localStorage.getItem('fv_token')
    const username = localStorage.getItem('fv_username')
    
    if (token && username) {
      // Verify token is still valid
      fetch(`${API_BASE}/auth/me`, {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      .then(res => {
        if (res.ok) {
          setIsAuthenticated(true)
          setUser({ username })
        } else {
          // Token invalid, clear it
          localStorage.removeItem('fv_token')
          localStorage.removeItem('fv_username')
        }
      })
      .catch(() => {
        localStorage.removeItem('fv_token')
        localStorage.removeItem('fv_username')
      })
      .finally(() => setLoading(false))
    } else {
      setLoading(false)
    }
  }, [])

  const handleLogin = () => {
    setIsAuthenticated(true)
    const username = localStorage.getItem('fv_username')
    setUser({ username })
  }

  const handleLogout = () => {
    localStorage.removeItem('fv_token')
    localStorage.removeItem('fv_username')
    setIsAuthenticated(false)
    setUser(null)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-600 to-indigo-800 flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return <AuthenticationFlow onLogin={handleLogin} />
  }

  return (
    <Router>
      <div className="min-h-screen p-6 bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
        <header className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">🔐 Face Verification System</h1>
          <div className="flex items-center gap-4">
            <span>Welcome, {user?.username}</span>
            <button 
              onClick={handleLogout}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
            >
              Logout
            </button>
          </div>
        </header>

        <Navigation />

        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/realtime" element={<RealTimeFaceVerification />} />
            <Route path="/identify" element={<FaceIdentification />} />
            <Route path="/dashboard" element={<SystemDashboard />} />
          </Routes>
        </AnimatePresence>

        <footer className="opacity-60 text-sm text-center mt-12">
          MVP UI • React + Tailwind + Vite • Advanced Face Verification System
        </footer>
      </div>
    </Router>
  )
}


function AuthInline({ onRegister, onLogin, authStatus }) {
  const [u, setU] = useState('')
  const [p, setP] = useState('')
  return (
    <div className="flex items-center gap-2">
      <input className="px-2 py-1 rounded" placeholder="user" value={u} onChange={e => setU(e.target.value)} />
      <input className="px-2 py-1 rounded" placeholder="pass" type="password" value={p} onChange={e => setP(e.target.value)} />
      <button className="btn" onClick={() => onLogin(u, p)}>Login</button>
      <button className="btn" onClick={() => onRegister(u, p)}>Create</button>
      <div className="mono">{authStatus}</div>
    </div>
  )
}

function HealthPanel() {
  const [health, setHealth] = useState(null)
  useEffect(() => {
    fetch(`${API_BASE}/health`).then(r => r.json()).then(setHealth).catch(() => setHealth(null))
  }, [])
  if (!health) return <div className="mono">Loading...</div>
  return (
    <div className="flex flex-wrap gap-2">
      <span className="badge">Status: {health.status}</span>
      <span className="badge">Model: {health.model}</span>
      <span className="badge">Detector: {health.detector}</span>
    </div>
  )
}
