import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { TwoFactorSetup, TwoFactorLogin } from './TwoFactorAuth'

const API_BASE = import.meta.env.VITE_API_BASE || ''

function PasswordStrengthIndicator({ password }) {
  const getStrength = (pwd) => {
    let strength = 0;
    if (pwd.length >= 8) strength++;
    if (/[A-Z]/.test(pwd)) strength++;
    if (/[a-z]/.test(pwd)) strength++;
    if (/\d/.test(pwd)) strength++;
    if (/[!@#$%^&*(),.?":{}|<>]/.test(pwd)) strength++;
    return strength;
  };
  
  const strength = getStrength(password);
  const labels = ["Very Weak", "Weak", "Fair", "Good", "Strong"];
  const colors = ["bg-red-500", "bg-orange-500", "bg-yellow-500", "bg-blue-500", "bg-green-500"];
  
  return (
    <div className="mt-2">
      <div className="flex space-x-1">
        {[1,2,3,4,5].map(i => (
          <div key={i} className={`h-2 w-8 rounded ${i <= strength ? colors[strength - 1] : 'bg-gray-300'}`}></div>
        ))}
      </div>
      <p className={`text-sm mt-1 ${strength >= 4 ? 'text-green-600' : strength >= 2 ? 'text-yellow-600' : 'text-red-600'}`}>
        {labels[strength - 1] || "Very Weak"}
      </p>
    </div>
  );
}

export function AuthenticationFlow({ onLogin }) {
  const [authStep, setAuthStep] = useState('login') // login -> register -> 2fa-login -> dashboard -> settings
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [credentials, setCredentials] = useState(null)
  const [show2FASetup, setShow2FASetup] = useState(false)
  const [twoFAStatus, setTwoFAStatus] = useState({ is_enabled: false, backup_codes_count: 0 })

  // Fetch current 2FA status
  const checkTwoFAStatus = async (token) => {
    try {
      const res = await fetch(`${API_BASE}/auth/totp/status`, {
        headers: { 'Authorization': `Bearer ${token}` },
      })
      if (res.ok) {
        const data = await res.json()
        setTwoFAStatus(data)
      }
    } catch (err) {
      console.error('Failed to fetch 2FA status:', err)
    }
  }

  // Step 0: Registration
  const handleRegisterSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setSuccess('')

    if (password !== confirmPassword) {
      setError('Passwords do not match')
      setLoading(false)
      return
    }

    try {
      const res = await fetch(`${API_BASE}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })

      if (res.ok) {
        setSuccess('Account created successfully! You can now log in.')
        setTimeout(() => {
          setAuthStep('login')
          setSuccess('')
        }, 2000)
      } else {
        const errorBody = await res.text()
        setError(errorBody || 'Registration failed')
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Step 1: Login with username/password
  const handleLoginSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      const formData = new FormData()
      formData.append('username', username)
      formData.append('password', password)

      const res = await fetch(`${API_BASE}/auth/token`, {
        method: 'POST',
        body: formData,
      })

      if (res.status === 403) {
        // 2FA is required
        setCredentials({ username, password })
        setAuthStep('2fa-login')
        sessionStorage.setItem('_temp_password', password)
      } else if (res.ok) {
        const data = await res.json()
        localStorage.setItem('fv_token', data.access_token)
        localStorage.setItem('fv_username', username)
        await checkTwoFAStatus(data.access_token)
        if (onLogin) onLogin()
        setAuthStep('dashboard')
      } else {
        const errorBody = await res.text()
        setError(errorBody || 'Login failed')
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Step 2: Complete 2FA login
  const handleTwoFASuccess = async () => {
    const token = localStorage.getItem('fv_token')
    await checkTwoFAStatus(token)
    if (onLogin) onLogin()
    setAuthStep('dashboard')
  }

  // Step 3: Handle logout
  const handleLogout = () => {
    localStorage.removeItem('fv_token')
    localStorage.removeItem('fv_username')
    sessionStorage.removeItem('_temp_password')
    setAuthStep('login')
    setUsername('')
    setPassword('')
    setError('')
    setTwoFAStatus({ is_enabled: false, backup_codes_count: 0 })
  }

  // Step 4: 2FA Setup success
  const handle2FASetupSuccess = async () => {
    setShow2FASetup(false)
    const token = localStorage.getItem('fv_token')
    await checkTwoFAStatus(token)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-600 to-indigo-800 flex items-center justify-center p-4">
      <AnimatePresence mode="wait">
        {/* ===================== REGISTRATION SCREEN ===================== */}
        {authStep === 'register' && (
          <motion.div
            key="register"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="w-full max-w-md bg-white rounded-2xl shadow-2xl p-8"
          >
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold text-gray-800 mb-2">
                Create Account
              </h1>
              <p className="text-gray-600">Join Face Verification</p>
            </div>

            {error && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg"
              >
                {error}
              </motion.div>
            )}

            {success && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mb-4 p-4 bg-green-100 border border-green-400 text-green-700 rounded-lg"
              >
                {success}
              </motion.div>
            )}

            <form onSubmit={handleRegisterSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Username
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Choose a username"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Password
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Create a strong password"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  required
                />
                {password && <PasswordStrengthIndicator password={password} />}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Confirm Password
                </label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Confirm your password"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full py-2 px-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold rounded-lg hover:shadow-lg transition-all disabled:opacity-50"
              >
                {loading ? 'Creating Account...' : 'Create Account'}
              </button>
            </form>

            <div className="text-center mt-4">
              <button
                onClick={() => {
                  setAuthStep('login')
                  setError('')
                  setSuccess('')
                }}
                className="text-blue-600 hover:text-blue-800 text-sm"
              >
                Already have an account? Login
              </button>
            </div>
          </motion.div>
        )}

        {/* ===================== LOGIN SCREEN ===================== */}
        {authStep === 'login' && (
          <motion.div
            key="login"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="w-full max-w-md bg-white rounded-2xl shadow-2xl p-8"
          >
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold text-gray-800 mb-2">
                🔐 Face Verification
              </h1>
              <p className="text-gray-600">With 2FA Protection</p>
            </div>

            {error && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg"
              >
                {error}
              </motion.div>
            )}

            <form onSubmit={handleLoginSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Username
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Enter your username"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Password
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full py-2 px-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white font-semibold rounded-lg hover:shadow-lg transition-all disabled:opacity-50"
              >
                {loading ? 'Logging in...' : 'Login'}
              </button>
            </form>

            <div className="text-center mt-4">
              <button
                onClick={() => {
                  setAuthStep('register')
                  setUsername('')
                  setPassword('')
                  setConfirmPassword('')
                  setError('')
                }}
                className="text-blue-600 hover:text-blue-800 text-sm"
              >
                Don't have an account? Create one
              </button>
            </div>
          </motion.div>
        )}

        {/* ===================== 2FA LOGIN SCREEN ===================== */}
        {authStep === '2fa-login' && (
          <motion.div
            key="2fa-login"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <TwoFactorLogin
              username={credentials?.username}
              onSuccess={handleTwoFASuccess}
              onCancel={() => {
                setAuthStep('login')
                setCredentials(null)
                sessionStorage.removeItem('_temp_password')
              }}
            />
          </motion.div>
        )}

        {/* ===================== DASHBOARD SCREEN ===================== */}
        {authStep === 'dashboard' && (
          <motion.div
            key="dashboard"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="w-full max-w-2xl bg-white rounded-2xl shadow-2xl p-8"
          >
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold text-gray-800 mb-2">
                ✅ Welcome!
              </h1>
              <p className="text-gray-600">
                {localStorage.getItem('fv_username')}
              </p>
            </div>

            {/* 2FA Status Card */}
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg mb-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-1">
                    🔐 Two-Factor Authentication
                  </h3>
                  <p className="text-gray-600">
                    {twoFAStatus.is_enabled
                      ? `✅ Enabled • ${twoFAStatus.backup_codes_count} backup codes remaining`
                      : '⚠️ Not enabled - Protect your account'}
                  </p>
                </div>
                <button
                  onClick={() => setShow2FASetup(true)}
                  className="px-6 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white font-semibold rounded-lg hover:shadow-lg transition-all"
                >
                  {twoFAStatus.is_enabled ? 'Manage' : 'Enable Now'}
                </button>
              </div>
            </div>

            {/* Features Preview */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold text-gray-800 mb-2">🎬 Face Verification</h4>
                <p className="text-sm text-gray-600">
                  Register and verify your face using deep learning
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold text-gray-800 mb-2">⏰ TOTP Tokens</h4>
                <p className="text-sm text-gray-600">
                  Time-based one-time passwords like Google Authenticator
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold text-gray-800 mb-2">🔄 Backup Codes</h4>
                <p className="text-sm text-gray-600">
                  Recovery codes for account access if you lose your phone
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold text-gray-800 mb-2">📊 ML/DL Models</h4>
                <p className="text-sm text-gray-600">
                  FaceNet, DeepFace, and other deep learning architectures
                </p>
              </div>
            </div>

            <button
              onClick={handleLogout}
              className="w-full py-2 px-4 bg-red-500 text-white font-semibold rounded-lg hover:bg-red-600 transition-all"
            >
              Logout
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 2FA Setup Modal */}
      <AnimatePresence>
        {show2FASetup && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50"
            onClick={() => setShow2FASetup(false)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              onClick={(e) => e.stopPropagation()}
            >
              <TwoFactorSetup
                onSuccess={handle2FASetupSuccess}
                onCancel={() => setShow2FASetup(false)}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default AuthenticationFlow
