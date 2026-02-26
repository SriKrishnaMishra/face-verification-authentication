import React, { useState } from 'react'
import { motion } from 'framer-motion'

const API_BASE = import.meta.env.VITE_API_BASE || ''

export function TwoFactorSetup({ onSuccess, onCancel }) {
  const [step, setStep] = useState('prepare') // prepare -> show -> verify
  const [qrCode, setQrCode] = useState(null)
  const [secret, setSecret] = useState(null)
  const [backupCodes, setBackupCodes] = useState([])
  const [totpCode, setTotpCode] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handlePrepare = async () => {
    setLoading(true)
    setError('')
    try {
      const token = localStorage.getItem('fv_token')
      const res = await fetch(`${API_BASE}/auth/totp/prepare`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setQrCode(data.qr_code)
      setSecret(data.secret)
      setBackupCodes(data.backup_codes)
      setStep('show')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleVerify = async () => {
    if (totpCode.length !== 6) {
      setError('TOTP code must be 6 digits')
      return
    }

    setLoading(true)
    setError('')
    try {
      const token = localStorage.getItem('fv_token')
      const username = localStorage.getItem('fv_username')
      const res = await fetch(`${API_BASE}/auth/totp/enable`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ username, totp_code: totpCode }),
      })
      if (!res.ok) throw new Error(await res.text())
      setStep('success')
      if (onSuccess) onSuccess()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-md mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg shadow-lg"
    >
      <h2 className="text-2xl font-bold mb-6 text-center text-gray-800">
        Set Up 2-Factor Authentication
      </h2>

      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded"
        >
          {error}
        </motion.div>
      )}

      {step === 'prepare' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <p className="text-gray-600 mb-4 text-center">
            Secure your account with Google Authenticator or similar TOTP apps.
          </p>
          <button
            onClick={handlePrepare}
            disabled={loading}
            className="w-full py-2 px-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white font-semibold rounded-lg hover:shadow-lg transition-all disabled:opacity-50"
          >
            {loading ? 'Preparing...' : 'Get Started'}
          </button>
          {onCancel && (
            <button
              onClick={onCancel}
              className="w-full mt-2 py-2 px-4 bg-gray-200 text-gray-700 font-semibold rounded-lg hover:bg-gray-300 transition-all"
            >
              Cancel
            </button>
          )}
        </motion.div>
      )}

      {step === 'show' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div className="mb-6 space-y-4">
            <div className="bg-white p-4 rounded-lg">
              <p className="text-sm text-gray-600 mb-3 font-semibold">
                1. Scan this QR code with your authenticator app:
              </p>
              {qrCode && (
                <img
                  src={`data:image/png;base64,${qrCode}`}
                  alt="TOTP QR Code"
                  className="w-48 h-48 mx-auto border-4 border-gray-200 rounded-lg"
                />
              )}
            </div>

            <div className="bg-white p-4 rounded-lg">
              <p className="text-sm text-gray-600 mb-2 font-semibold">
                Or enter this key manually:
              </p>
              <code className="block w-full p-2 bg-gray-100 text-gray-800 rounded text-center font-mono text-lg break-words">
                {secret}
              </code>
            </div>

            <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
              <p className="text-sm text-yellow-800 font-semibold mb-2">
                ⚠️ Save your backup codes (use if you lose access to your authenticator):
              </p>
              <div className="grid grid-cols-2 gap-2">
                {backupCodes.map((code, i) => (
                  <code
                    key={i}
                    className="text-xs bg-yellow-100 p-1 rounded text-center font-mono"
                  >
                    {code}
                  </code>
                ))}
              </div>
            </div>
          </div>

          <button
            onClick={() => setStep('verify')}
            className="w-full py-2 px-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold rounded-lg hover:shadow-lg transition-all"
          >
            I've Saved My Codes
          </button>
        </motion.div>
      )}

      {step === 'verify' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <p className="text-gray-600 mb-4 text-center">
            Enter a 6-digit code from your authenticator app to confirm:
          </p>
          <input
            type="text"
            maxLength="6"
            value={totpCode}
            onChange={(e) => setTotpCode(e.target.value.replace(/\D/g, ''))}
            placeholder="000000"
            className="w-full py-3 px-4 mb-4 text-center text-2xl tracking-widest border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
          />
          <button
            onClick={handleVerify}
            disabled={loading || totpCode.length !== 6}
            className="w-full py-2 px-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white font-semibold rounded-lg hover:shadow-lg transition-all disabled:opacity-50"
          >
            {loading ? 'Verifying...' : 'Verify & Enable'}
          </button>
          <button
            onClick={() => {
              setStep('show')
              setTotpCode('')
              setError('')
            }}
            className="w-full mt-2 py-2 px-4 bg-gray-200 text-gray-700 font-semibold rounded-lg hover:bg-gray-300 transition-all"
          >
            Back
          </button>
        </motion.div>
      )}

      {step === 'success' && (
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="text-center"
        >
          <div className="text-6xl mb-4">✅</div>
          <h3 className="text-2xl font-bold text-green-600 mb-2">
            2FA Enabled!
          </h3>
          <p className="text-gray-600 mb-6">
            Your account is now protected with two-factor authentication.
          </p>
          <button
            onClick={onSuccess}
            className="w-full py-2 px-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold rounded-lg hover:shadow-lg transition-all"
          >
            Done
          </button>
        </motion.div>
      )}
    </motion.div>
  )
}

export function TwoFactorLogin({ username, onSuccess, onCancel }) {
  const [totpCode, setTotpCode] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [useBackupCode, setUseBackupCode] = useState(false)

  const handleLogin = async () => {
    if (totpCode.length < 6) {
      setError('Please enter a valid code')
      return
    }

    setLoading(true)
    setError('')
    try {
      const url = `${API_BASE}/auth/token-with-2fa?totp_code=${encodeURIComponent(totpCode)}`
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(
          sessionStorage.getItem('_temp_password') || ''
        )}&grant_type=password`,
      })

      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      localStorage.setItem('fv_token', data.access_token)
      localStorage.setItem('fv_username', username)
      if (onSuccess) onSuccess()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-md mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg shadow-lg"
    >
      <h2 className="text-2xl font-bold mb-6 text-center text-gray-800">
        Two-Factor Authentication
      </h2>

      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded"
        >
          {error}
        </motion.div>
      )}

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            {useBackupCode ? 'Backup Code' : '6-Digit Code'}
          </label>
          <input
            type="text"
            maxLength={useBackupCode ? '8' : '6'}
            value={totpCode}
            onChange={(e) =>
              setTotpCode(useBackupCode ? e.target.value.toUpperCase() : e.target.value.replace(/\D/g, ''))
            }
            placeholder={useBackupCode ? 'XXXXXXXX' : '000000'}
            className="w-full py-3 px-4 text-center text-2xl tracking-widest border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
          />
        </div>

        <button
          onClick={handleLogin}
          disabled={loading || totpCode.length < 6}
          className="w-full py-2 px-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white font-semibold rounded-lg hover:shadow-lg transition-all disabled:opacity-50"
        >
          {loading ? 'Verifying...' : 'Verify'}
        </button>

        <button
          onClick={() => {
            setUseBackupCode(!useBackupCode)
            setTotpCode('')
          }}
          className="w-full py-2 px-4 text-blue-600 font-semibold text-sm hover:text-blue-800 transition-all"
        >
          {useBackupCode ? 'Use TOTP Code Instead' : 'Use Backup Code Instead'}
        </button>

        {onCancel && (
          <button
            onClick={onCancel}
            className="w-full py-2 px-4 bg-gray-200 text-gray-700 font-semibold rounded-lg hover:bg-gray-300 transition-all"
          >
            Cancel
          </button>
        )}
      </div>
    </motion.div>
  )
}

export default TwoFactorSetup
