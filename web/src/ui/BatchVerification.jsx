import React, { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const API_BASE = import.meta.env.VITE_API_BASE || ''

export function BatchVerification() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [selectedUsers, setSelectedUsers] = useState([])
  const [allUsers, setAllUsers] = useState([])
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const fileRef = useRef(null)

  // Load all users
  React.useEffect(() => {
    const fetchUsers = async () => {
      try {
        const res = await fetch(`${API_BASE}/users`)
        if (res.ok) {
          const data = await res.json()
          setAllUsers(data.users.map((u) => u.user_id))
        }
      } catch (err) {
        console.error('Failed to load users:', err)
      }
    }
    fetchUsers()
  }, [])

  // Handle file selection
  const handleFileSelect = (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setSelectedFile(file)

    // Create preview
    const reader = new FileReader()
    reader.onload = (ev) => {
      setPreview(ev.target?.result)
    }
    reader.readAsDataURL(file)
  }

  // Toggle user selection
  const toggleUser = (userId) => {
    setSelectedUsers((prev) =>
      prev.includes(userId) ? prev.filter((u) => u !== userId) : [...prev, userId]
    )
  }

  // Filter users
  const filteredUsers = allUsers.filter((u) =>
    u.toLowerCase().includes(searchTerm.toLowerCase())
  )

  // Batch verify
  const handleBatchVerify = async () => {
    if (!selectedFile || selectedUsers.length === 0) {
      alert('Please select an image and at least one user')
      return
    }

    setLoading(true)
    try {
      const fileReader = new FileReader()
      fileReader.onload = async (ev) => {
        const imageData = ev.target?.result

        const res = await fetch(`${API_BASE}/verify-batch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: imageData,
            user_ids: selectedUsers,
          }),
        })

        if (!res.ok) {
          throw new Error(await res.text())
        }

        const data = await res.json()
        setResults(data)
      }
      fileReader.readAsDataURL(selectedFile)
    } catch (err) {
      console.error('Batch verification error:', err)
      alert(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg shadow-lg">
      <h2 className="text-3xl font-bold mb-6 text-center text-gray-800">
        🔍 Batch Face Identification
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Image Upload */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-white p-6 rounded-lg shadow-md"
        >
          <h3 className="text-xl font-semibold mb-4 text-gray-800">1. Upload Photo</h3>

          <div
            onClick={() => fileRef.current?.click()}
            className="border-2 border-dashed border-blue-400 rounded-lg p-8 text-center cursor-pointer hover:border-blue-600 hover:bg-blue-50 transition-all"
          >
            {preview ? (
              <div className="space-y-3">
                <img src={preview} alt="Preview" className="w-full max-h-64 rounded-lg" />
                <div className="text-sm text-gray-600">
                  {selectedFile?.name}
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    setSelectedFile(null)
                    setPreview(null)
                  }}
                  className="text-red-500 text-sm hover:text-red-700"
                >
                  Change Image
                </button>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="text-4xl">📷</div>
                <p className="text-gray-700 font-semibold">Click to upload image</p>
                <p className="text-sm text-gray-500">JPG, PNG, or WebP</p>
              </div>
            )}
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>
        </motion.div>

        {/* User Selection */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-white p-6 rounded-lg shadow-md"
        >
          <h3 className="text-xl font-semibold mb-4 text-gray-800">
            2. Select Users ({selectedUsers.length}/{allUsers.length})
          </h3>

          <input
            type="text"
            placeholder="Search users..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg mb-3 focus:border-blue-500 focus:outline-none"
          />

          <div className="max-h-64 overflow-y-auto space-y-2 border border-gray-200 rounded-lg p-3 bg-gray-50">
            {filteredUsers.length > 0 ? (
              filteredUsers.map((userId) => (
                <label
                  key={userId}
                  className="flex items-center p-2 hover:bg-white rounded cursor-pointer transition-all"
                >
                  <input
                    type="checkbox"
                    checked={selectedUsers.includes(userId)}
                    onChange={() => toggleUser(userId)}
                    className="w-4 h-4 rounded"
                  />
                  <span className="ml-2 text-gray-700">{userId}</span>
                </label>
              ))
            ) : (
              <p className="text-gray-500 text-center py-4">No users found</p>
            )}
          </div>

          <button
            onClick={() => setSelectedUsers(allUsers)}
            className="mt-3 w-full text-sm text-blue-600 hover:text-blue-800 font-semibold"
          >
            Select All
          </button>
        </motion.div>
      </div>

      {/* Verify Button */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={handleBatchVerify}
        disabled={loading || !selectedFile || selectedUsers.length === 0}
        className="w-full py-3 px-6 bg-gradient-to-r from-blue-500 to-indigo-600 text-white font-bold rounded-lg hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? 'Identifying...' : '🚀 Identify Face'}
      </motion.button>

      {/* Results */}
      <AnimatePresence>
        {results && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mt-8 space-y-4"
          >
            <div className="bg-white p-6 rounded-lg shadow-lg">
              <h3 className="text-2xl font-bold mb-4 text-gray-800">Results</h3>

              {/* Processing Time */}
              <div className="mb-4 text-sm text-gray-600">
                Processing time: {results.processing_time_ms}ms
              </div>

              {/* Top Match */}
              {results.results && results.results.length > 0 && (
                <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border-2 border-green-400">
                  <h4 className="font-bold text-lg mb-2 text-gray-800">Best Match</h4>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div>
                      <div className="text-sm text-gray-600">User ID</div>
                      <div className="text-2xl font-bold text-green-600">
                        {results.results[0].user_id}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Similarity</div>
                      <div className="text-2xl font-bold">
                        {results.results[0].similarity.toFixed(4)}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Status</div>
                      <div className="text-2xl">
                        {results.results[0].verified ? '✅' : '❌'}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Confidence</div>
                      <div className="text-2xl font-bold">
                        {(results.results[0].confidence * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* All Results Ranked */}
              <div className="space-y-2">
                <h4 className="font-bold text-gray-800 mb-3">All Matches (Ranked)</h4>
                {results.results?.map((result, idx) => (
                  <motion.div
                    key={result.user_id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    className={`p-3 rounded-lg ${
                      result.verified
                        ? 'bg-green-100 border-l-4 border-green-500'
                        : 'bg-gray-100 border-l-4 border-gray-400'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="font-semibold text-gray-800">
                          {idx + 1}. {result.user_id}
                        </div>
                        <div className="text-sm text-gray-600">
                          Similarity: {result.similarity.toFixed(4)} • Confidence:{' '}
                          {(result.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="text-2xl">
                        {result.verified ? '✅' : '❌'}
                      </div>
                    </div>

                    {/* Confidence Bar */}
                    <div className="mt-2 bg-gray-300 rounded-full h-2 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${result.similarity * 100}%` }}
                        className={`h-full ${
                          result.verified
                            ? 'bg-green-500'
                            : 'bg-gray-500'
                        }`}
                      />
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Info Box */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg text-sm text-gray-700"
      >
        <p className="font-semibold mb-1">💡 How it works:</p>
        <ul className="list-disc list-inside space-y-1 text-xs">
          <li>Upload a photo of an unknown face</li>
          <li>Select users to search against</li>
          <li>System compares the face against all selected users</li>
          <li>Results ranked by similarity score</li>
          <li>Green checkmark = verified match (above threshold)</li>
        </ul>
      </motion.div>
    </div>
  )
}

export default BatchVerification
