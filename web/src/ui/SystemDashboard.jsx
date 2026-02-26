import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

const API_BASE = import.meta.env.VITE_API_BASE || ''

export function SystemDashboard() {
  const [systemStats, setSystemStats] = useState(null)
  const [datastoreStats, setDatastoreStats] = useState(null)
  const [activeSessions, setActiveSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Fetch all statistics
  const fetchStats = async () => {
    try {
      setLoading(true)
      setError(null)

      const [systemRes, datastoreRes, sessionsRes] = await Promise.all([
        fetch(`${API_BASE}/system/stats`),
        fetch(`${API_BASE}/datastore/stats`),
        fetch(`${API_BASE}/sessions/active`)
      ])

      if (!systemRes.ok || !datastoreRes.ok || !sessionsRes.ok) {
        throw new Error('Failed to fetch statistics')
      }

      const [systemData, datastoreData, sessionsData] = await Promise.all([
        systemRes.json(),
        datastoreRes.json(),
        sessionsRes.json()
      ])

      setSystemStats(systemData)
      setDatastoreStats(datastoreData)
      setActiveSessions(sessionsData.active_sessions || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStats()
    // Refresh every 5 seconds
    const interval = setInterval(fetchStats, 5000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading system statistics...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center text-red-600">
          <div className="text-6xl mb-4">❌</div>
          <p className="text-xl font-semibold mb-2">Error Loading Dashboard</p>
          <p>{error}</p>
          <button
            onClick={fetchStats}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Face Verification System Dashboard
          </h1>
          <p className="text-gray-600">
            Real-time performance monitoring and analytics
          </p>
        </div>

        {/* System Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Verification Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white p-6 rounded-lg shadow-sm border"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Verifications</p>
                <p className="text-3xl font-bold text-blue-600">
                  {systemStats?.verifications?.total || 0}
                </p>
              </div>
              <div className="text-4xl">🔍</div>
            </div>
            <div className="mt-4">
              <p className="text-sm text-green-600">
                Success Rate: {((systemStats?.verifications?.success_rate || 0) * 100).toFixed(1)}%
              </p>
            </div>
          </motion.div>

          {/* Active Sessions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white p-6 rounded-lg shadow-sm border"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Active Sessions</p>
                <p className="text-3xl font-bold text-green-600">
                  {activeSessions.length}
                </p>
              </div>
              <div className="text-4xl">📹</div>
            </div>
            <div className="mt-4">
              <p className="text-sm text-gray-600">
                Real-time streams
              </p>
            </div>
          </motion.div>

          {/* Datastore Users */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white p-6 rounded-lg shadow-sm border"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Registered Users</p>
                <p className="text-3xl font-bold text-purple-600">
                  {datastoreStats?.total_users || 0}
                </p>
              </div>
              <div className="text-4xl">👥</div>
            </div>
            <div className="mt-4">
              <p className="text-sm text-gray-600">
                {datastoreStats?.total_samples || 0} total samples
              </p>
            </div>
          </motion.div>

          {/* Performance */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white p-6 rounded-lg shadow-sm border"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Avg Processing Time</p>
                <p className="text-3xl font-bold text-orange-600">
                  {systemStats?.verifications?.average_processing_time_ms?.toFixed(0) || 0}ms
                </p>
              </div>
              <div className="text-4xl">⚡</div>
            </div>
            <div className="mt-4">
              <p className="text-sm text-gray-600">
                Real-time performance
              </p>
            </div>
          </motion.div>
        </div>

        {/* Detailed Stats */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* System Capabilities */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white p-6 rounded-lg shadow-sm border"
          >
            <h3 className="text-xl font-semibold mb-4 text-gray-800">System Capabilities</h3>
            <div className="space-y-3">
              {systemStats?.models && Object.entries(systemStats.models).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between">
                  <span className="text-gray-600 capitalize">
                    {key.replace('_', ' ')}:
                  </span>
                  <span className={`px-2 py-1 rounded text-sm font-semibold ${
                    value ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                  }`}>
                    {value ? '✓' : '✗'}
                  </span>
                </div>
              ))}
            </div>

            <div className="mt-6 pt-4 border-t">
              <h4 className="font-semibold text-gray-800 mb-2">Datastore Info</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Indexing:</span>
                  <span className={datastoreStats?.indexing_enabled ? 'text-green-600' : 'text-red-600'}>
                    {datastoreStats?.indexing_enabled ? 'FAISS Enabled' : 'Linear Search'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Embedding Dim:</span>
                  <span>{datastoreStats?.embedding_dim || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg Samples/User:</span>
                  <span>{datastoreStats?.avg_samples_per_user?.toFixed(1) || 'N/A'}</span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Active Sessions */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white p-6 rounded-lg shadow-sm border"
          >
            <h3 className="text-xl font-semibold mb-4 text-gray-800">
              Active Sessions ({activeSessions.length})
            </h3>

            {activeSessions.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <div className="text-4xl mb-2">📷</div>
                <p>No active verification sessions</p>
              </div>
            ) : (
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {activeSessions.map((session) => (
                  <div key={session.session_id} className="border rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-sm text-gray-800">
                        {session.user_id || 'Unknown User'}
                      </span>
                      <span className="text-xs text-gray-500">
                        {new Date(session.created).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                      <div>Frames: {session.frames_processed}</div>
                      <div>Verifications: {session.verifications_count}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        </div>

        {/* Performance Charts Placeholder */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white p-6 rounded-lg shadow-sm border"
        >
          <h3 className="text-xl font-semibold mb-4 text-gray-800">Performance Trends</h3>
          <div className="text-center py-12 text-gray-500">
            <div className="text-6xl mb-4">📊</div>
            <p className="text-lg mb-2">Performance Analytics</p>
            <p className="text-sm">
              Real-time charts and historical data would be displayed here.
              <br />
              Integration with monitoring tools like Grafana/Prometheus recommended for production.
            </p>
          </div>
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white p-6 rounded-lg shadow-sm border"
        >
          <h3 className="text-xl font-semibold mb-4 text-gray-800">Quick Actions</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button
              onClick={fetchStats}
              className="p-4 border-2 border-blue-200 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-all"
            >
              <div className="text-2xl mb-2">🔄</div>
              <div className="font-semibold text-blue-800">Refresh Stats</div>
              <div className="text-sm text-blue-600">Update dashboard data</div>
            </button>

            <button
              onClick={() => window.open('/docs', '_blank')}
              className="p-4 border-2 border-green-200 rounded-lg hover:border-green-400 hover:bg-green-50 transition-all"
            >
              <div className="text-2xl mb-2">📚</div>
              <div className="font-semibold text-green-800">API Docs</div>
              <div className="text-sm text-green-600">View API documentation</div>
            </button>

            <button
              onClick={() => window.location.href = '/'}
              className="p-4 border-2 border-purple-200 rounded-lg hover:border-purple-400 hover:bg-purple-50 transition-all"
            >
              <div className="text-2xl mb-2">🏠</div>
              <div className="font-semibold text-purple-800">Back to App</div>
              <div className="text-sm text-purple-600">Return to main application</div>
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default SystemDashboard
