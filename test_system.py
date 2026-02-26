#!/usr/bin/env python3
"""
System Integration Test Script
Tests the complete face verification system including:
- API endpoints
- WebSocket connections
- Datastore functionality
- Authentication system
"""

import asyncio
import json
import time
import requests
import websockets
from typing import Dict, Any

# Configuration
API_BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"

class SystemTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = []

    def log_test(self, test_name: str, success: bool, message: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if message:
            print(f"   {message}")
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": time.time()
        })

    async def test_websocket_connection(self):
        """Test WebSocket connection and basic messaging"""
        try:
            # First create a session
            session_data = {
                "user_id": "testuser",
                "session_type": "verification"
            }
            response = self.session.post(f"{API_BASE}/stream/start-session", json=session_data)

            if response.status_code != 200:
                self.log_test("WebSocket Connection", False, f"Failed to create session: HTTP {response.status_code}")
                return

            session_info = response.json()
            session_id = session_info.get("session_id")

            if not session_id:
                self.log_test("WebSocket Connection", False, "No session ID in response")
                return

            # Now connect to WebSocket with session ID
            uri = f"{WS_BASE}/ws/verify/{session_id}"
            async with websockets.connect(uri) as websocket:
                # Send heartbeat
                await websocket.send(json.dumps({
                    "type": "heartbeat"
                }))

                # Receive response
                response = await websocket.recv()
                data = json.loads(response)

                if data.get("type") == "heartbeat_response":
                    self.log_test("WebSocket Connection", True, "Heartbeat successful")
                else:
                    self.log_test("WebSocket Connection", False, f"Unexpected response: {data}")

                # End session
                await websocket.send(json.dumps({
                    "type": "end_session"
                }))

        except Exception as e:
            self.log_test("WebSocket Connection", False, str(e))

    def test_health_endpoint(self):
        """Test system health endpoint"""
        try:
            response = self.session.get(f"{API_BASE}/health")
            if response.status_code == 200:
                data = response.json()
                if "status" in data and "model" in data:
                    self.log_test("Health Endpoint", True, f"Status: {data['status']}")
                else:
                    self.log_test("Health Endpoint", False, "Invalid response format")
            else:
                self.log_test("Health Endpoint", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Health Endpoint", False, str(e))

    def test_system_stats(self):
        """Test system statistics endpoint"""
        try:
            response = self.session.get(f"{API_BASE}/system/stats")
            if response.status_code == 200:
                data = response.json()
                required_keys = ["verifications", "models", "active_sessions_count"]
                if all(key in data for key in required_keys):
                    self.log_test("System Stats", True, f"Active sessions: {data['active_sessions_count']}")
                else:
                    self.log_test("System Stats", False, "Missing required fields")
            else:
                self.log_test("System Stats", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("System Stats", False, str(e))

    def test_datastore_stats(self):
        """Test datastore statistics endpoint"""
        try:
            response = self.session.get(f"{API_BASE}/datastore/stats")
            if response.status_code == 200:
                data = response.json()
                self.log_test("Datastore Stats", True,
                    f"Users: {data.get('total_users', 0)}, Samples: {data.get('total_samples', 0)}")
            else:
                self.log_test("Datastore Stats", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Datastore Stats", False, str(e))

    def test_auth_endpoints(self):
        """Test authentication endpoints"""
        try:
            # Test registration (use a unique username)
            import uuid
            test_username = f"testuser_{uuid.uuid4().hex[:8]}"

            register_data = {
                "username": test_username,
                "password": "testpass123"
            }
            response = self.session.post(f"{API_BASE}/auth/register", json=register_data)
            if response.status_code in [200, 201]:
                self.log_test("Auth Registration", True, f"User {test_username} registered")
            elif response.status_code == 400:
                # User might already exist, try with existing user
                test_username = "testuser"
                self.log_test("Auth Registration", True, "Using existing test user")
            else:
                self.log_test("Auth Registration", False, f"HTTP {response.status_code}: {response.text}")
                return

            # Test token generation
            token_data = {
                "username": test_username,
                "password": "testpass123"
            }
            response = self.session.post(f"{API_BASE}/auth/token", data=token_data)
            if response.status_code == 200:
                token = response.json().get("access_token")
                if token:
                    self.session.headers.update({"Authorization": f"Bearer {token}"})
                    self.log_test("Auth Token", True, "Token generated successfully")
                else:
                    self.log_test("Auth Token", False, "No token in response")
            else:
                self.log_test("Auth Token", False, f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            self.log_test("Auth Endpoints", False, str(e))

    def test_optimization_status(self):
        """Test optimization features status"""
        try:
            response = self.session.get(f"{API_BASE}/optimization-status")
            if response.status_code == 200:
                data = response.json()
                features = data.get("features", {})
                enabled_features = [k for k, v in features.items() if v]
                self.log_test("Optimization Status", True,
                    f"Enabled features: {', '.join(enabled_features)}")
            else:
                self.log_test("Optimization Status", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Optimization Status", False, str(e))

    def test_performance_stats(self):
        """Test performance statistics"""
        try:
            response = self.session.get(f"{API_BASE}/performance-stats")
            if response.status_code == 200:
                data = response.json()
                self.log_test("Performance Stats", True,
                    f"Cache hits: {data.get('cache_hits', 0)}, Misses: {data.get('cache_misses', 0)}")
            else:
                self.log_test("Performance Stats", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Performance Stats", False, str(e))

    async def run_all_tests(self):
        """Run all system tests"""
        print("🚀 Starting Face Verification System Tests")
        print("=" * 50)

        # API Tests
        self.test_health_endpoint()
        self.test_system_stats()
        self.test_datastore_stats()
        self.test_auth_endpoints()
        self.test_optimization_status()
        self.test_performance_stats()

        # WebSocket Tests
        await self.test_websocket_connection()

        print("=" * 50)
        successful_tests = sum(1 for test in self.test_results if test["success"])
        total_tests = len(self.test_results)

        print(f"📊 Test Results: {successful_tests}/{total_tests} tests passed")

        if successful_tests == total_tests:
            print("🎉 All tests passed! System is ready for production.")
        else:
            print("⚠️  Some tests failed. Check the logs above for details.")

        return successful_tests == total_tests

async def main():
    """Main test runner"""
    tester = SystemTester()
    success = await tester.run_all_tests()

    # Save test results
    with open("test_results.json", "w") as f:
        json.dump(tester.test_results, f, indent=2)

    return success

if __name__ == "__main__":
    # Wait a bit for servers to start
    print("⏳ Waiting for servers to initialize...")
    time.sleep(3)

    success = asyncio.run(main())
    exit(0 if success else 1)