# Advanced Real-Time Face Verification System

A comprehensive, production-ready face verification system built with FastAPI, React, and advanced AI technologies for real-time authentication and identification.

## 🚀 Features

### Core Functionality
- **Real-time Face Verification**: Live webcam verification with sub-second response times
- **Face Registration**: Secure user registration with multiple face samples
- **Advanced Authentication**: JWT-based auth with 2FA (TOTP) support
- **Face Identification**: Identify unknown faces against the entire database
- **WebSocket Streaming**: Real-time bidirectional communication for live verification

### Advanced Technologies
- **FAISS Vector Search**: O(1) similarity search for millions of faces
- **Multi-Model Ensemble**: DeepFace + FaceNet + custom models for accuracy
- **GPU Acceleration**: PyTorch/CUDA support for faster processing
- **LRU Caching**: Intelligent embedding caching for performance
- **Batch Processing**: Concurrent verification for multiple users
- **Image Quality Assessment**: Automatic quality checks and preprocessing

### System Monitoring
- **Real-time Dashboard**: Live system statistics and performance metrics
- **Performance Analytics**: Processing times, success rates, and throughput
- **Session Management**: Active session monitoring and management
- **Health Checks**: System status and capability detection

## 🏗️ Architecture

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   React Frontend│◄──────────────►│  FastAPI Backend │
│                 │                │                 │
│ - Real-time UI  │   HTTP/REST    │ - Face Processing│
│ - Camera Access │◄──────────────►│ - Auth System    │
│ - Dashboard     │                │ - WebSocket Hub  │
└─────────────────┘                └─────────────────┘
         │                                   │
         └─────────────► Storage ◄────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
              Embeddings       User Data
              (FAISS Index)    (JSON/SQLite)
```

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- Webcam access
- (Optional) CUDA-compatible GPU for acceleration

## 🛠️ Installation

### Backend Setup

1. **Clone and navigate to the project:**
```bash
cd face-verification
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install additional dependencies:**
```bash
pip install pyotp qrcode[pil]
```

### Frontend Setup

1. **Navigate to web directory:**
```bash
cd web
```

2. **Install Node.js dependencies:**
```bash
npm install
```

## 🚀 Running the System

### Start Backend Server
```bash
# From project root
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start Frontend Server
```bash
# From web directory
npm run dev
```

Access the application at: http://localhost:5173

## 📖 Usage Guide

### 1. User Registration & Authentication

1. **Create Account**: Use the registration form to create a user account
2. **Setup 2FA**: Enable TOTP-based two-factor authentication for enhanced security
3. **Login**: Authenticate with username/password + 2FA code

### 2. Face Registration

1. **Navigate to Home**: Use the main interface
2. **Enter User ID**: Input your registered username
3. **Start Camera**: Click "Start Camera" to access webcam
4. **Register Face**: Click "Register Face" to capture and store face embeddings

### 3. Real-time Verification

1. **Start Verification**: Click "Start Verify" for continuous verification
2. **Live Feedback**: See real-time verification results and confidence scores
3. **Quality Metrics**: Monitor image quality and processing performance

### 4. Advanced Features

#### Real-time Streaming Verification (`/realtime`)
- WebSocket-based live verification
- Continuous face tracking and verification
- Quality assessment and feedback

#### Face Identification (`/identify`)
- Identify unknown faces against database
- Confidence scoring and candidate ranking
- Batch processing capabilities

#### System Dashboard (`/dashboard`)
- Real-time system statistics
- Performance monitoring
- Active session management

## 🔧 API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/token` - Login with JWT
- `POST /auth/token-with-2fa` - Login with 2FA
- `POST /auth/totp/setup` - Setup TOTP 2FA
- `GET /auth/me` - Get current user info

### Face Operations
- `POST /register` - Register face embeddings
- `POST /verify` - Verify face against user
- `POST /verify-realtime` - Real-time verification with quality checks
- `POST /verify-batch` - Batch verification
- `POST /embed` - Extract face embeddings

### Advanced Features
- `POST /stream/start-session` - Start WebSocket verification session
- `POST /identify` - Identify unknown face
- `WebSocket /ws/verify` - Real-time verification stream
- `WebSocket /ws/identify` - Real-time identification stream

### Monitoring
- `GET /health` - System health check
- `GET /system/stats` - Comprehensive system statistics
- `GET /datastore/stats` - Datastore information
- `GET /performance-stats` - Performance metrics
- `GET /sessions/active` - Active verification sessions

## ⚙️ Configuration

### Environment Variables
```bash
# Backend
export SECRET_KEY="your-secret-key"
export ALGORITHM="HS256"
export ACCESS_TOKEN_EXPIRE_MINUTES=30

# Frontend
export VITE_API_BASE="http://localhost:8000"
```

### Model Configuration
The system automatically detects and uses available models:
- **DeepFace**: Primary model with multiple backends (Facenet512, ArcFace, VGG-Face)
- **FaceNet**: PyTorch implementation for GPU acceleration
- **MTCNN**: Face detection and alignment

## 📊 Performance Optimization

### Caching Strategy
- **LRU Cache**: Recently used embeddings cached in memory
- **FAISS Indexing**: Vector similarity search for fast lookups
- **Batch Processing**: Concurrent verification requests

### Quality Assurance
- **Image Preprocessing**: Automatic quality enhancement
- **Face Detection**: Robust detection with multiple fallbacks
- **Confidence Thresholding**: Adaptive thresholds based on image quality

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **2FA Support**: TOTP-based two-factor authentication
- **Rate Limiting**: Built-in request rate limiting
- **Input Validation**: Comprehensive input sanitization
- **CORS Protection**: Configured CORS policies

## 📈 Monitoring & Analytics

### Dashboard Features
- **Real-time Metrics**: Live system performance data
- **Session Tracking**: Active verification sessions
- **Performance Charts**: Historical performance trends
- **System Health**: Model availability and GPU status

### Logging
- **Structured Logging**: JSON-formatted logs for analysis
- **Performance Metrics**: Request timing and success rates
- **Error Tracking**: Comprehensive error logging

## 🧪 Testing

### Manual Testing
1. **Unit Tests**: Run individual component tests
2. **Integration Tests**: Test API endpoints
3. **Performance Tests**: Load testing with multiple concurrent users

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# System stats
curl http://localhost:8000/system/stats

# Datastore info
curl http://localhost:8000/datastore/stats
```

## 🚀 Deployment

### Production Setup
1. **Environment Setup**: Configure production environment variables
2. **Database**: Use PostgreSQL for user data in production
3. **Reverse Proxy**: Nginx for load balancing and SSL
4. **Monitoring**: Integrate with Prometheus/Grafana
5. **Scaling**: Docker containerization for horizontal scaling

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Camera Not Accessible**
- Ensure HTTPS in production (camera requires secure context)
- Check browser permissions for camera access

**Slow Performance**
- Enable GPU acceleration if available
- Check FAISS indexing status
- Monitor cache hit rates

**Model Loading Errors**
- Verify all dependencies are installed
- Check CUDA installation for GPU support
- Ensure sufficient RAM for model loading

### Support
- Check the dashboard for system status
- Review logs for detailed error information
- Test individual API endpoints for isolation

## 🔄 Future Enhancements

- **Mobile App**: React Native implementation
- **Cloud Integration**: AWS/Azure face recognition APIs
- **Advanced Analytics**: ML-based performance prediction
- **Multi-modal Authentication**: Voice + face verification
- **Federated Learning**: Privacy-preserving model updates# face-verification-authentication
