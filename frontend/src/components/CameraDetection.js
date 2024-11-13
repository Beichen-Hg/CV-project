import React, { useState, useRef, useEffect } from 'react';

function CameraDetection() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionResults, setDetectionResults] = useState([]);
  const [stream, setStream] = useState(null);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = mediaStream;
      setStream(mediaStream);
      setIsDetecting(true);
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setStream(null);
      setIsDetecting(false);
      setDetectionResults([]);
    }
  };

  const captureFrame = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const context = canvas.getContext('2d');
      
      // 设置canvas尺寸与视频相同
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // 在canvas上绘制当前视频帧
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // 将canvas内容转换为base64
      return canvas.toDataURL('image/jpeg');
    }
    return null;
  };

  const performDetection = async () => {
    if (!isDetecting) return;

    const frameData = captureFrame();
    if (frameData) {
      try {
        const response = await fetch('http://localhost:5000/api/camera-detection', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ frame: frameData }),
        });
        
        const data = await response.json();
        if (data.success) {
          setDetectionResults(data.detections);
        }
      } catch (error) {
        console.error('Detection error:', error);
      }
    }
    
    // 继续下一帧检测
    requestAnimationFrame(performDetection);
  };

  useEffect(() => {
    if (isDetecting) {
      performDetection();
    }
    return () => {
      stopCamera();
    };
  }, [isDetecting]);

  const drawDetections = () => {
    if (canvasRef.current && detectionResults.length > 0) {
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      // 清除之前的绘制
      context.clearRect(0, 0, canvas.width, canvas.height);
      
      // 绘制当前视频帧
      context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      
      // 绘制检测框和标签
      detectionResults.forEach(detection => {
        const [x1, y1, x2, y2] = detection.bbox;
        const label = `${detection.class} (${(detection.confidence * 100).toFixed(1)}%)`;
        
        // 绘制边界框
        context.strokeStyle = '#00ff00';
        context.lineWidth = 2;
        context.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // 绘制标签背景
        context.fillStyle = '#00ff00';
        context.fillRect(x1, y1 - 20, context.measureText(label).width + 10, 20);
        
        // 绘制标签文字
        context.fillStyle = '#000000';
        context.font = '16px Arial';
        context.fillText(label, x1 + 5, y1 - 5);
      });
    }
  };

  useEffect(() => {
    drawDetections();
  }, [detectionResults]);

  return (
    <div className="mb-4">
      <h2 className="mb-4">Camera Detection</h2>
      
      <div className="mb-3">
        <button
          onClick={isDetecting ? stopCamera : startCamera}
          className={`btn ${isDetecting ? 'btn-danger' : 'btn-primary'} me-2`}
        >
          {isDetecting ? 'Stop Detection' : 'Start Detection'}
        </button>
      </div>
      
      <div className="position-relative">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{ display: 'none' }}
        />
        <canvas
          ref={canvasRef}
          className="w-100"
          style={{ maxWidth: '800px' }}
        />
      </div>
      
      <div className="mt-3">
        <h3>Detection Results:</h3>
        <ul className="list-group">
          {detectionResults.map((result, index) => (
            <li key={index} className="list-group-item">
              {result.class} - Confidence: {(result.confidence * 100).toFixed(1)}%
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default CameraDetection;