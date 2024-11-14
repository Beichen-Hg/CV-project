import React, { useState, useRef, useEffect } from 'react';

function CameraDetection() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionResults, setDetectionResults] = useState([]);
  const [stream, setStream] = useState(null);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.onloadedmetadata = () => {
        if (canvasRef.current) {
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
        }
      };
    }
  }, []);

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
      
      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw current video frame on canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Convert canvas content to base64
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
    
    // Continue with next frame detection
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
    if (!canvasRef.current || !videoRef.current || !videoRef.current.videoWidth) {
      return;
    }
  
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    // Ensure canvas dimensions match video
    if (canvas.width !== videoRef.current.videoWidth) {
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
    }
    
    // Clear previous drawings
    context.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw current video frame
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    
    // Draw detection boxes and labels
    detectionResults.forEach(detection => {
      const [x1, y1, x2, y2] = detection.bbox;
      const label = `${detection.class} (${(detection.confidence * 100).toFixed(1)}%)`;
      
      // Draw bounding box
      context.strokeStyle = '#00ff00';
      context.lineWidth = 2;
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
      
      // Draw label background
      context.fillStyle = 'rgba(0, 255, 0, 0.5)';
      const textWidth = context.measureText(label).width;
      context.fillRect(x1, y1 - 25, textWidth + 10, 20);
      
      // Draw label text
      context.fillStyle = '#000000';
      context.font = '14px Arial';
      context.fillText(label, x1 + 5, y1 - 10);
    });
  };

  useEffect(() => {
    let animationFrameId;
    
    const updateCanvas = () => {
      drawDetections();
      if (isDetecting) {
        animationFrameId = requestAnimationFrame(updateCanvas);
      }
    };
    
    if (isDetecting) {
      updateCanvas();
    }
    
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [isDetecting, detectionResults]);

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
          style={{ position: 'absolute', opacity: 0 }}
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
