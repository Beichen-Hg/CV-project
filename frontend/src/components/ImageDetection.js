import React, { useState } from 'react';

function ImageDetection() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [detectionResult, setDetectionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
      setIsLoading(true);
      
      const formData = new FormData();
      formData.append('image', file);

      try {
        const response = await fetch('http://localhost:5000/api/image-detection', {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        setDetectionResult(data);
      } catch (error) {
        console.error('Error:', error);
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="mb-4">
      <h2 className="mb-4">Image Detection</h2>
      
      <div className="mb-3">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="form-control"
        />
      </div>

      {selectedImage && (
        <div className="mb-3">
          <img
            src={selectedImage}
            alt="Selected"
            className="img-fluid rounded shadow-sm"
            style={{maxHeight: '400px'}}
          />
        </div>
      )}

      {isLoading && (
        <div className="text-center mb-3">
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
          <p className="mt-2">Processing...</p>
        </div>
      )}

      {detectionResult && (
        <div className="card mb-3">
          <div className="card-body">
            <h5 className="card-title">Detection Result</h5>
            <pre className="bg-light p-3 rounded">
              {JSON.stringify(detectionResult, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default ImageDetection;