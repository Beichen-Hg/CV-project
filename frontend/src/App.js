import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import ImageDetection from './components/ImageDetection';
import CameraDetection from './components/CameraDetection';
import Header from './components/Header';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  return (
    <Router>
      <div className="min-vh-100 bg-light">
        <Header /> {/* Page header component */}
        <div className="container py-4"> {/* Bootstrap container, top and bottom padding 4 units */}
          <nav className="mb-4">
            <ul className="nav nav-pills"> {/* Bootstrap pills navigation bar */}
              <li className="nav-item">
                <Link to="/" className="nav-link">Home</Link>
              </li>
              <li className="nav-item">
                <Link to="/image-detection" className="nav-link">Image Detection</Link>
              </li>
              <li className="nav-item">
                <Link to="/camera-detection" className="nav-link">Camera Detection</Link>
              </li>
            </ul>
          </nav>

          {/* Route config */}
          <Routes> 
            <Route path="/" element={
              <>
                <h1 className="display-4">Welcome to Fruit Detection!</h1>
                <br></br>
                <p className="lead">
                  This application demonstrates fruit detection using advanced machine learning techniques. 
                  You can upload images or use your camera to detect various types of fruits in real-time.
                </p>
                <br></br>
                <h5>
                  Choose "Image Detection" to upload and analyze images, or "Camera Detection" to use your device's camera for live fruit detection.
                </h5>
              </>
            } />
            {/* ImageDetection page route */}
            <Route path="/image-detection" element={<ImageDetection />} />
            {/* CameraDetection page route */}
            <Route path="/camera-detection" element={<CameraDetection />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
