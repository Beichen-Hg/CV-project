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
        <Header />
        <div className="container py-4">
          <nav className="mb-4">
            <ul className="nav nav-pills">
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

          <Routes>
            <Route path="/" element={<h1 className="display-4">Welcome to Fruit Detection!</h1>} />
            <Route path="/image-detection" element={<ImageDetection />} />
            <Route path="/camera-detection" element={<CameraDetection />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;