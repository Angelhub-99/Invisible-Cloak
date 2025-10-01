from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
import threading
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

class InvisibleCloak:
    def __init__(self):
        self.cap = None
        self.background = None
        self.previous_masks = []
        self.max_previous_masks = 5
        self.frame_count = 0
        self.is_streaming = False
        self.is_capturing_background = False
        
    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        print("Attempting to initialize camera...")
        
        # Try multiple camera indices
        for camera_index in [0, 1, 2, -1]:
            try:
                print(f"Trying camera index: {camera_index}")
                self.cap = cv2.VideoCapture(camera_index)
                
                if self.cap is not None and self.cap.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"Camera {camera_index} opened successfully!")
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        return True
                    else:
                        print(f"Camera {camera_index} opened but can't read frames")
                        if self.cap:
                            self.cap.release()
                else:
                    print(f"Camera {camera_index} failed to open")
                    if self.cap:
                        self.cap.release()
            except Exception as e:
                print(f"Error with camera {camera_index}: {e}")
                if self.cap:
                    self.cap.release()
        
        print("Failed to initialize any camera")
        return False
    
    def capture_background(self, num_frames=30):
        """Capture stable background using median of multiple frames"""
        if not self.cap or not self.cap.isOpened():
            return False
            
        print("Capturing background... Please step out of the frame!")
        self.is_capturing_background = True
        frames = []
        
        for i in range(num_frames):
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
                print(f"Capturing frame {i+1}/{num_frames}")
        
        if frames:
            # Use median to get stable background (removes noise and temporary objects)
            self.background = np.median(frames, axis=0).astype(np.uint8)
            self.background = cv2.GaussianBlur(self.background, (5, 5), 0)
            self.previous_masks.clear()
            self.is_capturing_background = False
            print("Background captured successfully.")
            return True
        
        self.is_capturing_background = False
        return False
    
    def adaptive_red_mask(self, hsv, frame_count):
        """Get red mask with adaptive parameters for better accuracy"""
        # Multiple red ranges for better coverage [web:75][web:78]
        red_ranges = [
            ([0, 100, 50], [15, 255, 255]),      # Lower red range
            ([160, 100, 50], [180, 255, 255]),   # Upper red range
            ([0, 80, 80], [20, 255, 255]),       # Lighter reds
            ([170, 80, 80], [180, 255, 255])     # Lighter upper reds
        ]
        
        masks = []
        for lower, upper in red_ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            masks.append(mask)
        
        # Combine all masks
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Advanced morphological operations for cleaner mask [web:75][web:76]
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Remove noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        # Fill holes
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        # Smooth edges
        combined_mask = cv2.dilate(combined_mask, kernel_large, iterations=1)
        
        # Gaussian blur for smooth edges
        combined_mask = cv2.GaussianBlur(combined_mask, (9, 9), 0)
        
        return combined_mask
    
    def temporal_smoothing(self, current_mask, alpha=0.7):
        """Smooth mask over time to reduce flickering"""
        if not self.previous_masks:
            return current_mask
        
        # Average with previous masks
        avg_mask = np.mean(self.previous_masks, axis=0).astype(np.uint8)
        smoothed = cv2.addWeighted(current_mask, alpha, avg_mask, 1-alpha, 0)
        return smoothed
    
    def enhance_invisibility_effect(self, frame, mask):
        """Enhanced blending for better invisibility effect [web:75]"""
        if self.background is None:
            return frame
            
        # Convert mask to 3-channel for better blending
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        inv_mask_3ch = 1.0 - mask_3ch
        
        # Use floating point for better precision
        frame_float = frame.astype(np.float64)
        background_float = self.background.astype(np.float64)
        
        # Blend with smooth transitions [web:75]
        # This is the core invisibility effect - replace red areas with background
        result = (frame_float * inv_mask_3ch + background_float * mask_3ch)
        
        return result.astype(np.uint8)
    
    def generate_frames(self):
        """Generate frames for video streaming with full cloak processing"""
        while self.is_streaming and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                self.frame_count += 1
                
                if self.background is not None and not self.is_capturing_background:
                    # Apply invisible cloak effect [web:74][web:75]
                    
                    # Convert to HSV for better color detection [web:78]
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    
                    # Get red mask with adaptive parameters
                    mask_red = self.adaptive_red_mask(hsv, self.frame_count)
                    
                    # Apply temporal smoothing to reduce flickering
                    if len(self.previous_masks) > 0:
                        mask_red = self.temporal_smoothing(mask_red)
                    
                    # Update previous masks for temporal smoothing
                    self.previous_masks.append(mask_red)
                    if len(self.previous_masks) > self.max_previous_masks:
                        self.previous_masks.pop(0)
                    
                    # Create invisibility effect - this is the main cloak logic! [web:75]
                    frame = self.enhance_invisibility_effect(frame, mask_red)
                
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    print("Failed to encode frame")
                
            except Exception as e:
                print(f"Error in generate_frames: {e}")
                break
    
    def cleanup(self):
        """Clean up resources"""
        self.is_streaming = False
        if self.cap:
            self.cap.release()

# Global cloak instance
cloak = InvisibleCloak()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/start_camera')
def start_camera():
    """Initialize and start camera"""
    if cloak.initialize_camera():
        cloak.is_streaming = True
        return jsonify({'status': 'success', 'message': 'Camera started successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to initialize camera. Try Client Camera Mode for web deployment.'})

@app.route('/stop_camera')
def stop_camera():
    """Stop camera"""
    cloak.cleanup()
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

@app.route('/capture_background')
def capture_background():
    """Capture background for invisible cloak effect"""
    if cloak.cap and cloak.cap.isOpened():
        success = cloak.capture_background()
        if success:
            return jsonify({'status': 'success', 'message': 'Background captured successfully! Put on red cloth to become invisible.'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to capture background'})
    else:
        return jsonify({'status': 'error', 'message': 'Camera not initialized'})

@app.route('/reset_smoothing')
def reset_smoothing():
    """Reset temporal smoothing"""
    cloak.previous_masks.clear()
    return jsonify({'status': 'success', 'message': 'Temporal smoothing reset'})

@app.route('/video_feed')
def video_feed():
    """Video streaming route with invisibility effect"""
    if cloak.is_streaming:
        return Response(cloak.generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({'status': 'error', 'message': 'Camera not started'})

@app.route('/debug_camera')
def debug_camera():
    """Debug endpoint to check camera availability"""
    debug_info = []
    
    for i in range(3):
        try:
            cap = cv2.VideoCapture(i)
            is_opened = cap.isOpened()
            debug_info.append(f"Camera {i}: {'Available' if is_opened else 'Not available'}")
            if is_opened:
                ret, frame = cap.read()
                debug_info.append(f"Camera {i} can read frames: {ret}")
            cap.release()
        except Exception as e:
            debug_info.append(f"Camera {i}: Error - {e}")
    
    return jsonify({'debug_info': debug_info})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'message': 'Invisible Cloak App is running!',
        'background_captured': cloak.background is not None,
        'camera_active': cloak.is_streaming
    })

@app.route('/cloak_status')
def cloak_status():
    """Get current cloak processing status"""
    return jsonify({
        'camera_active': cloak.is_streaming,
        'background_captured': cloak.background is not None,
        'frames_processed': cloak.frame_count,
        'masks_in_memory': len(cloak.previous_masks)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
