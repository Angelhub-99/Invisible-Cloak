from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
import threading
import base64
from io import BytesIO
from PIL import Image

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
        self.cap = cv2.VideoCapture(0)
        if self.cap is None or not self.cap.isOpened():
            # Try different camera indices
            for i in range(1, 5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    break
        
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            return True
        return False
    
    def capture_background(self, num_frames=30):
        """Capture stable background using median of multiple frames"""
        if not self.cap or not self.cap.isOpened():
            return False
            
        self.is_capturing_background = True
        frames = []
        
        for i in range(num_frames):
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
        
        if frames:
            self.background = np.median(frames, axis=0).astype(np.uint8)
            self.background = cv2.GaussianBlur(self.background, (5, 5), 0)
            self.previous_masks.clear()
            self.is_capturing_background = False
            return True
        
        self.is_capturing_background = False
        return False
    
    def adaptive_red_mask(self, hsv):
        """Get red mask with adaptive parameters"""
        red_ranges = [
            ([0, 100, 50], [15, 255, 255]),
            ([160, 100, 50], [180, 255, 255]),
            ([0, 80, 80], [20, 255, 255]),
            ([170, 80, 80], [180, 255, 255])
        ]
        
        masks = []
        for lower, upper in red_ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            masks.append(mask)
        
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        combined_mask = cv2.dilate(combined_mask, kernel_large, iterations=1)
        combined_mask = cv2.GaussianBlur(combined_mask, (9, 9), 0)
        
        return combined_mask
    
    def temporal_smoothing(self, current_mask, alpha=0.7):
        """Smooth mask over time to reduce flickering"""
        if not self.previous_masks:
            return current_mask
        
        avg_mask = np.mean(self.previous_masks, axis=0).astype(np.uint8)
        smoothed = cv2.addWeighted(current_mask, alpha, avg_mask, 1-alpha, 0)
        return smoothed
    
    def enhance_invisibility_effect(self, frame, mask):
        """Enhanced blending for better invisibility effect"""
        if self.background is None:
            return frame
            
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        inv_mask_3ch = 1.0 - mask_3ch
        
        frame_float = frame.astype(np.float64)
        background_float = self.background.astype(np.float64)
        
        result = (frame_float * inv_mask_3ch + background_float * mask_3ch)
        return result.astype(np.uint8)
    
    def generate_frames(self):
        """Generate frames for video streaming"""
        while self.is_streaming and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            if self.background is not None and not self.is_capturing_background:
                # Apply invisible cloak effect
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask_red = self.adaptive_red_mask(hsv)
                
                if len(self.previous_masks) > 0:
                    mask_red = self.temporal_smoothing(mask_red)
                
                self.previous_masks.append(mask_red)
                if len(self.previous_masks) > self.max_previous_masks:
                    self.previous_masks.pop(0)
                
                frame = self.enhance_invisibility_effect(frame, mask_red)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
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
        return jsonify({'status': 'error', 'message': 'Failed to initialize camera'})

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
            return jsonify({'status': 'success', 'message': 'Background captured successfully'})
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
    """Video streaming route"""
    if cloak.is_streaming:
        return Response(cloak.generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({'status': 'error', 'message': 'Camera not started'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
