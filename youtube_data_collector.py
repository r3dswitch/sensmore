import os
import cv2
import numpy as np
import yt_dlp

class YouTubeDataCollector:
    """Collect training data from YouTube videos"""
    
    def __init__(self, output_dir="data/frames"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def download_and_extract_frames(self, youtube_urls, frames_per_video=50):
        """Download videos and extract frames"""
        print("🎬 Downloading and extracting frames from YouTube videos...")
        
        ydl_opts = {
            'format': 'best[height<=720]',  # Keep it lightweight
            'outtmpl': 'temp_video.%(ext)s'
        }
        
        all_frames = []
        
        for i, url in enumerate(youtube_urls):
            try:
                print(f"Processing video {i+1}/{len(youtube_urls)}: {url}")
                
                # Download video
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                # Extract frames
                frames = self._extract_frames_from_video('temp_video.mp4', 
                                                       f"video_{i}", frames_per_video)
                all_frames.extend(frames)
                
                # Cleanup
                if os.path.exists('temp_video.mp4'):
                    os.remove('temp_video.mp4')
                    
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue
        
        print(f"✅ Extracted {len(all_frames)} frames total")
        return all_frames
    
    def _extract_frames_from_video(self, video_path, prefix, max_frames):
        """Extract frames from a single video"""
        if not os.path.exists(video_path):
            return []
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames evenly throughout video
        frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        
        extracted_frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count in frame_indices:
                frame_path = os.path.join(self.output_dir, f"{prefix}_frame_{len(extracted_frames):04d}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(frame_path)
                
            frame_count += 1
        
        cap.release()
        return extracted_frames