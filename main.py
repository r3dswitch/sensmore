import os
import json
import argparse
from youtube_data_collector import YouTubeDataCollector
from wheel_loader_vlm import WheelLoaderVLM

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Wheel Loader Vision-Language-Action Model")
    parser.add_argument("--mode", choices=["collect", "train", "demo", "interactive"], 
                       default="demo", help="Operation mode")
    parser.add_argument("--image", type=str, help="Single image path for demo")
    parser.add_argument("--command", type=str, default="Fill the bucket", 
                       help="Command for demo mode")
    parser.add_argument("--output-dir", type=str, default="data", 
                       help="Output directory for data")
    
    args = parser.parse_args()
    
    # YouTube URLs from the prompt
    youtube_urls = [
        "https://www.youtube.com/watch?v=o5LxOWSQSIk",
        # "https://www.youtube.com/watch?v=hWp9vZ7eeaM", 
        # "https://www.youtube.com/watch?v=PnFhMHbcL44",
        # "https://www.youtube.com/watch?v=s7K23lRaLwA",
        # "https://www.youtube.com/watch?v=wc72kf9DWaY",
        # "https://www.youtube.com/watch?v=6Ce7CiQ9yk8"
    ]
    
    vlm = WheelLoaderVLM()
    
    if args.mode == "collect":
        print("🎬 Data Collection Mode")
        collector = YouTubeDataCollector(args.output_dir)
        frames = collector.download_and_extract_frames(youtube_urls)
        print(f"✅ Collected {len(frames)} frames")
        
    elif args.mode == "train":
        print("🏋️ Training Dataset Creation Mode")
        
        # Get all image files from data directory
        data_dir = args.output_dir
        image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            import glob
            image_files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        
        if not image_files:
            print("❌ No images found! Run --mode collect first")
            return
        
        # Create training dataset
        dataset = vlm.create_training_dataset(image_files)
        
        # Save dataset
        dataset_path = os.path.join(args.output_dir, "training_dataset.json")
        vlm.save_dataset(dataset, dataset_path)
        
        print(f"📊 Dataset Statistics:")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Unique images: {len(set(item['image'] for item in dataset))}")
        
    elif args.mode == "demo":
        print("🎯 Demo Mode")
        
        if not args.image:
            print("❌ Please provide --image path for demo mode")
            return
            
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
        
        result = vlm.process_command(args.image, args.command, "res.png")
        
        print("\n" + "="*50)
        print("🤖 WHEEL LOADER VLM DEMO RESULTS")
        print("="*50)
        
        print(f"\n📷 Image: {result['image_path']}")
        print(f"💬 Command: {result['input_command']}")
        
        print(f"\n🔍 Detection Results:")
        if result['detected_piles']:
            for i, pile in enumerate(result['detected_piles']):
                print(f"   Pile {i+1}: {pile['label']}")
                print(f"      Position: ({pile['position'][0]:.0f}, {pile['position'][1]:.0f})")
                print(f"      Confidence: {pile['confidence']:.2f}")
                print(f"      Distance: {pile['distance']:.0f}px")
        else:
            print("   No piles detected")
        
        print(f"\n🧠 LLM Response:")
        print(f"   {result['llm_response']}")
        
        print(f"\n⚡ Action Commands:")
        for cmd in result['action_commands']:
            print(f"   {cmd}")
        
        print("\n" + "="*50)
        
    elif args.mode == "interactive":
        vlm.demo_interactive()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()