import os
from datetime import datetime
from wheel_loader_vlm import WheelLoaderVLM

def main():
    
    vlm = WheelLoaderVLM()
    dir = "/Users/soumya/Desktop/Sensmore/data"
    commands = ["Where is the nearest pile?",]
    #             "Which pile should I target first?",
    #             "Where should the loader go next?",
    #             "What is the best pile to dig from?",
    #             "What action should the loader take?",
    #             "How should the loader approach the pile?",
    #             "What's the next step for filling the bucket?",
    #             "How to position for optimal digging?",
    #             "Which direction should the loader move?",
    #             "What's the safest approach path?",
    #             "How to navigate to the material pile?",
    #             "Where to position the loader?"]  
    images = [f for f in os.listdir(dir) if f.lower().endswith(".jpg")]
    out_dir = "/Users/soumya/Desktop/Sensmore/artefacts/"
    command_code = "PTT"
    if not dir:
        print("❌ Please provide a valid directory")
        return
    
    for command in commands:
        for image in images:
            # res = vlm.process_command(dir+"/"+image, command, out_dir+command_code+"_"+datetime.now().strftime('%H-%M-%S')+".png")
            vlm.process_command(dir+"/"+image, command, out_dir+command_code+"_"+image[-8:-4]+".png")

    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()