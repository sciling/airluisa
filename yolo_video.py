import sys
import argparse
from yolo_v3 import detect_video #, detect_dir_frames
from PIL import Image
#import capture_stream


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser()
    '''
    Command line options
    '''
    parser.add_argument(
        "--dir", nargs='?', type=str, default="",
        help = "Directory with images (frames)"
    )

    parser.add_argument(
        "--stream", action='store_true',
        help = "[Optional] Process a live streaming or not"
    )
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default="",
        help = "Video input path or url"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    parser.add_argument(
        "--sampling-fps", nargs='?', type=int, default="",
        help = "[Optional] This is the rate to use in order to sample a video frames: FRAMES TO SAMPLE = OROGINAL_FPS/sampling-fps == 30/3 = 10. That is, will sample 1 frame each 10"
    )

    parser.add_argument(
        "--use-cuda", action='store_true',
        help = "[Optional] Use cuda or not"
    )


    FLAGS = parser.parse_args()

    
    print(FLAGS.input, "INPUT")
    print(FLAGS.output, "OUTPUT")
    print(FLAGS.sampling_fps, "SAMPLING FPS")
    print(f"Use cuda flag: {FLAGS.use_cuda} ")
    detect_video(video_path=FLAGS.input, output_path=FLAGS.output, use_cuda=FLAGS.use_cuda, smapling_fps=FLAGS.sampling_fps, streaming=FLAGS.stream) 

    if not FLAGS.input:
        print("Must specify at least video_input_path.  See usage with --help.")
