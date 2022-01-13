import sys
import argparse
from yolo_v3 import detect_video, track_cap #, detect_dir_frames
from PIL import Image
import capture_stream


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        "--dir", nargs='?', type=str, default="",
        help = "Directory with images (frames)"
    )

    parser.add_argument(
        "--stream", nargs='?', type=str, required=False, default="",
        help = "Link with a YouTube video streaming"
    )
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default="",
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    
    if FLAGS.input:
        print(FLAGS.input, "INPUT")
        print(FLAGS.output, "OUTPUT")
        detect_video(FLAGS.input, FLAGS.output) 
        #track_cap(FLAGS.input)    
    
    elif FLAGS.stream:
        tempFile = "vidCalle.ts"  #files are format ts, open cv can view them
        videoURL = FLAGS.stream
        print(videoURL)
        capture_stream.dl_stream(videoURL, tempFile, 2000)
        detect_video(tempFile, FLAGS.output)
    
    # elif FLAGS.dir:
    #     print(FLAGS.dir)
    #     detect_dir_frames(YOLO(**vars(FLAGS)), FLAGS.dir, FLAGS.output)

    else:
        print("Must specify at least video_input_path.  See usage with --help.")
