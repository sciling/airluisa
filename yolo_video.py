import sys
import argparse
from yolo_v3 import detect_video #, detect_dir_frames
from PIL import Image
import capture_stream


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

    parser.add_argument(
        "--use-cuda", action='store_true',
        help = "[Optional] Use cuda or not"
    )


    FLAGS = parser.parse_args()

    
    if FLAGS.input:
        print(FLAGS.input, "INPUT")
        print(FLAGS.output, "OUTPUT")
        print(f"Use cuda flag: {FLAGS.use_cuda} ")
        detect_video(video_path=FLAGS.input, output_path=FLAGS.output, use_cuda=FLAGS.use_cuda) 
        #track_cap(FLAGS.input)    
    
    elif FLAGS.stream:
        tempFile = "vidCalle.ts"  #files are format ts, open cv can view them
        videoURL = FLAGS.stream
        print(videoURL)
        capture_stream.dl_stream(videoURL, tempFile, 2000)
        detect_video(video_path=tempFile, output_path=FLAGS.output, use_cuda=FLAGS.use_cuda)
    
    # elif FLAGS.dir:
    #     print(FLAGS.dir)
    #     detect_dir_frames(YOLO(**vars(FLAGS)), FLAGS.dir, FLAGS.output)

    else:
        print("Must specify at least video_input_path.  See usage with --help.")
