import sys
import argparse
from yolo_v2 import YOLO, detect_video, detect_dir_frames
from PIL import Image
import capture_stream

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true", #directorio imágenes
        help='Image detection mode, will ignore all positional arguments'
    )

    parser.add_argument(
        "--dir", nargs='?', type=str, default="",
        help = "Directory with images (frames)"
    )

    parser.add_argument(
        "--stream", nargs='?', type=str, required=False, default="",
        help = "Link with a YouTube video streaming"
    )

    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default="",
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )


    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))

    elif FLAGS.dir:
        print(FLAGS.dir)
        detect_dir_frames(YOLO(**vars(FLAGS)), FLAGS.dir, FLAGS.output)
    
    elif FLAGS.input:
        print(FLAGS.input)
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)     
    
    elif FLAGS.stream:
        tempFile = "temp.ts"  #files are format ts, open cv can view them
        videoURL = FLAGS.stream
        print(videoURL)
        capture_stream.dl_stream(videoURL, tempFile, 800)
        detect_video(YOLO(**vars(FLAGS)), tempFile, FLAGS.output)

    else:
        print("Must specify at least video_input_path.  See usage with --help.")
