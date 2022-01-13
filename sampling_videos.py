import cv2
import sys

def sampling_video(video_path, output_path, num_frames):
    vid = cv2.VideoCapture(video_path)

    print(vid.isOpened())
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    print("FPS originals: ", video_fps)

    print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
    out = cv2.VideoWriter(output_path, fourcc, video_fps, video_size)

    i=0
    print(video_fps, "/", num_frames, " =")
    num_frames = round(video_fps / num_frames)
    print(num_frames)
    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret == False:
            break
        if i % num_frames == 0: # this is the line I added to make it only save one frame every 20
            #cv2.imwrite('/home/tgonzalez/Desktop/projects/airLUISA/detect_and_track/out/edit_frames/kang'+str(i)+'.jpg',frame)
            out.write(frame)
        i+=1

    vid.release()

if __name__ == '__main__':

    video_path = sys.argv[1]
    output_path = sys.argv[2]
    num_frames = float(sys.argv[3])

    sampling_video(video_path, output_path, num_frames)

