import ffmpeg

(
    ffmpeg
    .input('/home/tgonzalez/Desktop/projects/airLUISA/data/montaje_video/*.jpg', pattern_type='glob', framerate=30)
    .output('/home/tgonzalez/Desktop/projects/airLUISA/data/movie.mp4')
    .run()
)