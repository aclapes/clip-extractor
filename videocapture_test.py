__author__ = 'aclapes'

import numpy as np
import cv2
import cv
import argparse
from os import listdir, rename
from os.path import splitext, join, isfile
import time
import random
import subprocess
from joblib import delayed, Parallel
from os.path import isfile, exists
from os import makedirs
import cPickle

PARAMETERS = dict(
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml'),
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml'),
    bar_height=15,  # (in pixels)
    fadding=0.25,  # video and audio in-out fadding time in a segment (in seconds)
    seg_effective_duration=15,  # segments' duration without fadding (in seconds)
    greetings_duration=0.5, # hello and goodbye segment duration (seconds)
    video_extension=['.mp4'],
    # face_displacement_threshold=0.75,
    #video_score_threshold=0.33,  # 0-1 range
    segment_score_threshold=0.75,  # 0-1 range
    cuts_per_video=6,
    time_gran_secs=1,
)

def process(external_parameters, nt=1):
    # create segments' output directory if it doesn't exist already
    output_path = external_parameters['segments_dir_path']
    if not exists(output_path):
        makedirs(output_path)
    else:
        if not exists(join(output_path, 'clips/')):
            makedirs(join(output_path, 'clips/'))
        if not exists(join(output_path, 'clip_metadata/')):
            makedirs(join(output_path, 'clip_metadata/'))

    # get a list of files to process
    dirlist = listdir(external_parameters['videos_dir_path'])

    # process all the files!!
    Parallel(n_jobs=nt, backend='threading')(delayed(process_file)(file, external_parameters)
                                                   for file in dirlist)


def process_file(file, external_parameters):
    videos_path = external_parameters['videos_dir_path']
    # discarded_path = external_parameters['discarded_dir_path']
    output_path = external_parameters['segments_dir_path']

    clip_duration = PARAMETERS['seg_effective_duration'] + 2*PARAMETERS['fadding']
    # greet_duration = PARAMETERS['greetings_duration'] + PARAMETERS['fadding']

    # process .mp4 videos
    if splitext(file)[1] not in PARAMETERS['video_extension']:
        return

    input_videofile_path = join(videos_path, file)

    file_parts = splitext(file)
    output_metadata_path = join(output_path, 'clip_metadata/', file_parts[0] + '.pkl')
    if isfile(output_metadata_path):
        print('%s -> OK (already processed)' % file_parts[0])
        return

    video_cap = cv2.VideoCapture(input_videofile_path)
    fps = video_cap.get(cv.CV_CAP_PROP_FPS)

    # proceed normally
    st_total_time = time.time()
    try:
        st_sub_time = time.time()
        clips, remaining_segs = get_random_clips(video_cap, clip_duration, \
                                                 n=PARAMETERS['cuts_per_video'], \
                                                 time_gran_secs=PARAMETERS['time_gran_secs'])
        print('[Segment generation] Time took: %.2f' % (time.time() - st_sub_time))
    except cv2.error:
        print('%s -> ERROR' % input_videofile_path)
        return

    st_sub_time = time.time()
    for i, clip in enumerate(clips):
        # play_video(video_cap, start=seg[0], end=seg[1], frameskip=1, repeat=True, detect_faces=True)
        output_cutfile_path = file_parts[0] + '.' + str(i).zfill(3) + file_parts[1]
        cut_videofile(join(videos_path, input_videofile_path), \
                      join(output_path, 'clips/', output_cutfile_path), \
                      int(clip[0]/fps), clip_duration, \
                      fade_in=PARAMETERS['fadding'], fade_out=PARAMETERS['fadding'])
    print('[Encoding video] Time took: %.2f secs' % (time.time() - st_sub_time))

    # Save beginning-end of segments into pickle
    with open(output_metadata_path, 'wb'):
        cPickle.dump(
            dict(
                clips=clips, \
                remaining_segs=remaining_segs, \
                fps=fps, \
                total_frames=video_cap.get(cv.CV_CAP_PROP_FRAME_COUNT), \
                parameters_dict=PARAMETERS
            )
        )

    print('%s -> DONE (Total time took: %.2f secs)' % (input_videofile_path, time.time()-st_total_time))


def display_mosaic_and_ask_oracle(cap, counts, steps, nx=5, ny=5):
    global small_img
    _, frame_size, _ = get_capture_info(cap)
    side_size = max(frame_size)

    if nx*ny < len(counts):
        sorted_sample = sorted(random.sample(range(len(counts)), nx*ny), key=lambda x:x)
    else:
        sorted_sample = range(len(counts))

    M = np.zeros((360,640,3), dtype=np.uint8)
    h = M.shape[0]/ny
    w = M.shape[1]/nx
    for i in xrange(ny):
        for j in xrange(nx):
            small_img = np.zeros((h,w,3), dtype=np.uint8) + 127
            if i*nx+j < len(sorted_sample):
                idx = sorted_sample[i*nx+j]

                cap.set(cv.CV_CAP_PROP_POS_FRAMES, int(steps[idx]))
                ret, img = cap.retrieve()
                if ret:
                    small_img = cv2.resize(img, (w,h), interpolation = cv2.INTER_CUBIC)
                    thickness = 3
                    cv2.rectangle(small_img, \
                                  (int(thickness/2),int(thickness/2)), \
                                  (small_img.shape[1]-int(thickness/2)-1, small_img.shape[0]-int(thickness/2)-1), \
                                  (0, 255 if counts[idx] == 1 else 0, 0 if counts[idx] == 1 else 255), \
                                  thickness)

            M[i*h:(i+1)*h,j*w:(j+1)*w] = small_img

    ret = 0

    cv2.namedWindow('mosaic')
    cv2.imshow('mosaic', M)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('r'):
            ret = -1
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('mosaic')

    return ret


def moving_average_filter_in_1d_vec(x, kernel_size=5):
    v = np.zeros(len(x), dtype=np.float32)

    acc = np.sum(x[:kernel_size],dtype=np.float32)
    for i in range(kernel_size/2, len(x) - kernel_size/2 - 1):
        v[i] = acc / kernel_size
        acc += x[i+kernel_size/2+1]
        acc -= x[i-kernel_size/2]
    v[i+1] = acc / kernel_size

    return v


def get_random_clips(video_cap, duration, n=3, time_gran_secs=1.0, candidates=None):
    """

    :param video_cap:
    :param duration:
    :param n: number of middle segment cuts per video
    :param time_gran_secs: time granularity in seconds. if 1, face detection is performed every second; if 2, every two seconds, if 0.5, every half second
    :return:
    """
    fps = video_cap.get(cv.CV_CAP_PROP_FPS)
    step = seconds_to_num_frames(duration, fps)
    fskip = time_gran_secs*fps  # that is performing face detection once per second of video
    segment_kernel = int(step/fskip)

    if candidates is None:
        counts, cuts, faces = count_speakers_in_video_cuts(video_cap, frameskip=fskip)

        # Filtering criteria
        # -------
        validness = np.ones((len(counts),), dtype=np.int32)
        # (1) one and only speaker
        validness[counts != 1] = 0
        # -------

        # not probable to find contiguious perfect segments, but good enough segments (as many as possible valid frames)
        validness_avg = moving_average_filter_in_1d_vec(validness,kernel_size=segment_kernel)
        candidates = np.where(validness_avg > PARAMETERS['segment_score_threshold'])[0]

    segs = []
    while len(segs) < n and len(candidates) >= segment_kernel:
        # center_seg = random.sample(candidates, 1)[0]
        idx = candidates[random.randint(0,len(candidates)-1)]
        segs.append(((idx - int(segment_kernel/2)) * fskip, (idx + int(segment_kernel/2) + 1) * fskip))

        mask = (candidates > idx + int(segment_kernel/2)) | (candidates < idx - int(segment_kernel/2))
        candidates = candidates[np.where(mask)[0]]  # update with the remaining candidates

    return segs, candidates


def count_speakers_in_video_cuts(cap, start=-1, end=-1, frameskip=1):
    _, frame_size, nframes = get_capture_info(cap)
    height = frame_size[1]

    if start < 0:
        start = 0
    if start >= end:
        end = nframes

    counts = []
    cuts = []
    faces = []

    r = 1.0 / (height/240)

    cap.set(cv.CV_CAP_PROP_POS_FRAMES, start)
    while (True if end == 0 else cap.get(cv.CV_CAP_PROP_POS_FRAMES) < end):
        fid = int(cap.get(cv.CV_CAP_PROP_POS_FRAMES))
        ret, img = cap.retrieve()
        if not ret:
            break

        faces_f, _ = detect_faces_and_contained_eyes(img, r=r)
        counts.append(len(faces_f))
        cuts.append(fid)
        faces.append(faces_f)

        cap.set(cv.CV_CAP_PROP_POS_FRAMES, fid + frameskip)  # advance forward

    faces_f, _ = detect_faces_and_contained_eyes(img, r=r)
    counts.append(len(faces_f))
    cuts.append(nframes)
    faces.append(faces_f)

    return np.array(counts), np.array(cuts), faces


def detect_faces_and_contained_eyes(img, r=1.0):
    """
    Avoid true positives by not detecting faces that doesn't contain at least one eye.

    Parameters
    ----------
    img : colored input image where the faces must be detected
    r : resizing factor of img (to speed up the detections)

    Return
    ------
    faces, eyes : the faces and eyes detected (and filtered)
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape[:2]
    gray = cv2.resize(gray,(int(r*width), int(r*height)), interpolation = cv2.INTER_CUBIC)

    min_side = height if height < width else width
    # Get an initial set of face hypthesis
    faces_h = PARAMETERS['face_cascade'].detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, \
                                                         minSize=(int(min_side/10),int(min_side/10)), maxSize=(min_side,min_side), \
                                                         flags = cv.CV_HAAR_SCALE_IMAGE) # naive detection

    # Confirm hypothesis detecting eyes within those faces
    faces = []  # final set of faces
    eyes = []
    for face in faces_h:
        x,y,w,h = face
        face_eyes = PARAMETERS['eye_cascade'].detectMultiScale(gray[y:y+h,x:x+w])
        if len(face_eyes) > 0:  # at least one eye detected
            faces.append(face)
            eyes.append(face_eyes)

    return faces, eyes


def play_video(cap, start=-1, end=-1, frameskip=1, repeat=True, detect_faces=True):
    if not cap.isOpened() or start > end:
        return

    fps, frame_size, nframes = get_capture_info(cap)
    height = frame_size[1]

    if start < 0:
        start = 0
    if start >= end:
        end = nframes

    first_play = True
    while first_play or repeat:

        if detect_faces:
            detection_counts = np.zeros((nframes,), dtype=np.int32)  # -1 means we are not checking it
            detection_counts[:start-1] = -1
            detection_counts[end+1:] = -1

        fid = start
        cap.set(cv.CV_CAP_PROP_POS_FRAMES, start)
        while cap.grab() and cap.get(cv.CV_CAP_PROP_POS_FRAMES) < end:
            ret, img = cap.retrieve()

            if detect_faces:
                r=1.0/(height/240)
                faces, eyes = detect_faces_and_contained_eyes(img, r=r)
                detection_counts[fid:fid+frameskip] = len(faces)

                draw_faces_and_eyes(faces, eyes, img, r=r)
                draw_detection_counts(fid, detection_counts, img)

            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                repeat = False
                break

            cap.set(cv.CV_CAP_PROP_POS_FRAMES, cap.get(cv.CV_CAP_PROP_POS_FRAMES) + frameskip - 1)
            fid += frameskip

        first_play = False

    cv2.destroyAllWindows()


# def write_video(cap, filename, start=-1, end=-1, frameskip=1):
#     if not cap.isOpened() or start > end:
#         return
#
#     fourcc = int(cap.get(cv.CV_CAP_PROP_FOURCC))
#     fps = int(cap.get(cv.CV_CAP_PROP_FPS))
#     frame_size = (int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)))
#     writer = cv2.VideoWriter(filename, fourcc, fps, frame_size, isColor=True)
#     if not writer.isOpened():
#         return
#
#     fps, frame_size, nframes = get_capture_info(cap)
#     height = frame_size[1]
#
#     if start < 0:
#         start = 0
#     if start >= end:
#         end = nframes
#
#     fid = start
#     cap.set(cv.CV_CAP_PROP_POS_FRAMES, start)
#     while cap.grab() and cap.get(cv.CV_CAP_PROP_POS_FRAMES) < end:
#         ret, img = cap.retrieve()
#
#         # write img to disk
#         writer.write(img)
#
#         cap.set(cv.CV_CAP_PROP_POS_FRAMES, cap.get(cv.CV_CAP_PROP_POS_FRAMES) + frameskip - 1)
#         fid += frameskip
#
#     writer.release()
#     return


def cut_videofile(input_videofile_path, output_cutfile_path, start, duration, \
                  fade_in=-1, fade_out=-1, verbose=False):
    ''' Use external program (DenseTrack) to extract the features '''
    # argsArray = ['ffmpeg',
    #              '-ss', str(start),  # important to have -ss before -i
    #              '-i', input_videofile_path, \
    #              '-t', str(duration), \
    #              '-vcodec copy -acodec copy -async 1 -copyinkf -y',  # copyinkf to keep non-key frames before the starting (avoid initial freeze)
    #              '' if verbose else '-loglevel quiet', \
    #              output_cutfile_path]  # DenseTrackStab is not accepting parameters, hardcoded the L in there
    # argsArray = ['ffmpeg',
    #          '-i', input_videofile_path, \
    #          '-ss', str(start),  # note -ss in here goes after -i
    #          '-t', str(duration), \
    #          '-vcodec copy -acodec copy -async 1',
    #          '' if verbose else '-loglevel quiet', \
    #          output_cutfile_path]  # DenseTrackStab is not accepting parameters, hardcoded the L in there

    vfade_arg = ''
    afade_arg = ''
    if fade_in > 0:
        afade_arg += 'afade=t=in:st=0:d=' + str(fade_in)
        vfade_arg += 'fade=t=in:st=0:d=' + str(fade_in)
    if fade_out > 0:
        if fade_in > 0:
            vfade_arg += ','
            afade_arg += ','
        vfade_arg += 'fade=t=out:st=' + str(duration-fade_out) + ':d=' + str(fade_out)
        afade_arg += 'afade=t=out:st=' + str(duration-fade_out) + ':d=' + str(fade_out)

    argsArray = ['ffmpeg',
                 '-ss', str(start), \
                 '-i', input_videofile_path, \
                 '-t', str(duration), \
                 "-vf '"+ vfade_arg + "'" if vfade_arg != '' else '', \
                 "-af '"+ afade_arg + "'" if afade_arg != '' else '', \
                 '-y', \
                 '' if verbose else '-loglevel quiet', \
                 output_cutfile_path]  # DenseTrackStab is not accepting parameters, hardcoded the L in there

    cmd = ' '.join(argsArray)
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()


def draw_faces_and_eyes(faces, eyes, img, r=1.0):
    """
    Draw the faces and the corresponding eyes.

    Parameters
    ----------
    faces: faces returned by detect_faces_and_contained_eyes(...)
    eyes: eyes returned by detect_faces_and_contained_eyes(...)
    img: colored image where to draw the faces and eyes
    """

    for i, (x,y,w,h) in enumerate(faces):
        cv2.rectangle(img,(int(x/r),int(y/r)),(int((x+w)/r),int((y+h)/r)),(255,0,0),2)
        roi_color = img[int(y/r):int((y+h)/r), int(x/r):int((x+w)/r)]
        for (ex,ey,ew,eh) in eyes[i]:
            cv2.rectangle(roi_color,(int(ex/r),int(ey/r)),(int((ex+ew)/r),int((ey+eh)/r)),(0,255,0),2)


def draw_detection_counts(ptr, counts, img):
    height, width = img.shape[:2]
    bar = np.zeros((PARAMETERS['bar_height'], len(counts), 3), dtype=np.uint8)

    # draw colors in function of detections
    bar[:,counts == 0] = (255,255,255)  # white if no faces
    bar[:,counts == 1] = (0,192,0)  # green if one and only one face
    bar[:,counts > 1]  = (0,0,192)  # red if more than one

    resized_bar = cv2.resize(bar, (width,PARAMETERS['bar_height']))
    img[:PARAMETERS['bar_height'],:] = resized_bar


def get_capture_info(cap):
    fps =  int(cap.get(cv.CV_CAP_PROP_FPS))
    frame_size = (int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)))
    nframes = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

    return fps, frame_size, nframes

def seconds_to_num_frames(time_seconds, fps=24):
    return fps * time_seconds

if __name__ == "__main__":
    # parse the input arguments
    parser = argparse.ArgumentParser(description='Process the videos to see whether they contain speaking-while-facing-a-camera scenes.')
    parser.add_argument('--videos-dir-path', help='the directory where videos are downloaded.')
    # parser.add_argument('--discarded-dir-path', help='the directory where videos are discarded.')
    parser.add_argument('--segments-dir-path', help='the directory where to output segments.')
    parser.add_argument('--num-threads', help='the directory where to output segments.')
    # parser.add_argument('-O', '--oracle-mode', action='store_true')

    args = parser.parse_args()

    external_parameters = dict()

    external_parameters['videos_dir_path'] ='videos/'
    if args.videos_dir_path:
        external_parameters['videos_dir_path'] = args.videos_dir_path

    # external_parameters['discarded_dir_path'] ='discarded/'
    # if args.discarded_dir_path:
    #     external_parameters['discarded_dir_path'] = args.discarded_dir_path

    external_parameters['segments_dir_path'] ='segments/'
    if args.segments_dir_path:
        external_parameters['segments_dir_path'] = args.segments_dir_path

    nt = 1
    if args.num_threads:
        nt = int(args.num_threads)

    # external_parameters['oracle_mode'] = args.oracle_mode

    # PROCEED downloading videos from the queries
    process(external_parameters=external_parameters, nt=nt)


