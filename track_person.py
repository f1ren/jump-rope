import time

import cv2
import numpy as np

from jump_detect import JumpCounter

VIDEO_SOURCE = 0  # Camera input
VIDEO_SOURCE = 'rope_jump_2.mp4'  # File input

BOUNDING_BOX_SCALE_FACTOR = 0.7

GREEN = (0, 255, 0)
RED = (0, 0, 255)


def _show_frame(frame, box, color, jumps):
    if box is not None:
        (x, y, w, h) = map(int, box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(frame, f'{jumps}', (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 4, (36, 255, 12), 2)
    cv2.imshow("Video", frame)


def _scale_box(box, f):
    (x, y, w, h) = box
    return x + int(.5 * w * (1 - f)), y + int(.5 * h * (1 - f)), w * f, h * f


def _smaller_box(box):
    return _scale_box(box, BOUNDING_BOX_SCALE_FACTOR)


def _bigger_box(box):
    return _scale_box(box, BOUNDING_BOX_SCALE_FACTOR)


def _init_tracker_and_box(cnts, frame, weights):
    box = cnts[np.argmax(weights)]
    box = _smaller_box(box)
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, tuple(box))
    return box, tracker


def _init_tracker_if_person_detected(frame, hog):
    (cnts, weights) = hog.detectMultiScale(
        frame,
        winStride=(4, 4),
        padding=(4, 4),
        scale=1.05)

    if len(cnts) == 0:
        return None, None

    return _init_tracker_and_box(cnts, frame, weights)


def _init_variables():
    # Capturing video
    video = cv2.VideoCapture(VIDEO_SOURCE)
    # Initializing the HOG person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    tracker = None
    jump_counter = JumpCounter()
    return hog, jump_counter, tracker, video


def _update_tracker_get_box(frame, hog, tracker):
    if tracker is None:
        return _init_tracker_if_person_detected(frame, hog)

    (success, box) = tracker.update(frame)
    if not success:
        box = None
        tracker.clear()
        tracker = None

    return box, tracker


def _get_jump_count(box, jump_counter, video):
    if box is None:
        return 0

    if VIDEO_SOURCE == 0:
        timestamp = int(time.time())
    else:
        timestamp = video.get(cv2.CAP_PROP_POS_MSEC)

    return jump_counter.count_jumps(_bigger_box(box), timestamp)


def _q_key_pressed():
    return cv2.waitKey(1) == ord('q')


def _cleanup(video):
    video.release()
    cv2.destroyAllWindows()


def main_loop():
    hog, jump_counter, tracker, video = _init_variables()

    # Infinite while loop to treat stack of image as video
    while True:
        # Reading frame(image) from video
        check, frame = video.read()

        if frame is None:
            break

        box, tracker = _update_tracker_get_box(frame, hog, tracker)
        jumps = _get_jump_count(box, jump_counter, video)
        _show_frame(frame, box, GREEN, jumps)

        if _q_key_pressed():
            break

    _cleanup(video)


if __name__ == '__main__':
    main_loop()
