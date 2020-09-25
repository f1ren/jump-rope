import cv2
import numpy as np

from jump_detect import JumpCounter

VIDEO_SOURCE = 0  # Camera input
# VIDEO_SOURCE = 'rope_jump_1.mp4'  # File input

BOUNDING_BOX_SCALE_FACTOR = 0.7

GREEN = (0, 255, 0)
RED = (0, 0, 255)


def show_frame(frame, box, color, jumps):
    if box is not None:
        (x, y, w, h) = map(int, box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(frame, f'{jumps}', (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 4, (36, 255, 12), 2)
    cv2.imshow("Video", frame)


def scale_box(box, f):
    (x, y, w, h) = box
    return x + int(.5 * w * (1 - f)), y + int(.5 * h * (1 - f)), w * f, h * f


def smaller_box(box):
    return scale_box(box, BOUNDING_BOX_SCALE_FACTOR)


def bigger_box(box):
    return scale_box(box, BOUNDING_BOX_SCALE_FACTOR)


def _init_tracker_if_person_detected(frame, hog):
    (cnts, weights) = hog.detectMultiScale(
        frame,
        winStride=(4, 4),
        padding=(4, 4),
        scale=1.05)

    if len(cnts) > 0:
        box = cnts[np.argmax(weights)]
        box = smaller_box(box)
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, tuple(box))
    else:
        box = None
        tracker = None

    return box, tracker


def main_loop():
    # Capturing video
    video = cv2.VideoCapture(VIDEO_SOURCE)

    # Initializing the HOG person
    # detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    tracker = None
    jump_counter = JumpCounter()

    # Infinite while loop to treat stack of image as video
    while True:
        # Reading frame(image) from video
        check, frame = video.read()

        if frame is None:
            break

        if tracker is None:
            # Finding contour of person
            box, tracker = _init_tracker_if_person_detected(frame, hog)
        else:
            (success, box) = tracker.update(frame)
            if not success:
                box = None
                tracker.clear()
                tracker = None

        color = GREEN
        if box is not None:
            timestamp = video.get(cv2.CAP_PROP_POS_MSEC)
            jumps = jump_counter.count_jumps(bigger_box(box), timestamp)
        else:
            box = None
            jumps = 0
        show_frame(frame, box, color, jumps)

        key = cv2.waitKey(1)
        # if q entered whole process will stop
        if key == ord('q'):
            # if something is moving then it append the end time of movement
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_loop()
