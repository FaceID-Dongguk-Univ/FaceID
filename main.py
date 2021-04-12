import numpy as np
import tensorflow as tf
import cv2
import dlib
import time
from imutils import face_utils
from imutils.video import VideoStream

from networks.recognition.resnet50_arcface import resnet50_arcface
from utils import is_verified_idnum, is_verified_age, crop_face_from_id, eye_aspect_ratio, mouth_aspect_ratio

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--idfile", type=str,
#                 help="id card image path")
# args = vars(ap.parse_args())


def main(detector, predictor):

    # --- human test parameters ---
    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    eye_counter = 0
    mouth_counter = 0
    smile_counter = 0
    # is_eyes_closed = False
    # is_mouth_open = False
    # is_smiling = False
    is_human = False
    eye_ar_thresh = 0.3
    mouth_ar_thresh = 0.6
    smile_ar_thresh = 0.2
    frame_ar_thresh = 20
    task = np.random.randint(1, 4)

    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2)

    if task == 1:
        print("[INFO] *** Please close your eyes ***")
    elif task == 2:
        print("[INFO] *** Please open your mouth ***")
    else:
        print("[INFO] *** Please smile for me ***")
    print("*** If you passed \"Human Test\", press \'s\' button and take a picture ***")

    while True:
        frame = vs.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detector = detector(frame_gray, 1)

        cropped = frame.copy()

        for face in face_detector:
            landmarks = predictor(frame, face)

            # --- draw 68 face landmarks ---
            landmark_list = []
            for p in landmarks.parts():
                landmark_list.append([p.x, p.y])
                cv2.circle(frame, (p.x, p.y), 2, (0, 255, 0), -1)

            h = landmark_list[8][1] - landmark_list[27][1]
            w = landmark_list[16][0] - landmark_list[0][0]
            (x, y) = landmark_list[33]
            cv2.rectangle(frame, (x-w, y+h), (x+w, y-h), (0, 0, 255), 3)

            # cropped = cropped.copy()
            cropped = cropped[y - h: y + h, x - w: x + w]
            # print(cropped.shape)

            # --- human test ---
            landmarks = face_utils.shape_to_np(landmarks)

            if task == 1:
                left_eye = landmarks[l_start: l_end]
                right_eye = landmarks[r_start: r_end]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                # print("ear: ", ear)

                if ear < eye_ar_thresh:
                    eye_counter += 1
                    cv2.putText(frame, "EYES are closed", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if eye_counter >= frame_ar_thresh:
                        is_human = True
                else:
                    eye_counter = 0
                    # is_human = False

            elif task == 2:
                in_mouth = landmarks[mouth_start: mouth_end]
                mar = mouth_aspect_ratio(in_mouth)
                # print("mar: ", mar)

                if mar > mouth_ar_thresh:
                    mouth_counter += 1
                    cv2.putText(frame, "MOUTH is open", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if mouth_counter >= frame_ar_thresh:
                        is_human = True
                else:
                    mouth_counter = 0
                    # is_human = False

            else:
                in_mouth = landmarks[mouth_start: mouth_end]
                mar = mouth_aspect_ratio(in_mouth)
                # print("mar: ", mar)

                if mar < smile_ar_thresh:
                    smile_counter += 1
                    cv2.putText(frame, "You're SMILING", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if smile_counter >= frame_ar_thresh:
                        is_human = True
                else:
                    smile_counter = 0
                    # is_human = False

            if is_human:
                cv2.putText(frame, "You're Human, thank you", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # is_human = is_eyes_closed or is_mouth_open or is_smiling
            # print(is_human)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)

        if (key == ord("s") and is_human) or key == ord('q'):
            break

    # --- face recognition ---
    # if is_human:
    #     is_valid = recognition_model(cropped, sr_id_cropped)

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    LR_SHAPE = (100 // 4, 100 // 4, 3)
    INPUT_SHAPE = (100, 100, 3)
    N_CLASSES = 5749

    print("[INFO] loading face detector weights...")
    main_detector = dlib.get_frontal_face_detector()
    main_predictor = dlib.shape_predictor("weights/shape_predictor_68_face_landmarks.dat")
    # arcface = resnet50_arcface(INPUT_SHAPE, N_CLASSES)
    # arcface = tf.keras.models.load_model("./weights/resnet50_arcface_epochs50.hdf5")
    # arcface = tf.keras.Model(inputs=arcface.input, outputs=arcface.layers[-3].output)
    # resolution_model =

    while True:
        file_name = input("[INFO] Type ID card image file path: ")

        if file_name == "q":
            break

        print("[INFO] Verifying ID card...")
        id_image = cv2.imread(file_name)

        if is_verified_idnum(id_image) and is_verified_age(id_image):
            # sr_id_cropped = crop_face_from_id(id_image)
            # sr_id_cropped = cv2.resize(sr_id_cropped, LR_SHAPE)
            # sr_id_cropped = resolution_model(sr_id_cropped)

            print("[INFO] Your ID card is verified")
            main(main_detector, main_predictor)

        else:
            print("[INFO] Invalid ID card!")

