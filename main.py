import numpy as np
import tensorflow as tf
import cv2
import dlib
import time
from imutils import face_utils
from imutils.video import VideoStream

from networks.recognition.models import ArcFaceModel
from utils import (
    is_verified_idnum, is_verified_age, crop_face_from_id, eye_aspect_ratio, mouth_aspect_ratio, l2_norm
)

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--idfile", type=str,
#                 help="id card image path")
# args = vars(ap.parse_args())


def human_test(is_human, task, landmarks, img, counter):
    # --- human test parameters ---
    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    # eye_counter = 0
    # mouth_counter = 0
    # smile_counter = 0
    # is_eyes_closed = False
    # is_mouth_open = False
    # is_smiling = False
    # is_human = False
    eye_ar_thresh = 0.3
    mouth_ar_thresh = 0.6
    smile_ar_thresh = 0.2
    frame_ar_thresh = 20

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
            counter += 1
            cv2.putText(img, "EYES are closed", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if counter >= frame_ar_thresh:
                is_human = True
        else:
            eye_counter = 0
            # is_human = False

    elif task == 2:
        in_mouth = landmarks[mouth_start: mouth_end]
        mar = mouth_aspect_ratio(in_mouth)
        # print("mar: ", mar)

        if mar > mouth_ar_thresh:
            counter += 1
            cv2.putText(img, "MOUTH is open", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if counter >= frame_ar_thresh:
                is_human = True
        else:
            counter = 0
            # is_human = False

    else:
        in_mouth = landmarks[mouth_start: mouth_end]
        mar = mouth_aspect_ratio(in_mouth)
        # print("mar: ", mar)

        if mar < smile_ar_thresh:
            counter += 1
            cv2.putText(img, "You're SMILING", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if counter >= frame_ar_thresh:
                is_human = True
        else:
            counter = 0
            # is_human = False

    return is_human, img, counter


def get_embeddings(recognition_model, img):
    img = cv2.resize(img, (112, 112))
    img = img / 255.
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    embeds = l2_norm(recognition_model(img))
    return embeds


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def main(detector, predictor, recognition_model, sr_id_cropped):

    # # --- human test parameters ---
    # (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # (mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    # eye_counter = 0
    # mouth_counter = 0
    # smile_counter = 0
    human_counter = 0
    # # is_eyes_closed = False
    # # is_mouth_open = False
    # # is_smiling = False
    is_human = False
    # eye_ar_thresh = 0.3
    # mouth_ar_thresh = 0.6
    # smile_ar_thresh = 0.2
    # frame_ar_thresh = 20
    task = np.random.randint(1, 4)
    recog_thresh = 0.4

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
            cv2.rectangle(frame, (x-w//2, y+h//2), (x+w//2, y-h//2), (0, 0, 255), 3)

            # cropped = cropped.copy()
            cropped = cropped[y - h//2: y + h//2, x - w//2: x + w//2]
            # print(cropped.shape)

            # # --- human test ---
            # landmarks = face_utils.shape_to_np(landmarks)
            #
            # if task == 1:
            #     left_eye = landmarks[l_start: l_end]
            #     right_eye = landmarks[r_start: r_end]
            #     left_ear = eye_aspect_ratio(left_eye)
            #     right_ear = eye_aspect_ratio(right_eye)
            #     ear = (left_ear + right_ear) / 2.0
            #     # print("ear: ", ear)
            #
            #     if ear < eye_ar_thresh:
            #         eye_counter += 1
            #         cv2.putText(frame, "EYES are closed", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #         if eye_counter >= frame_ar_thresh:
            #             is_human = True
            #     else:
            #         eye_counter = 0
            #         # is_human = False
            #
            # elif task == 2:
            #     in_mouth = landmarks[mouth_start: mouth_end]
            #     mar = mouth_aspect_ratio(in_mouth)
            #     # print("mar: ", mar)
            #
            #     if mar > mouth_ar_thresh:
            #         mouth_counter += 1
            #         cv2.putText(frame, "MOUTH is open", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            #         if mouth_counter >= frame_ar_thresh:
            #             is_human = True
            #     else:
            #         mouth_counter = 0
            #         # is_human = False
            #
            # else:
            #     in_mouth = landmarks[mouth_start: mouth_end]
            #     mar = mouth_aspect_ratio(in_mouth)
            #     # print("mar: ", mar)
            #
            #     if mar < smile_ar_thresh:
            #         smile_counter += 1
            #         cv2.putText(frame, "You're SMILING", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #         if smile_counter >= frame_ar_thresh:
            #             is_human = True
            #     else:
            #         smile_counter = 0
            #         # is_human = False
            is_human, frame, human_counter = human_test(is_human, task, landmarks, frame, human_counter)

            if is_human:
                cv2.putText(frame, "You're Human, thank you", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # is_human = is_eyes_closed or is_mouth_open or is_smiling
            # print(is_human)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)

        if (key == ord("s") and is_human and face is not None) or key == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()

    # --- face recognition ---
    if is_human:
        id_embeds = get_embeddings(recognition_model, sr_id_cropped)
        cam_embeds = get_embeddings(recognition_model, cropped)
        embeds_similarity = cosine_similarity(id_embeds[0], cam_embeds[0])
        print(f"[INFO] Similarity: {embeds_similarity:.4f}")

        if embeds_similarity > recog_thresh:
            print("[INFO] Thank you")
        else:
            print("[INFO] Try again")

    else:
        print("[INFO] Try again")


if __name__ == "__main__":
    # LR_SHAPE = (100 // 4, 100 // 4, 3)
    # INPUT_SHAPE = (100, 100, 3)
    # N_CLASSES = 5749

    print("[INFO] loading face detector weights...")
    main_detector = dlib.get_frontal_face_detector()
    main_predictor = dlib.shape_predictor("weights/shape_predictor_68_face_landmarks.dat")

    print("[INFO] loading face recognition weights...")
    arcface = ArcFaceModel(size=112, backbone_type='ResNet50', training=False)
    ckpt_path = tf.train.latest_checkpoint('./weights/arc_res50_ccrop')
    arcface.load_weights(ckpt_path)#, options=tf.train.Checkpoint())

    # resolution_model =
    print("[INFO] System ready!")

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
            main(main_detector, main_predictor, arcface, id_image)

        else:
            print("[INFO] Invalid ID card!")

