import cv2
# import imutils
from imutils.video import VideoStream
import dlib
import time
# import argparse
from idcard import is_verified_idnum, is_verified_age

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--idfile", type=str,
#                 help="id card image path")
# args = vars(ap.parse_args())


def identification(detector, predictor):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2)

    while True:
        frame = vs.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_detector = detector(frame_gray, 1)
        # print("[INFO] The number of faces detected : {}".format(len(face_detector)))

        for face in face_detector:
            landmarks = predictor(frame, face)

            landmark_list = []
            for p in landmarks.parts():
                landmark_list.append([p.x, p.y])
                cv2.circle(frame, (p.x, p.y), 2, (0, 255, 0), -1)

            h = landmark_list[8][1] - landmark_list[27][1]
            w = landmark_list[16][0] - landmark_list[0][0]
            (x, y) = landmark_list[33]

            # if 얼굴 움직임 테스트를 만족하면:
            #     label = "Verified"
            #     color = (255, 0, 0)
            #     print("[INFO] you're not picture!")
            #     얼굴 크롭해서 얼굴 인식 모델에 넘겨줌
            #     break
            # else:
            #     label = "Robot"
            #     color = (0, 0, 255)

            cv2.rectangle(frame, (x - w, y + h), (x + w, y - 2 * h),
                          (0, 0, 255), 3)
            # cv2.rectangle(frame, (x - w, y + h), (x + w, y - 2 * h),
            #               color, 3)
            # cv2.putText(frame, label, (x - w, y + h - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    print("[INFO] loading face detector weights...")
    main_detector = dlib.get_frontal_face_detector()
    main_predictor = dlib.shape_predictor("weights/shape_predictor_68_face_landmarks.dat")

    while True:
        file_name = input("[INFO] Type ID card image file path: ")

        if file_name == "q":
            break

        print("[INFO] Verifying ID card...")
        idcard_image = cv2.imread(file_name)

        if is_verified_idnum(idcard_image) and is_verified_age(idcard_image):
            print("[INFO] Your ID card is verified")
            identification(main_detector, main_predictor)

        else:
            print("[ERROR] Invalid ID card!")

