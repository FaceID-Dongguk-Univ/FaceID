import re
import time

import pytesseract as tess
import dlib
import cv2
import numpy as np

# ALL = list(range(0, 68))
# RIGHT_EYEBROW = list(range(17, 22))
# LEFT_EYEBROW = list(range(22, 27))
# RIGHT_EYE = list(range(36, 42))
# LEFT_EYE = list(range(42, 48))
# NOSE = list(range(27, 36))
# MOUTH_OUTLINE = list(range(48, 61))
# MOUTH_INNER = list(range(61, 68))
# JAWLINE = list(range(0, 17))


def crop_face_idcard(cv_image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('weights/shape_predictor_68_face_landmarks.dat')

    img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    face_detector = detector(img_gray, 1)
    print("The number of faces detected : {}".format(len(face_detector)))

    if len(face_detector) == 1:
        face = face_detector[0]

        # cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()),
        #               (0, 0, 255), 3)

        landmarks = predictor(cv_image, face)

        landmark_list = []

        for p in landmarks.parts():
            landmark_list.append([p.x, p.y])
            # cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)

        h = landmark_list[8][1] - landmark_list[27][1]
        w = landmark_list[16][0] - landmark_list[0][0]
        (x, y) = landmark_list[33]

        # cv2.rectangle(image,  (x-w, y+h), (x+w, y-2*h),
        #               (255, 0, 0), 3)

        cropped = cv_image.copy()
        cropped = cropped[y-2*h: y+h, x-w: x+w]

        return cropped

    else:
        print("invalid")


def get_idnum(cv_image):
    try:
        tess.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        config = ("-l kor+eng --oem 3 --psm 4 -c preserve_interword_spaces=1")
        text = tess.image_to_string(cv_image, config=config)
        text = re.compile(r"\d{6}-\d{7}").search(text).group()
        text = text.replace('-', '')
            
        return text
    
    except Exception as e:
        print("[INFO] Cannot find ID card")


def is_verified_idnum(image):
    id_num = get_idnum(image)

    weights = np.array([2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5])
    id_num_array = np.array(list(id_num[:-1]), np.int32)

    if int(id_num[-1]) == (11 - np.dot(weights, id_num_array.transpose()) % 11) % 11:
        return True
    
    else:
        return False


def is_verified_age(image):
    id_num = get_idnum(image)
    year_curr = int(time.strftime('%Y', time.localtime(time.time())))
    if id_num[6] == '1' or id_num[6] == '2':
        year_birth = int(id_num[:1]) + 1900
    else:
        year_birth = int(id_num[:1]) + 2000

    age = year_curr - year_birth
    if age >= 19:
        return True
    else:
        return False


if __name__ == "__main__":
    # image = cv2.imread("your idcard here")
    image = cv2.imread("resource/idcard.JPG")

    # get ID number
    id_num = get_idnum(image)
    print(id_num)

    # check ID number
    print(is_verified_idnum(image))

    # check age
    print(is_verified_age(image))

    # crop face from ID card
    result = crop_face_idcard(image)
    cv2.imshow("result", result)
    cv2.waitKey(0)
