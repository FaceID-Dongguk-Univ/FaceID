import dlib
import cv2

# ALL = list(range(0, 68))
# RIGHT_EYEBROW = list(range(17, 22))
# LEFT_EYEBROW = list(range(22, 27))
# RIGHT_EYE = list(range(36, 42))
# LEFT_EYE = list(range(42, 48))
# NOSE = list(range(27, 36))
# MOUTH_OUTLINE = list(range(48, 61))
# MOUTH_INNER = list(range(61, 68))
# JAWLINE = list(range(0, 17))


def crop_face_idcard(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('weights/shape_predictor_68_face_landmarks.dat')

    image = cv2.imread(image_path)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_detector = detector(img_gray, 1)
    print("The number of faces detected : {}".format(len(face_detector)))

    if len(face_detector) == 1:
        face = face_detector[0]

        # cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()),
        #               (0, 0, 255), 3)

        landmarks = predictor(image, face)

        landmark_list = []

        for p in landmarks.parts():
            landmark_list.append([p.x, p.y])
            # cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)

        h = landmark_list[8][1] - landmark_list[27][1]
        w = landmark_list[16][0] - landmark_list[0][0]
        (x, y) = landmark_list[33]

        # cv2.rectangle(image,  (x-w, y+h), (x+w, y-2*h),
        #               (255, 0, 0), 3)

        cropped = image.copy()
        cropped = cropped[y-2*h: y+h, x-w: x+w]

        return cropped

    else:
        print("unvalid")


if __name__ == "__main__":
    result = crop_face_idcard("your image here")
    cv2.imshow("result", result)
    cv2.waitKey(0)
