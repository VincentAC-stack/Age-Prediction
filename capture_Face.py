import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

def captureFace(img_original):
    frame = img_original
    img = frame
    face_area_image = img
    faces = face_cascade.detectMultiScale(frame, 1.1, minNeighbors = 3,minSize=(20, 20))
    SAFE_MARGIN_W = 0
    SAFE_MARGIN_H = 0

    for (x, y, w, h) in faces:
        # 画出人脸框，蓝色，画笔宽度微
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 框选出人脸区域，在人脸区域而不是全图中进行人眼检测，节省计算资源
        face_area = img[y:y + h, x:x + w]
        face_area_image = frame[y - int(h * SAFE_MARGIN_H):y + int(h * (SAFE_MARGIN_H + 1)),
                          x - int(w * SAFE_MARGIN_W):x + int(w * (SAFE_MARGIN_W + 1))]

    # cv2.imshow('frameFace', face_area_image)
    # cv2.imshow('frame2Q', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return face_area_image,len(faces),img

#
# img_path='89605_1958-07-06_2014.jpg'
# iaaa=cv2.imread(img_path, 0)
# captureFace(iaaa)

