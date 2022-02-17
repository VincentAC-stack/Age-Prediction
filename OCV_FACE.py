# import cv2
# img = cv2.imread('image1.jpg',1)
# face_engine = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
# faces = face_engine.detectMultiScale(img,scaleFactor=1.3,minNeighbors=5)
# for (x,y,w,h) in faces:
#     img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import time
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # haarcascade_frontalface_alt2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
# nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

# 调用摄像头摄像头
cap = cv2.VideoCapture(0)

SAFE_MARGIN_W = 0
SAFE_MARGIN_H = 0

ret,frame = cap.read()
# faces = face_cascade.detectMultiScale(frame,1.1,5,cv2.CASCADE_SCALE_IMAGE,(50,50),(100,100))
img = frame
face_area_image = img

while (True):
    startRunTimePrint = time.time()
    # 获取摄像头拍摄到的画面
    ret, frame = cap.read()
    frame = img = cv2.flip(frame, 1, dst=None)
    '''
    faces = face_cascade.detectMultiScale(img, 1.3, minNeighbors = 5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 框选出人脸区域，在人脸区域而不是全图中进行人眼检测，节省计算资源
        face_area = img[y:y + h, x:x + w]
        face_area_image = frame[y-int(h*SAFE_MARGIN_H):y + int(h*(SAFE_MARGIN_H+1)), x-int(w*SAFE_MARGIN_W):x + int(w*(SAFE_MARGIN_W+1))]
        # face_area_image = cv2.rectangle(face_area_image, ( int(w/2)-2,  int(h/2)-2), ( int(w/2)+2,  int(h/2)+2), (255, 255, 0), 2)

        ## 人眼检测
        # 用人眼级联分类器引擎在人脸区域进行人眼识别，返回的eyes为眼睛坐标列表
        # noses = nose_cascade.detectMultiScale(face_area,1.3,20)
        # for (ex,ey,ew,eh) in noses:
        #     # 画出人眼框，绿色，画笔宽度为1
        #     cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)

        # ## 人眼检测
        # # 用人眼级联分类器引擎在人脸区域进行人眼识别，返回的eyes为眼睛坐标列表
        # eyes = eye_cascade.detectMultiScale(face_area,1.3,20)
        # for (ex,ey,ew,eh) in eyes:
        #     # 画出人眼框，绿色，画笔宽度为1
        #     cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
        #
        # ## 微笑检测
        # # 用微笑级联分类器引擎在人脸区域进行人眼识别，返回的eyes为眼睛坐标列表
        # smiles = smile_cascade.detectMultiScale(face_area,scaleFactor= 1.16,minNeighbors=65,minSize=(25, 25),flags=cv2.CASCADE_SCALE_IMAGE)
        # for (ex,ey,ew,eh) in smiles:
        #     # 画出微笑框，红色（BGR色彩体系），画笔宽度为1
        #     cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,0,255),1)
        #     cv2.putText(img,'Smile',(x,y-7), 3, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
'''
    # 每5毫秒监听一次键盘动作
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # 实时展示效果画面
    # cv2.imshow('frameFace', face_area_image)
    finishRunTimePrint = time.time()
    cv2.putText(img, 'T='+(str(int((finishRunTimePrint - startRunTimePrint)*1000)))+'ms', (0, 20), 1, 1, (0, 0, 255))
    cv2.imshow('frame2Q', img)


# 最后，关闭所有窗口
cap.release()
cv2.destroyAllWindows()
