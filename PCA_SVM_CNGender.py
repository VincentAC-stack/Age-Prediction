import numpy as np
import random,os,cv2,glob
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import time
import pywt
from sklearn.svm import  SVC
import joblib
from capture_Face import captureFace

import matplotlib.pyplot as plt
batch_size = 2
def loadImageSet(folder='.\Faces_gender'): #加载图像集，随机选择sampleCount张图片用于训练
    trainData = []; testData = [];yTrain2=[];yTest2=[]

    for k in range(batch_size):
        yTrain1 = [k]
        yTest1 =[k]
        folder2 = os.path.join(folder, 's%d' % (k))
        print(folder2)

        #data 每次10*112*92
        data = [ cv2.imread(d,0) for d in glob.glob(os.path.join(folder2, '*.jpg'))]
        print(str(len(data)))#cv2.imread()第二个参数为0的时候读入为灰度图，即使原图是彩色图也会转成灰度图#glob.glob匹配所有的符合条件的文件，并将其以list的形式返回
        data_pwt=[]
        every_simple_size_renew=0
        # -------------------------------------------------------------------
        for i in range(len(data)):
            tempimg, face_lenth, img_full = captureFace(data[i])
            if face_lenth > 0:
                item = cv2.resize(tempimg, (200, 200))
                # cv2.imshow('frame2Q', item)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                xTrain_pwt = pywt.wavedec2(item, 'sym2', mode='symmetric', level=4, axes=(-2, -1))
                data_pwt.append(xTrain_pwt[0])
            else:
                continue

        # print(data_pwt)
        every_simple_size_renew=len(data_pwt)
        print('-------------face_cut completed--------------' + str(every_simple_size_renew))
        sampleCount = int(every_simple_size_renew * 0.7)
        sample = random.sample(range(every_simple_size_renew), sampleCount)#random.sample()可以从指定的序列中，随机的截取指定长度的片断，不作原地修改

        trainData.extend([data_pwt[i].ravel() for i in range(every_simple_size_renew) if i in sample])#ravel将多维数组降位一维####40*5*112*92
        testData.extend([data_pwt[i].ravel() for i in range(every_simple_size_renew) if i not in sample])#40*5*112*92

        yTrain = np.matrix(yTrain1)
        yTrain= np.tile(yTrain1,sampleCount)#   np.tile(a,(2,1))就是把a先沿x轴（就这样称呼吧）复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍
        yTest=np.tile(yTest1,every_simple_size_renew-sampleCount)#沿着x轴复制5倍，增加列数
        yTrain2.extend(yTrain)
        yTest2.extend(yTest)
    return np.array(trainData),  np.array(yTrain2), np.array(testData), np.array(yTest2)

def main():
    #loadImageSet()
    xTrain_, yTrain, xTest_, yTest = loadImageSet()# 200*10304
    pca1=PCA(n_components=0.9)
    #hyp.plot(xTrain_,'o')
    xTrain_pca=pca1.fit_transform(xTrain_) # 把原始训练集映射到主成分组成的子空间中
    xTest_pca=pca1.transform(xTest_)# 把原始测试集映射到主成分组成的子空间中
    joblib.dump(pca1, 'pca_train_CNGender_model.m')

    # hyp.plot(xTrain_pca, 'o', n_clusters=40)
    clf=SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='poly',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
    #hyp.plot(xTrain_pca, 'o')
    clf.fit(xTrain_pca,yTrain)
    predict=clf.predict(xTest_pca)
    # hyp.plot(predict.reshape(200,1),'o', n_clusters=40)
    # hyp.plot(yTest.reshape(200,1), 'o', n_clusters=40)
    print(clf.score(xTest_pca,yTest))
    print(u'支持向量机识别率: %.2f%%' % ((predict == np.array(yTest)).mean()*100)  )

    print('------------------------')

    joblib.dump(clf, 'svm_train_CNGender_model.m')


if __name__ == '__main__':
    # main()
    pca_m=joblib.load('pca_train_CNGender_model.m')
    svm_m=joblib.load('svm_train_CNGender_model.m')
    cap = cv2.VideoCapture(0)

    while (True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1, dst=None)
        tempimg, face_lenth, img_cap = captureFace(frame)
        if face_lenth > 0:
            item = cv2.resize(tempimg, (200, 200))
            item = cv2.cvtColor(item, cv2.COLOR_RGB2GRAY)
            # cv2.imshow('frame2Q', item)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            xTrain_pwtpp = pywt.wavedec2(item, 'sym2', mode='symmetric', level=4, axes=(-2, -1))
            xtrain_data = xTrain_pwtpp[0]
            predictdata = []
            predictdata.extend(xtrain_data.ravel())
            test_pac = pca_m.transform(np.array(predictdata).reshape(1, -1))
            predict2 = svm_m.predict(test_pac)
            outcome = ''
            if (predict2 == 0):
                outcome = 'Female'
            elif (predict2 == 1):
                outcome = 'Male'
            cv2.putText(frame, outcome, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
            cv2.imshow('frame2Q', frame)
        else:
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
