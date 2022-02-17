import numpy as np
import random, os, cv2
from sklearn.decomposition import PCA
import time
import pywt
from sklearn.svm import SVC
import joblib
import load_lable
from capture_Face import captureFace
import matplotlib.pyplot as plt


def loadImageSet(folder='.\wiki_crop'): #加载图像集，随机选择sampleCount张图片用于训练
    trainData = [];
    testData = [];
    gender_Train=[];
    gender_Test=[]
    age_full,gender_full,path_full,sample_size=load_lable.load_lable(35000)
    address1st=[]
    for path in path_full :
        path_cata=path[0:2]
        path_filename=path[3:]
        address1st.append(os.path.join(folder, path_cata,path_filename))

    address=[]
    age_2rd=[]
    gender_2rd=[]
    for iaddress in range(len(address1st)):
        if os.path.exists(address1st[iaddress]):
            address.append(address1st[iaddress])
            age_2rd.append(age_full[iaddress])
            gender_2rd.append(gender_full[iaddress])
        else:
            continue


    print('-------address_validiation completed----------'+str(len(address)))
    data = [cv2.imread(d, 0) for d in address]
    age_full_renew=[]
    gender_full_renew=[]
    data_pwt = []
    for m in range(len(address)):
        tempimg,face_lenth,img_full=captureFace(data[m])
        if face_lenth>0:
            item=cv2.resize(tempimg,(200,200))
            # cv2.imshow('frame2Q', item)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            xTrain_pwt = pywt.wavedec2(item, 'sym2', mode='symmetric', level=4, axes=(-2, -1))
            data_pwt.append(xTrain_pwt[0])
            age_full_renew.append(age_2rd[m])
            gender_full_renew.append(gender_2rd[m])
        else:
            continue
    simple_size_renew=len(data_pwt)
    print('-------------face_cut completed--------------'+str(simple_size_renew))
    sampleCount=int(simple_size_renew*0.7)
    sample = random.sample(range(simple_size_renew), sampleCount)  # random.sample()可以从指定的序列中，随机的截取指定长度的片断，不作原地修改
    trainData.extend([data_pwt[i].ravel() for i in range(simple_size_renew) if i in sample])  # ravel将多维数组降位一维####40*5*112*92
    testData.extend([data_pwt[i].ravel() for i in range(simple_size_renew) if i not in sample])  # 40*5*112*92
    gender_Train.extend(gender_full_renew[i] for i in range(simple_size_renew) if i in sample)
    gender_Test.extend(gender_full_renew[i] for i in range(simple_size_renew) if i not in sample)
    return np.array(trainData),  np.array(gender_Train), np.array(testData), np.array(gender_Test)

def main():
    #loadImageSet()
    t_start_load = time.process_time()
    xTrain_, yTrain, xTest_, yTest = loadImageSet()# 200*10304
    t_end_load = time.process_time()
    t_diff_load = t_end_load - t_start_load
    print('loadimageset cpmpleted---------------------------{:.5f} s'.format(t_diff_load))

    t_start_pca = time.process_time()
    pca1=PCA(n_components=0.9)
    #hyp.plot(xTrain_,'o')
    xTrain_pca=pca1.fit_transform(xTrain_) # 把原始训练集映射到主成分组成的子空间中
    xTest_pca=pca1.transform(xTest_)# 把原始测试集映射到主成分组成的子空间中
    t_end_pca = time.process_time()
    t_diff_pca = t_end_pca - t_start_pca
    print('PCA completed---------------------------------{:.5f} s'.format(t_diff_pca))
    joblib.dump(pca1, 'pca_wiki_model.m')

    # hyp.plot(xTrain_pca, 'o', n_clusters=40)
    clf=SVC(C=1000.0, cache_size=2000, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='poly',
  probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
    #hyp.plot(xTrain_pca, 'o')

    t_start_SVM = time.process_time()
    clf.fit(xTrain_pca,yTrain)
    t_end_SVM = time.process_time()
    t_diff_SVM = t_end_SVM - t_start_SVM
    print('SVM completed---------------------------------{:.5f} s'.format(t_diff_SVM))
    predict=clf.predict(xTest_pca)
    # hyp.plot(predict.reshape(200,1),'o', n_clusters=40)
    # hyp.plot(yTest.reshape(200,1), 'o', n_clusters=40)
    print(clf.score(xTest_pca,yTest))
    print(u'支持向量机识别率: %.2f%%' % ((predict == np.array(yTest)).mean()*100)  )

    print('------------------------')

    joblib.dump(clf, 'svm_gender_model.m')

if __name__ == '__main__':
    # main()
    pca_m = joblib.load('pca_wiki_model.m')
    svm_m = joblib.load('svm_gender_model.m')

    cap = cv2.VideoCapture(0)

    while(True):
        ret,frame = cap.read()
        frame= cv2.flip(frame, 1, dst=None)
        tempimg, face_lenth,img_cap = captureFace(frame)
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
            test_pac = pca_m.transform(np.array(predictdata).reshape(1,-1))
            predict2 = svm_m.predict(test_pac)
            outcome=''
            if(predict2== 0):
                outcome='Female'
            elif(predict2== 1):
                outcome='Male'
            cv2.putText(frame,outcome , (5, 50),  cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 255),2)
            cv2.imshow('frame2Q', frame)
        else:
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
