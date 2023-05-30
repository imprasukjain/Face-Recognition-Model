import numpy as np
import cv2
import sklearn
import pickle
# Haar_Cascade Model
haar = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
# SVM mode
model_SVM = pickle.load(open('./models/model_svm.pickle',mode='rb'))
# PCA model
PCA_model = pickle.load(open('./models/pca_dict.pickle',mode='rb'))

model_PCA = PCA_model['pca']
mean_face_Arr = PCA_model['mean_face']

def Face_Recognition(filename,path = True):
    if path:
        # Reading the image
        img = cv2.imread(filename)
    else:
        img = filename #array
    #img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # Converting it into Gray-scale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Detecting face using Haar-cascade
    faces= haar.detectMultiScale(gray,1.5,4)
    predictions = []
    for x,y,w,h in faces:
        #cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,255,0),2)
        roi = gray[y:y+h,x:x+w]
        # Normalization
        roi= roi/255.0
        # Resizing
        if roi.shape[1]>100:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
        # Flattening
        roi_reshape = roi_resize.reshape(1,10000)
        # Subtracting with mean face
        roi_mean = roi_reshape-mean_face_Arr
        # Eigen_image
        eigen_image = model_PCA.transform(roi_mean)
        # Eigen Image
        eig_image = model_PCA.inverse_transform(eigen_image)
        # Results
        results = model_SVM.predict(eigen_image)
        prob_Score = model_SVM.predict_proba(eigen_image)
        prob_Score_max = prob_Score.max()

    
    # Generating Report

        text = "%s : %d"%(results[0],prob_Score_max*100)

        #defining color based on results
        if results[0] == 'male':
            color = (255,255,0)
        elif results[0] == 'female':
            color = (255,0,255)

        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color,-1)
        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
        output = {
            'roi':roi,
            'eig_img':eig_image,
            'prediction_name':results[0],
            'score':prob_Score_max
        }
        predictions.append(output)
    return img,predictions 



