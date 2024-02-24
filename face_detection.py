#importing librarys
import cv2
import numpy as npy
import face_recognition as face_rec

#function
def resize(img,size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension =(width,height)
    return cv2.resize(img , dimension, interpolation= cv2.INTER_AREA)

# img declaration
ashi=face_rec.load_image_file('sampleing/ashi.jpg')
ashi = cv2.cvtColor(ashi,cv2.COLOR_BGR2RGB)
ashi = resize(ashi, 0.50)
ashi_sample=face_rec.load_image_file('sampleing/elon.jpg')
ashi_sample = cv2.cvtColor(ashi_sample,cv2.COLOR_BGR2RGB)
ashi_sample = resize(ashi_sample, 0.50)

# findng the face location

faceLocation_ashi = face_rec.face_locations(ashi)[0]
encode_ashi = face_rec.face_encodings(ashi)[0]
cv2.rectangle(ashi, (faceLocation_ashi[3], faceLocation_ashi[0]), (faceLocation_ashi[1], faceLocation_ashi[2]), (255,0, 255), 3)

faceLocation_ashi_sample = face_rec.face_locations(ashi_sample)[0]
encode_ashi_sample = face_rec.face_encodings(ashi_sample)[0]

cv2.rectangle(ashi_sample, (faceLocation_ashi_sample[3], faceLocation_ashi_sample[0]),
              (faceLocation_ashi_sample[1], faceLocation_ashi_sample[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_ashi], encode_ashi_sample)
print(results)

cv2.putText(ashi_sample, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2)
cv2.imshow('main_img', ashi)
cv2.imshow('test_img', ashi_sample)
cv2.waitKey(0)
cv2.destroyAllWindows()