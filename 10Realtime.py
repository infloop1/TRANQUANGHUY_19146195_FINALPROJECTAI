import numpy as np
import cv2
from keras.models import load_model
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
threshold=0.8
cap=cv2.VideoCapture(0)
cap.set(3, 1000)
cap.set(4, 1000)
font=cv2.FONT_HERSHEY_COMPLEX
model_final = load_model('Final_10Main2006.h5')


def wearMask_className(result):
    if result == 0:
        return "BHUY WITHOUT MASK"
    elif result == 1:
        return "BHUY MASK"
    elif result ==2:
        return "DHUNG WITHOUT MASK"
    elif result ==3:
        return "DHUNG MASK"
    elif result ==4:
        return "DTAI WITHOUT MASK"
    elif result ==5:
        return "DTAI MASK"
    elif result ==6:
        return "MTUAN WITHOUT MASK"
    elif result ==7:
        return "MTUAN MASK"
    elif result ==8:
        return "QHUY WITHOUT MASK"
    elif result ==9:
        return "QHUY MASK"


while True:
	sucess, imgOrignal=cap.read()
	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	for x,y,w,h in faces:
		# cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
		# cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
		crop_img=imgOrignal[y:y+h,x:x+h]
		img=cv2.resize(crop_img, (150,150))
		img=img.reshape(1, 150, 150, 3)
		img = img.astype('float32')
		img = img/255
		prediction=model_final.predict(img)
		result_model =np.argmax(model_final.predict(img),axis=-1)
		probabilityValue=np.amax(prediction)
		if probabilityValue>threshold:
			if result_model==1:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
				cv2.putText(imgOrignal, str(wearMask_className(result_model)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
				cv2.putText(imgOrignal, str("19146194"),(x,y+200), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			elif result_model==3:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
				cv2.putText(imgOrignal, str(wearMask_className(result_model)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
				cv2.putText(imgOrignal, str("19146016"),(x,y+200), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			elif result_model==5:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
				cv2.putText(imgOrignal, str(wearMask_className(result_model)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
				cv2.putText(imgOrignal, str("19146255"),(x,y+200), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			elif result_model==7:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
				cv2.putText(imgOrignal, str(wearMask_className(result_model)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
				cv2.putText(imgOrignal, str("19146297"),(x,y+200), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			elif result_model==9:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
				cv2.putText(imgOrignal, str(wearMask_className(result_model)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
				cv2.putText(imgOrignal, str("19146195"),(x,y+200), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			elif result_model==0:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
				cv2.putText(imgOrignal, str(wearMask_className(result_model)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
				cv2.putText(imgOrignal, str("19146194"),(x,y+200), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			elif result_model==2:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
				cv2.putText(imgOrignal, str(wearMask_className(result_model)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
				cv2.putText(imgOrignal, str("19146016"),(x,y+200), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			elif result_model==4:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
				cv2.putText(imgOrignal, str(wearMask_className(result_model)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
				cv2.putText(imgOrignal, str("19146255"),(x,y+200), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			elif result_model==6:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
				cv2.putText(imgOrignal, str(wearMask_className(result_model)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
				cv2.putText(imgOrignal, str("19146297"),(x,y+200), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			elif result_model==8:
				cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(50,50,255),2)
				cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (50,50,255),-2)
				cv2.putText(imgOrignal, str(wearMask_className(result_model)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)	
				cv2.putText(imgOrignal, str("19146195"),(x,y+200), font, 0.75, (255,255,255),1, cv2.LINE_AA)

				
	cv2.imshow("TRAN QUANG HUY 19146195",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break
    # cv2.imshow('Face ID Wear Mask Detector', imgOrignal)
	# if cv2.waitKey(1) & 0xFF == ord('q'):  #Press q to exit
	# 	break

cap.release()
cv2.destroyAllWindows()