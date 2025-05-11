import cv2
cam=cv2.VideoCapture(0)
face_cas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_smile=cv2.CascadeClassifier('haarcascade_smile.xml')
while True:
    _,frame=cam.read()
    original=frame.copy()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_cas.detectMultiScale(gray,1.3,5)
    for x,y,w,h in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        face_roi=frame[y:y+h,x:x+w]
        gray_roi=gray[y:y+h,x:x+w]
        smile=face_smile.detectMultiScale(gray_roi,1.3,25)
        for a,b,c,d in smile:
            cv2.rectangle(face_roi,(a,b),(a+c,b+c),(0,0,255),2)
            cv2.imwrite('selfie.png',original)
    cv2.imshow('cam star',frame)
    if cv2.waitKey(5)==ord('q'):
        break