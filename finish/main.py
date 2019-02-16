import cv2
import numpy as np
import os
from PIL import Image
from matrixKeypad import keypad
import RPi.GPIO as GPIO
import time
import threading
import pymysql
##GPIO===========
LED_RED=23
LED_GREEN=18
servo_motor=24
BUZZER=15
##============
isOPEN=0
##===========
password=[1,2,3,4]
isWrong=0
isCorrect=0
##==============

def dbtorasp():
    global isOPEN
    while(True):
        time.sleep(0.1)
        con = pymysql.connect(host='localhost',user='iot',password='123456',db='iot',charset='utf8')
        curs = con.cursor()
        sql = "select door from pin";
        curs.execute(sql)
        temp=curs.fetchone()
        if(temp[0]=='1'):
            isOPEN=temp[0]
            sql = "update pin set door = 0";
            curs.execute(sql)
            con.commit()
        con.close()
        
def dbCon():
    global isOPEN
    while(True):
        time.sleep(0.1)
        if(isOPEN):
            con = pymysql.connect(host='localhost',user='iot',password='123456',db='iot',charset='utf8')
            curs = con.cursor()
            sql = "insert into logrecord values(sysdate())";
            curs.execute(sql)
            con.commit()
            con.close()
            while(True):
                time.sleep(0.1)
                if(isOPEN==0):
                    break
            
    
def initGPIO():
    global LED_RED
    global LED_GREEN
    global servo_motor
    global p
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_RED,GPIO.OUT)
    GPIO.setup(LED_GREEN,GPIO.OUT)
    GPIO.setup(BUZZER,GPIO.OUT)
    GPIO.setup(servo_motor, GPIO.OUT)
    p = GPIO.PWM(servo_motor,50)
    p.start(3)


def servo():
    global isOPEN
    global p
    while(True):
        time.sleep(0.01)
        if(isOPEN):
            p.ChangeDutyCycle(7)
            time.sleep(2)
            isOPEN = 0
            p.ChangeDutyCycle(3)
            
def buzzer():
    global isWrong
    global isCorrect
    global BUZZER
    global isOPEN
    while(True):
        time.sleep(0.01)
        if(isWrong):
            GPIO.output(BUZZER,True)
            time.sleep(0.1)
            GPIO.output(BUZZER,False)
            time.sleep(0.1)
            GPIO.output(BUZZER,True)
            time.sleep(0.1)
            GPIO.output(BUZZER,False)
            time.sleep(0.1)
            GPIO.output(BUZZER,True)
            time.sleep(0.1)
            GPIO.output(BUZZER,False)
            time.sleep(0.1)
            GPIO.output(BUZZER,True)
            time.sleep(0.1)
            GPIO.output(BUZZER,False)
            time.sleep(0.1)
            isWrong=0
        if(isCorrect or isOPEN):
            GPIO.output(BUZZER,True)
            time.sleep(1)
            GPIO.output(BUZZER,False)
            time.sleep(1)
            isCorrect=0
            isOPEN=0
    
def LED():
    global isOPEN
    global LED_RED
    global LED_GREEN
    while(True):
        time.sleep(0.01)
        if(isOPEN):
            GPIO.output(LED_GREEN,True)
            GPIO.output(LED_RED,False)
        else:
            GPIO.output(LED_RED,True)
            GPIO.output(LED_GREEN,False)
            
def digit(kp):
    r= None
    save=None
    while r==None:
        r=kp.getKey()
    save=r
    while r !=None:
        r=kp.getKey()
    return save

def keypadinput():
    global isOPEN
    global isCorrect
    global isWrong
        
    kp= keypad()
    input=[0,0,0,0]
    cnt=0
    enter=0
    while(True):
        temp=digit(kp)
        
        if(temp=='*'):
            if(cnt>0):
                cnt=cnt-1
                print("back")
            
        elif(temp=='#'):
            enter=1
            print(temp)
        else:
            if(cnt>=4):
                print(" input overflow!!")
                
            else:
                input[cnt]=temp
                cnt=cnt+1
                print(temp)
                
        if(enter):
            print(input[0],input[1],input[2],input[3])
            if(password[0] ==input[0] and password[1] == input[1] and password[2]==input[2] and password[3]==input[3]):
                print("password correct! OOO")
                isOPEN=1
                isCorrect=1
            else:
                print("password incorrect!XXX ")
                isWrong=1
                cnt=0
            cnt=0
            enter=0
            for i in range(4):
                input[i]=0
        time.sleep(0.5)







initGPIO()
thread1=threading.Thread(target=keypadinput)
thread2=threading.Thread(target=servo)
thread3=threading.Thread(target=LED)
thread4=threading.Thread(target=buzzer)
thread5=threading.Thread(target=dbCon)
thread6=threading.Thread(target=dbtorasp)
thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
thread6.start()



names = ['None']
##=========================
def face_dataset():
    global names
    
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    face_detector = cv2.CascadeClassifier('FACE/front_face.xml')
    # For each person, enter one numeric face id
    face_id = input('\n enter user id end press <return> ==>  ')
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    while(True):
        ret, img = cam.read()
        #img = cv2.flip(img, -1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User_" + str(face_id) + '_' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30: # Take 30 face sample and stop video
            names.append(face_id)
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

##=========================
def face_training():
        
    # Path for face image database
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("FACE/front_face.xml");
    # function to get the images and label data
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split("_")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples,ids
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


##====================
    
def face_recognize():
    global names
    global isOPEN
    local_ids=[]
    path = 'dataset'
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        local_id = int(os.path.split(imagePath)[-1].split("_")[1])
        local_ids.append(local_id)
    local_ids=list(set(local_ids))
    for i in range(len(local_ids)):
        names.append(local_ids[i])
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "FACE/front_face.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX
    #iniciate id counter
    id = 0
    # names related to ids: example ==> Marcelo: id=1,  etc
    
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    while True:
        ret, img =cam.read()
        #img = cv2.flip(img, -1) # Flip vertically
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                if(confidence<75 and id>0):
                    isOPEN=1
                confidence = "  {0}%".format(round(100 - confidence))
                
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
        cv2.imshow('camera',img)
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    

def test():
    while(True):
        time.sleep(1)
        print('menu : add , training , recognize')
        keyboard=raw_input()
        if(keyboard=='add'):
            face_dataset()
        elif(keyboard=='training'):
            face_training()
        elif(keyboard=='recognize'):
            face_recognize()
        else:
            print('no operate')
            
            
thread77=threading.Thread(target=test)
thread77.start()
