from matrixKeypad import keypad
import time
kp= keypad()

def digit():
    r= None
    while r==None:
        r=kp.getKey()
    return r

##
d1=digit()
print(d1)
time.sleep(1)
d2=digit()
print(d2)
time.sleep(1)
d3=digit()
print(d3)
time.sleep(1)
d4=digit()
print(d4)
print(d1,d2,d3,d4)