import RPi.GPIO as G
import time

G.setmode(G.BOARD)
G.setup(18, G.OUT)
G.setup(22, G.OUT)
pwm = G.PWM(18, 1000)
servo = G.PWM(22, 50)

servo.start(7.5)
for i in range(101):
	pwm.start(i)
	time.sleep(0.1)


try:
    while True:
        input()
        servo.ChangeDutyCycle(10.5)
        time.sleep(0.5)
        servo.ChangeDutyCycle(7.5)
except:
	servo.ChangeDutyCycle(10.5)
	pwm.stop()