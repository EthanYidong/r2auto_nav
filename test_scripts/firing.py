import RPi.GPIO as G
import pigpio
import time

pi = pigpio.pi()
G.setmode(G.BOARD)

pi.set_mode(18, pigpio.OUTPUT)
G.setup(22, G.OUT)
servo = G.PWM(22, 50)

servo.start(7.5)

for i in range(100):
    pi.hardware_PWM(18, 10000, 10000 * i)
    time.sleep(0.01)

print("spun up!")

try:
    while True:
        input()
        servo.ChangeDutyCycle(10.5)
        time.sleep(0.5)
        servo.ChangeDutyCycle(7.5)
except:
    servo.ChangeDutyCycle(7.5)
    pi.hardware_PWM(18, 10000, 0)
    pi.stop()
    time.sleep(0.5)