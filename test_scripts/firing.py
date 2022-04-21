import RPi.GPIO as G
import pigpio
import time

# Use pigpio for motors to utilize hardware PWM. This addresses surge current that software PWM may fail to prevent under heavy software load.
pi = pigpio.pi()
# Use RPi.GPIO for servo.
# It is unknown if using two different GPIO control libraries have any conflicts, however we did not experience any issues.
G.setmode(G.BOARD)

pi.set_mode(18, pigpio.OUTPUT)

G.setup(22, G.OUT)
servo = G.PWM(22, 50)

# Start at 90%, such that the servo arm blocks the ball from falling into the tube.
servo.start(7.5)

# Slowly increase the duty cycle to prevent surge current from the motors.
for i in range(100):
    pi.hardware_PWM(18, 10000, 10000 * i)
    time.sleep(0.01)

print("spun up!")

# On enter, pull the servo arm back before pushing it forward, allowing the ball to drop into the firing tube and subsequently pushing it into the flywheels.
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
