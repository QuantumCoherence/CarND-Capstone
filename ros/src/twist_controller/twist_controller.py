from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
        accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle, max_speed):

        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        kp = 0.3
        ki = 0.1
        kd = 0.
        mn = 0. # Minimum throttle value
        mx = 0.2 # Maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)
        self.current_angular_vel = 0.


        tau = 0.5 # 1/(2pi*tau) = cutoff frequency
        ts = 0.02 # sample time
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()
        self.max_speed = max_speed


    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel, current_ang_vel):

        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        current_vel = self.vel_lpf.filt(current_vel)

        #
        # steering_gain - do damoen the steering angle
        # doesn't seem to be necessary if camera images are dropped
        # most
        #
        #if current_ang_vel >0.01:
        #    angular_gain = angular_vel / current_ang_vel;
        #else:
        #    angular_gain = 1.0
        #if abs(angular_gain) > 1.15:
        #    angular_gain = 1.1
        #elif abs(angular_gain) < 0.85   :
        #    angular_gain = 0.9
        #else:
        #    angular_gain = 1.0

        #speed_gain = current_vel/ self.max_speed
        #if speed_gain > 0.5:
        #    speed_gain = 0.65
        #elif speed_gain > 0.65:
        #        speed_gain = 0.45
        #else:
        #    speed_gain = 1.0

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel, 1.0) #speed_gain*angular_gain)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        if linear_vel == 0 and current_vel < 0.1:
            throttle = 0
            brake = 400 # N*m - to hold the car in place if we are stopped at a light, Acceleration = 1m/s^2
        elif throttle < 0.1 and vel_error < 0 :
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius # Torque is measured in N*m

        return throttle, brake, steering








