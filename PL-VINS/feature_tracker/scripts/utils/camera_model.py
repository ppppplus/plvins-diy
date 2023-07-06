import os
import math
import rospy 
class CameraModel():
    def __init__(self, params):
        self.support_camera_types = ["PINHOLE", "KANNALA_BRANDT"]
        self.params = params
        self.model_type = self.params["model_type"]
        self.checkCameraType()
        rospy.loginfo(params.keys())
    
    def checkCameraType(self):
        if self.model_type not in self.support_camera_types:
            raise ValueError(
                "[Error] The camera type selection '%s' is not supported.", 
                self.model_type)

    def generateCameraModel(self):
        if self.model_type == "PINHOLE":
            camera_model = PinholeCamera(self.params["distortion_parameters"], self.params["projection_parameters"])
            return camera_model
        elif self.model_type == "KANNALA_BRANDT":
            camera_model = KannalabrantCamera(self.params["distortion_parameters"], self.params["projection_parameters"])
            return camera_model
        else:
            raise ValueError(
                "[Error] The camera type selection '%s' is not supported.", 
                self.model_type)

class KannalabrantCamera:
    def __init__(self, projection_parameters):
        self.k2 = projection_parameters["k2"]
        self.k3 = projection_parameters["k3"]
        self.k4 = projection_parameters["k4"]
        self.k5 = projection_parameters["k5"]
        self.mu = projection_parameters["mu"]
        self.mv = projection_parameters["mv"]
        self.u0 = projection_parameters["u0"]
        self.v0 = projection_parameters["v0"]
    def distortion(self, theta, x, y, r):
        theta_2 = theta * theta
        theta_d = theta*(1+self.k2*theta_2+self.k3*theta_2*theta_2
                         +self.k3*theta_2*theta_2*theta_2
                         +self.k4*theta_2*theta_2*theta_2*theta_2)
        dx = self.mu*theta_d*x / math.sqrt(r) + self.u0
        dy = self.mv*theta_d*y / math.sqrt(r) + self.v0
        return [dx, dy, 1.0]

    def liftProjective(self, p):
        # convert pixel coord to normalized coord w/o distortion
        r = math.sqrt(p[0]*p[0] + p[1]*p[1])
        theta = math.atan2(p[1], p[0])
        d = self.distortion(theta, p[0], p[1], theta)
        return d



class PinholeCamera:

    def __init__(self, distortion_parameters, projection_parameters):

        self.fx = projection_parameters["fx"]
        self.fy = projection_parameters["fy"]
        self.cx = projection_parameters["cx"]
        self.cy = projection_parameters["cy"]
        self.d = list(distortion_parameters.values())

    def distortion(self, p_u):

        k1 = self.d[0]
        k2 = self.d[1]
        p1 = self.d[2]
        p2 = self.d[3]

        mx2_u = p_u[0] * p_u[0]
        my2_u = p_u[1] * p_u[1]
        mxy_u = p_u[0] * p_u[1]
        rho2_u = mx2_u + my2_u
        rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u

        d_u0 = p_u[0] * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u)
        d_u1 = p_u[1] * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u)

        return (d_u0, d_u1)

    def liftProjective(self, p):
        # convert pixel coord to normalized coord w/o distortion
        m_inv_K11 = 1.0 / self.fx
        m_inv_K13 = -self.cx / self.fx
        m_inv_K22 = 1.0 / self.fy
        m_inv_K23 = -self.cy / self.fy

        mx_d = m_inv_K11 * p[0] + m_inv_K13
        my_d = m_inv_K22 * p[1] + m_inv_K23

        n = 8
        d_u = self.distortion((mx_d,my_d))
        mx_u = mx_d - d_u[0]
        my_u = my_d - d_u[1]
        
        for _ in range(n-1):
            d_u = self.distortion((mx_u, my_u))
            mx_u = mx_d - d_u[0]
            my_u = my_d - d_u[1]
        
        return (mx_u, my_u, 1.0)
