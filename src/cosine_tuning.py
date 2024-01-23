import numpy as np
from scipy.optimize import curve_fit

class CosineTuning():
    
    def get_radius_angles(self, x_coords, y_coords):
        '''Get radius and angles from x and y coordinates by fitting a parametric model'''
        angle = np.arctan2(y_coords - np.mean(y_coords), x_coords - np.mean(x_coords)) * 180 / np.pi
        radius = np.linalg.norm([x_coords - np.mean(x_coords), y_coords - np.mean(y_coords)], axis = 0)
        return radius, angle

    def get_angle_indices(self, angles, num_segments = 20):
        angle_ranges = np.linspace(-180,180,num_segments)
        angle_indices = {}
        for start, stop in list(zip(angle_ranges[:-1], angle_ranges[1:])):
            angle_indices[f"{round(start,2)} - {round(stop,2)}"] = (np.where((angles > start) & (angles < stop))[0])
        return angle_indices

        
if __name__ == "__main__":
    
    cosine_tuning = CosineTuning()
    radius, angles = cosine_tuning.get_radius_angles(np.array([5,5,5]), np.array([1,2,3]))
    angles_indices = cosine_tuning.get_angle_indices(angles)
    print(angles_indices)
    