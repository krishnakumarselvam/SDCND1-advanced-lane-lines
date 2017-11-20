import pickle

def read_camera_calibration(relative_file_path):
	camera_calibration_data = pickle.load( open(relative_file_path, "rb" ))
	mtx, dist = camera_calibration_data['mtx'], camera_calibration_data['dist']
	return mtx, dist