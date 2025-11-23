import cv2


def get_aruco_types() -> dict:
    aruco_dict_options = {}
    
    for attr in dir(cv2.aruco):
        if attr.startswith("DICT_"):
            dict_id = getattr(cv2.aruco, attr)
            aruco_dict_options[attr] = dict_id
    
    aruco_dict_options = dict(sorted(aruco_dict_options.items()))
    
    return aruco_dict_options