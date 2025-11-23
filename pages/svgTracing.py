from typing import List, Tuple
import cv2
import io
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_cropper import st_cropper
import svgwrite
from sklearn.cluster import KMeans

from enum import Enum

from pages.arucoHelper import get_aruco_types

DEBUG = True

class ColorFormat(Enum):
    BGR = 1
    RGB = 2
    HSV = 3
    LAB = 4
    GRAY = 5

st.set_page_config(layout="wide", page_title="SVG Tracing")

# ---------------- Helper functions ----------
def pil_to_cv2(img_pil: Image.Image) -> cv2.typing.MatLike:
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2: cv2.typing.MatLike, color_format: ColorFormat=ColorFormat.BGR) -> Image.Image:
    if color_format != ColorFormat.RGB:
        img_rgb = cv2.cvtColor(img_cv2, cv_code(color_format, ColorFormat.RGB))
    else:
        img_rgb = img_cv2
    return Image.fromarray(img_rgb)

def hex_to_bgr(hex_string: str) -> cv2.typing.MatLike:
    hex_color = hex_string.lstrip("#")
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)][::-1], dtype=np.uint8).reshape(1,1,3)

def hex_to_hsv(hex_string: str) -> cv2.typing.MatLike:
    bgr = hex_to_bgr(hex_string)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

def cv_code(input: ColorFormat, output: ColorFormat) -> int:
    code = f"COLOR_{input.name}2{output.name}"
    if not hasattr(cv2, code):
        raise ValueError(f"No conversion from {input.name} to {output.name} in OpenCV")
    
    return getattr(cv2, code)
    
def normalize_lighting(img: cv2.typing.MatLike, inputFormat: ColorFormat=ColorFormat.BGR) -> cv2.typing.MatLike:
    img_lab = cv2.cvtColor(img, cv_code(inputFormat, ColorFormat.LAB))
    l, a, b = cv2.split(img_lab)
    l = cv2.equalizeHist(l)
    img_equalized = cv2.merge([l, a, b])
    
    return cv2.cvtColor(img_equalized, cv_code(ColorFormat.LAB, inputFormat))

def generate_color_mask(img: cv2.typing.MatLike, lower_color_hsv: cv2.typing.MatLike, upper_color_hsv: cv2.typing.MatLike, inputFormat: ColorFormat=ColorFormat.BGR) -> cv2.typing.MatLike:
    img_hsv = cv2.cvtColor(img, cv_code(inputFormat, ColorFormat.HSV))
    return cv2.inRange(img_hsv, lower_color_hsv, upper_color_hsv)

def template_find_markers(img: cv2.typing.MatLike, aruco_type: int, img_format: ColorFormat) -> Tuple[List[List[float]], List[int]]:
    img_gray = cv2.cvtColor(img, cv_code(img_format, ColorFormat.GRAY))
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dictionary, parameters)
     
    return detector.detectMarkers(img_gray)

def find_center_from_top_left(top_left_point: Tuple[float, float], shape: Tuple[int, int]) -> Tuple[int, int]:
    y, x = top_left_point
    height, width = shape
    return (round(x + width / 2.0), round(y + height / 2.0))

def angle(u: Tuple[int, int], v: Tuple[int, int]) -> float:
    return np.degrees(
        np.arccos(
            np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)
        )
    )

def organize_markers(markers: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    marker_a, marker_b, marker_c = markers
    
    marker_a = np.array(markers[0])
    marker_b = np.array(markers[1])
    marker_c = np.array(markers[2])
    
    angle_a = angle(marker_b - marker_a, marker_c - marker_a)
    angle_b = angle(marker_a - marker_b, marker_c - marker_b)
    angle_c = angle(marker_a - marker_c, marker_b - marker_c)
    
    angles = np.array([round(angle_a), round(angle_b), round(angle_c)])
    if (angles > 45).sum() == 2:  
        center_index = np.argmin(angles)
    else: 
        center_index = np.argmax(angles)
    
    corner_marker = markers[center_index]
    leg_marker_1, leg_marker_2 = markers[:center_index] + markers[center_index+1:]
    
    # choose the marker closet to horizontal with corner
    if abs(corner_marker[1] - leg_marker_1[1]) <= abs(corner_marker[1] - leg_marker_2[1]):
        return (corner_marker, leg_marker_1, leg_marker_2)
    else:
        return (corner_marker, leg_marker_2, leg_marker_1)  

def draw_markers(img: cv2.typing.MatLike, centers: List[Tuple[int, int]], color=(0, 0, 255), radius=8) -> cv2.typing.MatLike:
    marked_img = img.copy()
    for x, y in centers:
        cv2.circle(marked_img, (round(x), round(y)), radius, color, -1)
        
    return marked_img

def compute_transform_from_markers(src_pts: List[Tuple[int, int]], dst_pts: List[Tuple[int, int]]) -> cv2.typing.MatLike:
    full_projection = len(src_pts) >= 4
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    
    if full_projection:
        transform_matrix, _ = cv2.findHomography(src, dst, method=0)
    else:
        transform_matrix, _ = cv2.estimateAffine2D(src, dst)
     
    return transform_matrix

def warp_image(img: cv2.typing.MatLike, transform_matrix: cv2.typing.MatLike, dsize: cv2.typing.Size) -> cv2.typing.MatLike:
    if transform_matrix.shape == (3, 3):
        return cv2.warpPerspective(img, transform_matrix, dsize)
    else:
        return cv2.warpAffine(img, transform_matrix, dsize)
    
def adjust_transform(img_shape: Tuple[int, int], transform_matrix: cv2.typing.MatLike) -> Tuple[Tuple[int, int], cv2.typing.MatLike]:
    corners = np.float32([[0, 0], [img_shape[0], 0], [img_shape[0], img_shape[1]], [0, img_shape[1]]])
    modified_matrix, transformed_corners = calculate_new_coordinates(corners, transform_matrix)
    min_x = np.min(transformed_corners[:, 0])
    max_x = np.max(transformed_corners[:, 0])
    
    min_y = np.min(transformed_corners[:, 1])
    max_y = np.max(transformed_corners[:, 1])
    
    new_shape = (round(max_x - min_x), round(max_y - min_y))
    translation = np.array([[1, 0, -min_x], [0, 1, -min_y]], dtype=np.float32)
    m_corrected = translation @ np.vstack(modified_matrix)
    
    return new_shape, m_corrected[:2]

def calculate_new_coordinates(corners: np.ndarray, transform_matrix: cv2.typing.MatLike):
    modified_matrix = np.vstack([transform_matrix, [0, 0, 1]])
    transformed_corners = cv2.transform(np.array([corners]), modified_matrix[:2])[0]
    
    return modified_matrix, transformed_corners

def trace_to_contour(img: cv2.typing.MatLike, blur: int=5, kernel_size: int=3, canny_threshold1: float=50.0, canny_threshold2: float=200.0, approx_eps_ratio: float=.01, color_format: ColorFormat=ColorFormat.BGR) -> cv2.typing.MatLike:
    img_gray = cv2.cvtColor(img, cv_code(color_format, ColorFormat.GRAY))
    
    if blur > 0:
        img_gray = cv2.GaussianBlur(img_gray, (blur, blur), 0)
        
    edges = cv2.Canny(img_gray, canny_threshold1, canny_threshold2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return None, edges
    
    main = max(contours, key=cv2.contourArea)
    perim = cv2.arcLength(main, True)
    eps = max(1.0, approx_eps_ratio * perim)
    approx = cv2.approxPolyDP(main, eps, True)
    
    return approx, edges

def polygon_to_svg_path(points: List[Tuple[float, float]]) -> str:
    if len(points) == 0:
        return ""
    
    d = f"M {points[0][0]:.3f} {points[0][1]:.3f} "
    for p in points[1:]:
        d += f"L {p[0]:.3f} {p[1]:.3f} "
    d += "Z"
    
    return d

def svg_bytes_from_contours(points: List[Tuple[float, float]], width: int, height: int, stroke_width=1.0) -> bytes:
    dwg = svgwrite.Drawing(size=(f"{width}mm", f"{height}mm"), viewBox=f"0 0 {width} {height}")
    if points is not None and len(points) > 0:
        path_d = polygon_to_svg_path(points)
        dwg.add(dwg.path(d=path_d, fill="none", stroke="black", stroke_width=stroke_width))
    
    return dwg.tostring().encode("utf-8")

def make_download_button(data_bytes, file_name, label):
    st.download_button(label, data=data_bytes, file_name=file_name, mime="image/svg+xml")
    
def make_st_image(img: cv2.typing.MatLike, caption: str, color_format: ColorFormat=ColorFormat.BGR):
    img_pil = cv2_to_pil(img, color_format)
    st.image(img_pil, caption, width="stretch")
    
def create_setup_section():
    st.header("Setup")
    column1, column2 = st.columns([1,2])

    with column1:
        image_types = ["png", "jpg", "jpeg", "bmp"]
        uploaded_image = st.file_uploader("Image with ArUco markers and item to trace", type=image_types)

    with column2:
        aruco_types = get_aruco_types()
        aruco_type: int = st.selectbox("ArUco marker type", list(aruco_types.keys()))
        aruco_size: int = st.number_input("Size of ArUco marker (in mm)", 10, 100, 40)
        aruco_rows: int = st.number_input("Number of ArUco markers vertically", 1, 5, 2)
        aruco_columns: int = st.number_input("Number of ArUco markers horizontally", 1, 2, 2)
        aruco_spacing: int = st.number_input("How many mm appart are the ArUco markers", 1, 800, 155)
    return uploaded_image, aruco_types[aruco_type], (aruco_columns, aruco_rows), aruco_spacing, aruco_size

def build_page():
    uploaded_image, aruco_type, aruco_board_dimensions, aruco_spacing_mm, aruco_size_mm = create_setup_section()
    total_points = aruco_board_dimensions[0] * aruco_board_dimensions[1]
        
    st.divider()
    if uploaded_image is not None: 
        main_pil = Image.open(uploaded_image).convert("RGB")
        column1, column2 = st.columns([1, 1])
        with column1:
            main_pil = st_cropper(main_pil, realtime_update=True, box_color="red", aspect_ratio=None)
        with column2:
            main_cv = pil_to_cv2(main_pil)
            make_st_image(main_cv, "Cropped image", ColorFormat.BGR)
            main_cv = normalize_lighting(main_cv, ColorFormat.RGB)
            
        
    # ---------- Processing ----------
    if uploaded_image is not None:
        image_size = main_cv.shape[:2][::-1]
        
        corners, ids, _ = template_find_markers(main_cv, aruco_type, ColorFormat.BGR)
        detected_markers_img = main_cv.copy()
        cv2.aruco.drawDetectedMarkers(detected_markers_img, corners, ids)
        st.divider()
        if DEBUG:
            make_st_image(detected_markers_img, "Detected markers", ColorFormat.BGR)
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_type)
        board = cv2.aruco.GridBoard(aruco_board_dimensions, aruco_size_mm, aruco_spacing_mm, aruco_dictionary)
        object_points = np.array([board.getObjPoints()[int(id)] for id in ids], dtype=np.float32).reshape(-1, 3)
        image_points = np.array([corners[int(id)][0] for id in range(total_points)], dtype=np.float32).reshape(-1, 2)

        object_points_min_x = object_points[:,0].min()
        object_points_min_y = object_points[:,1].min()
        
        object_points_normalized = object_points.copy()
        object_points_normalized[:, 0] -= object_points_min_x
        object_points_normalized[:, 1] -= object_points_min_y
        
        H, _ = cv2.findHomography(image_points, object_points[:, :2])
        
        image_corners = np.array([[0,0], [image_size[0], 0], [image_size[0], image_size[1]], [0, image_size[1]]], dtype=np.float32)
        final_corners = cv2.perspectiveTransform(image_corners[None, :, :], H)
        min_x, min_y = final_corners.reshape(-1, 2).min(axis=0)
        max_x, max_y = final_corners.reshape(-1, 2).max(axis=0)
        final_dimensions = int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y))
        T = np.array([[1, 0, -min_x],
                      [0, 1, -min_y],
                      [0, 0, 1]])
        H_shifted = T @ H
        warped_image = cv2.warpPerspective(main_cv, H_shifted, final_dimensions)
        if DEBUG:
            make_st_image(warped_image, "Results after warpPerspective", ColorFormat.BGR)
        
build_page()