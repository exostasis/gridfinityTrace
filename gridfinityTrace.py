from typing import List, Tuple
import cv2
import io
import numpy as np
from PIL import Image
import streamlit as st
import svgwrite
from sklearn.cluster import KMeans

from enum import Enum

class ColorFormat(Enum):
    BGR = 1
    RGB = 2
    HSV = 3
    LAB = 4
    GRAY = 5

st.set_page_config(layout="wide", page_title="Gridfinity Trace")

# ---------------- Helper functions ----------
def pil_to_cv2(img_pil: Image.Image) -> cv2.typing.MatLike:
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2: cv2.typing.MatLike) -> Image.Image:
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
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

def template_find_markers(img: cv2.typing.MatLike, template: cv2.typing.MatLike, img_format: ColorFormat, template_format: ColorFormat):
    img_gray = cv2.cvtColor(img, cv_code(img_format, ColorFormat.GRAY))
    template_gray = cv2.cvtColor(template, cv_code(template_format, ColorFormat.GRAY))
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = .4
    
    loc = np.where(res >= threshold)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
    pts = np.column_stack([loc[0], loc[1]])
    kmeans.fit(pts)
     
    return tuple(find_center_from_top_left(pt, template_gray.shape) for pt in kmeans.cluster_centers_)

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
    
    print(np.degrees(np.arctan2(transform_matrix[0, 1], transform_matrix[0, 0])))
    
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

def trace_to_contour(img: cv2.typing.MatLike, blur: float=5, canny_threshold1: float=50.0, canny_threshold2: float=150, approx_eps_ratio: float=.01) -> cv2.typing.MatLike:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if blur > 0:
        img_gray = cv2.GaussianBlur(img_gray, (blur, blur), 0)
        
    edges = cv2.Canny(img_gray, canny_threshold1, canny_threshold2)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return None, edges
    
    main = max(contours, key=cv2.contourArea)
    perim = cv2.arcLength(main, True)
    eps = max(1.0, approx_eps_ratio * perim)
    approx = cv2.approxPolyDP(main, eps, True)
    
    return approx.reshape(-1, 2), edges

def polygon_to_svg_path(points: List[Tuple[float, float]]) -> str:
    if len(points) == 0:
        return ""
    
    d = f"M {points[0][0]:.3f} {points[0][1]:.3f} "
    for p in points[1:]:
        d += f"L {p[0]:.3f} {p[1]:.3f} "
    d += "Z"
    
    return d

def svg_bytes_from_contours(points: List[Tuple[float, float]], width: int, height: int, stroke_width=1.0) -> bytes:
    dwg = svgwrite.Drawing(size=(width, height))
    if points is not None and len(points) > 0:
        path_d = polygon_to_svg_path(points)
        dwg.add(dwg.path(d=path_d, fill="none", stroke="black", stroke_width=stroke_width))
    
    return dwg.tostring().encode("utf-8")

def make_download_button(data_bytes, file_name, label):
    st.download_button(label, data=data_bytes, file_name=file_name, mime="image/svg+xml")
    
def make_st_image(img: cv2.typing.MatLike, caption: str):
    img_pil = cv2_to_pil(img)
    st.image(img_pil, caption)
    
def create_setup_section():
    st.header("Setup")
    column1, column2 = st.columns([1,2])

    with column1:
        image_types = ["png", "jpg", "jpeg", "bmp"]
        uploaded_image = st.file_uploader("Image of item to trace", type=image_types)
        st.caption("Image contain the item + the 3 markers")
        marker = st.file_uploader("Marker image", type=image_types)

    with column2:
        corner_to_arm_marker_distance = st.slider("Distance from corner marker to arm", 1, 1000, 180)
    return uploaded_image,marker,corner_to_arm_marker_distance

def build_page():
    uploaded_image, marker, corner_to_arm_marker_distance = create_setup_section()
        
    if marker is not None: 
        marker_pil = Image.open(marker).convert("RGB")
        marker_cv = pil_to_cv2(marker_pil)
        marker_cv = normalize_lighting(marker_cv, ColorFormat.RGB)
        # st.markdown("---")
        # st.subheader("Normalized lighting")
        # make_st_image(marker_cv, "Marker image after normalizing lighting")
        
    if uploaded_image is not None: 
        main_pil = Image.open(uploaded_image).convert("RGB")
        dpi = main_pil.info["dpi"]
        if dpi[0] != dpi[1]:
            raise ValueError("Dpi is not same in x as y for trace image")
        corner_marker_to_arm_pixel = corner_to_arm_marker_distance / 25.4 * dpi[0]
        main_cv = pil_to_cv2(main_pil)
        main_cv = normalize_lighting(main_cv, ColorFormat.RGB)
        # make_st_image(main_cv, "Trace image after normalizing lighting")
        
    if marker is not None:  
        st.markdown("---")
        st.subheader("Template Matching")

    # ---------- Processing ----------
    if uploaded_image is not None and marker is not None:
        markers = template_find_markers(main_cv, marker_cv, ColorFormat.RGB, ColorFormat.RGB)
        corner_marker, leg_x, leg_y = organize_markers(markers)
        x_direction = 1 if corner_marker[0] <= leg_x[0] else -1
        y_direction = 1 if corner_marker[1] <= leg_y[1] else -1 
        organized_markers = [corner_marker, leg_x, leg_y]
        moved_markers = [organized_markers[0], (organized_markers[0][0] + x_direction * corner_marker_to_arm_pixel, organized_markers[0][1]), (organized_markers[0][0], organized_markers[0][1] + y_direction * corner_marker_to_arm_pixel)]
        transform_matrix = compute_transform_from_markers(organized_markers, moved_markers)
        new_shape, modified_transform_matrix = adjust_transform(main_cv.shape[:2][::-1], transform_matrix)
            
        main_transformed = warp_image(main_cv, modified_transform_matrix, new_shape)
        _, transformed_markers = calculate_new_coordinates(organized_markers, modified_transform_matrix)
        make_st_image(draw_markers(main_transformed, transformed_markers), "The red dots show the markers after fixing perspective")

build_page()