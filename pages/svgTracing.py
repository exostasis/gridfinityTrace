from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_cropper import st_cropper
import svgwrite

from enum import Enum

from pages.arucoHelper import get_aruco_types

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import supervision as sv

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

def template_find_markers(img: cv2.typing.MatLike, aruco_type: int, img_format: ColorFormat) -> Tuple[List[List[float]], List[int]]:
    img_gray = cv2.cvtColor(img, cv_code(img_format, ColorFormat.GRAY))
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dictionary, parameters)
     
    return detector.detectMarkers(img_gray)

def correct_image(aruco_type, aruco_board_dimensions, aruco_spacing_mm, aruco_size_mm, total_points, main_cv, image_width, image_height, corners, ids):
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_type)
    board = cv2.aruco.GridBoard(aruco_board_dimensions, aruco_size_mm, aruco_spacing_mm, aruco_dictionary)
    object_points = np.array([board.getObjPoints()[int(id)] for id in ids], dtype=np.float32).reshape(-1, 3)
    image_points = np.array([corners[int(id)][0] for id in range(total_points)], dtype=np.float32).reshape(-1, 2)
 
    H, _ = cv2.findHomography(image_points, object_points[:, :2])
        
    final_dimensions, H_shifted = shift_homography(image_width, image_height, H)
    corrected_image = cv2.warpPerspective(main_cv, H_shifted, final_dimensions)
    
    return corrected_image

def shift_homography(image_width, image_height, H):
    # Calculate new bounding box
    image_corners = np.array([[0,0], [image_width, 0], [image_width, image_height], [0, image_height]], dtype=np.float32)
    final_corners = cv2.perspectiveTransform(image_corners[None, :, :], H)
    
    min_x, min_y = final_corners.reshape(-1, 2).min(axis=0)
    max_x, max_y = final_corners.reshape(-1, 2).max(axis=0)
    final_dimensions = int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y))
    T = np.array([[1, 0, -min_x],
                      [0, 1, -min_y],
                      [0, 0, 1]])
    H_shifted = T @ H
    
    return final_dimensions, H_shifted

def shadow_removal(corrected, color_format):
    hsv = cv2.cvtColor(corrected, cv_code(color_format, ColorFormat.HSV))
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)
    v_norm = cv2.normalize(v_clahe, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    hsv_corrected = cv2.merge([h, s, v_norm])
    illum_corrected = cv2.cvtColor(hsv_corrected, cv_code(ColorFormat.HSV, ColorFormat.BGR))
    bgr = cv2.split(illum_corrected)
    correct_channels = []
    for ch in bgr:
        dillatd = cv2.dilate(ch, np.ones((9, 9), np.uint8))
        bg = cv2.medianBlur(dillatd, 31)
        diff = 255 - cv2.absdiff(ch, bg)
        correct_channels.append(diff)
    
    return cv2.merge(correct_channels)
    # h, w = dimensions
    # k = int(max(51, min(h, w) * .2))
    # if k % 2 == 0:
    #     k += 1
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # background = cv2.morphologyEx(corrected_gray, cv2.MORPH_CLOSE, kernel)
    # normalized = cv2.normalize((corrected_gray.astype(np.float32) - background.astype(np.float32)), None, 0, 255, cv2.NORM_MINMAX)
    # return np.clip(normalized, 0, 255).astype(np.uint8)
    
def msrcr(img, scales=[15,80,250], G=5.0, b=25.0, alpha=125.0, beta=46.0):
    # Multi-scale Retinex with Color Restoration (basic)
    img = img.astype(np.float32) + 1.0
    retinex = np.zeros_like(img)
    for s in scales:
        blur = cv2.GaussianBlur(img, (0,0), s)
        retinex += np.log(img) - np.log(blur + 1e-8)
    retinex = retinex / len(scales)
    # color restoration
    sum_channels = np.sum(img, axis=2, keepdims=True)
    c = alpha * (np.log(beta * img) - np.log(sum_channels + 1e-8))
    msrcr = G * (retinex * c + b)
    # normalize to 0..255
    msrcr = (msrcr - msrcr.min())/(msrcr.max()-msrcr.min())*255.0
    msrcr = np.clip(msrcr, 0, 255).astype(np.uint8)
    return msrcr

def remove_shadows_retinex(bgr):
    # Convert to float
    img = bgr.astype(np.float32) / 255.0

    # Parameters for MSR
    sigma_list = [15, 80, 250]

    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        blur = cv2.GaussianBlur(img, (0,0), sigma)
        retinex += np.log10(img + 1e-6) - np.log10(blur + 1e-6)

    retinex = retinex / len(sigma_list)

    # Normalize to 0–255
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex))
    retinex = (retinex * 255).astype(np.uint8)

    return retinex

def process_image(aruco_type, aruco_board_dimensions, aruco_spacing_mm, aruco_size_mm, total_points, main_cv):
    image_width, image_height = main_cv.shape[:2][::-1]
        
    corners, ids, _ = template_find_markers(main_cv, aruco_type, ColorFormat.BGR)
    detected_markers_img = main_cv.copy()
    cv2.aruco.drawDetectedMarkers(detected_markers_img, corners, ids)
    st.divider()
    
    if DEBUG:
        make_st_image(detected_markers_img, "Detected markers", ColorFormat.BGR)
        
    corrected_image = correct_image(aruco_type, aruco_board_dimensions, aruco_spacing_mm, aruco_size_mm, total_points, main_cv, image_width, image_height, corners, ids)
    
    if DEBUG:
        make_st_image(corrected_image, "Results after warpPerspective", ColorFormat.BGR)
        
    # # --- SHADOW ROBUST PREPROCESSING -----------------------------------
    # # Convert to LAB because L channel separates lighting from color
    # lab = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2LAB)
    # L, A, B = cv2.split(lab)

    # # 1) Estimate illumination using a large Gaussian blur
    # illum = cv2.GaussianBlur(L, (0, 0), sigmaX=35, sigmaY=35)

    # # Avoid divide-by-zero
    # illum = np.maximum(illum, 1)

    # # 2) Flatten illumination (homomorphic-like)
    # L_flat = (L.astype(np.float32) / illum) * 128
    # L_flat = cv2.normalize(L_flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # # 3) Local contrast improvement
    # clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    # L_final = clahe.apply(L_flat)

    # # Reassemble image for visualization (optional)
    # shadow_free = cv2.merge([L_final, A, B])
    # shadow_free_bgr = cv2.cvtColor(shadow_free, cv2.COLOR_LAB2BGR)

    # if DEBUG:
    #     make_st_image(shadow_free_bgr, "Shadow-corrected (LAB Illumination Flattened)", ColorFormat.BGR)

    # # --- EDGE DETECTION (clean & stable) --------------------------------
    # # Smooth noise without destroying edges
    # blur = cv2.bilateralFilter(L_final, 7, 75, 75)

    # # Canny on cleaned luminance
    # edges = cv2.Canny(blur, 40, 120)

    # # Close gaps in the contour
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    # edges = cv2.dilate(edges, kernel, iterations=1)

    # if DEBUG:
    #     make_st_image(edges, "Edges after shadow removal", ColorFormat.GRAY)
    
    # ------------ SHADOW-INVARIANT OBJECT SEGMENTATION ----------------
    dino = Model(
        model_config_path=".venv/lib/python3.12/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="models/gdino_swint_ogc.pth"
    )
    
    detections, descriptions = dino.predict_with_caption(
        image=corrected_image,
        caption="Detected items",
        box_threshold=.25,
        text_threshold=.25
    )
    
    if len(detections) == 0:
        st.markdown("Nothing detected")
    padding = 2
    
    for box in detections:
        x1, y1, x2, y2 = box[0]
        x1, y1, x2, y2 = int(x1 - padding), int(y1 - padding), int(round(x2) + padding), int(round(y2) + padding)
        roi = corrected_image[y1:y2, x1:x2]
        make_st_image(roi, "detected objects")
        sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h.pth")
        predictor = SamPredictor(sam)
        predictor.set_image(corrected_image)
        tool_mask, _, _ = predictor.predict(
            box=np.array([x1, y1, x2, y2])
        )
        if tool_mask.dtype != np.uint8:
            tool_mask = (tool_mask * 255).astype(np.uint8)
        if len(tool_mask.shape) == 3:
            tool_mask = np.transpose(tool_mask, (1, 2, 0))
            tool_mask = cv2.cvtColor(tool_mask, cv_code(ColorFormat.BGR, ColorFormat.GRAY))
        cnts, _ = cv2.findContours(tool_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main = max(cnts, key=cv2.contourArea)
        contour_image = corrected_image.copy()
        make_st_image(cv2.drawContours(contour_image, [main], -1, (0, 255, 0), 1), "Contour found", ColorFormat.BGR)

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
            
        
    # ---------- Processing ----------
    if uploaded_image is not None:
        process_image(aruco_type, aruco_board_dimensions, aruco_spacing_mm, aruco_size_mm, total_points, main_cv)
 
build_page()