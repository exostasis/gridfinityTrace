from typing import Dict
import cv2
from PIL import Image
import streamlit as st
import io
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from pages.arucoHelper import get_aruco_types


def build_page():
    st.set_page_config(layout="wide", page_title="Marker Generation")
    st.sidebar.header("Marker generation")

    aruco_types = get_aruco_types()
    column1, column2 = st.columns([1, 1])
    with column1:
        st.subheader("Marker configuration")
        aruco_type: int = st.selectbox("ArUco Marker Type", list(aruco_types.keys()))
        # 40 mm at 300 DPI
        marker_mm: int = st.slider("Marker size (in mm)", 10, 100, 40) 
        st.subheader("Pdf output configuration")
        margin_mm: int = st.slider("Margin (in mm) to inset the markers in the pdf", 10, 100, 10)
        maximum_rows = int(letter[1] - 2 * margin_mm * mm // (marker_mm * mm))
        maximum_columns = int(letter[1] - 2 * margin_mm * mm // (marker_mm * mm))
        columns: int = st.number_input("Number of markers to print horizontally", 1, maximum_columns, 2, 1)
        rows: int = st.number_input("Number of markers to print vertically", 1, maximum_rows, 2, 1)
        

    buf, rows, columns, center_spacing = generate_pdf(aruco_types, aruco_type, marker_mm, margin_mm, (rows, columns))

    with column2:
        pdf_bytes = buf.getvalue()
        st.pdf(pdf_bytes)
        st.download_button("Download pdf", data = pdf_bytes, file_name=f"ArUco_{rows}x{columns}_{center_spacing}_{marker_mm}.pdf", mime="application/pdf", width="stretch")

def generate_pdf(aruco_types, aruco_type, marker_mm, margin_mm, size):
    rows, columns = size
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    
    margin_points = margin_mm * mm
    marker_points = marker_mm * mm
    
    width_points, height_points = letter
    
    grid_width_points, grid_height_points = (width_points - 2 * margin_points), (height_points - 2 * margin_points)
    assert grid_width_points > 0
    assert grid_height_points > 0
       
    if (rows > 1):
        row_spacing = (grid_height_points - rows * marker_points) // (rows - 1)
    else: 
        row_spacing = 0
    if (columns > 1):   
        column_spacing = (grid_width_points - columns * marker_points) // (columns - 1)
    else:
        column_spacing = 0
     
    spacing = int(min(column_spacing, row_spacing))
    spacing_mm = int(spacing // mm)
    spacing = spacing_mm * mm
    center_spacing = spacing_mm + marker_mm
    
    marker_png_data = [generate_marker(aruco_types, aruco_type, marker_mm, i) for i in range(rows * columns)] 
    marker_coordinates = [(letter[1] - margin_points - (i // columns) * (spacing + marker_points) - marker_points, margin_points + (i % columns) * (spacing + marker_points)) for i in  range(rows * columns)]
    
    for png_data, coordinates in zip(marker_png_data, marker_coordinates):
        image_y, image_x = coordinates
        c.drawImage(ImageReader(io.BytesIO(png_data.tobytes())),
                image_x,
                image_y,
                width=marker_points,
                height=marker_points,
                preserveAspectRatio=True,
                mask="auto")
    c.save()
    
    return buf, rows, columns, center_spacing

def generate_marker(aruco_types: Dict[int, int], aruco_type: int, marker_mm: int, marker_id: int):
    pixels: int = int(marker_mm * 10)

    # ArUco dictionary
    aruco_dict: cv2.aruco.Dictionary = cv2.aruco.getPredefinedDictionary(aruco_types[aruco_type])
    marker_img: cv2.typing.MatLike = cv2.aruco.generateImageMarker(aruco_dict, marker_id, pixels)
    _, png_data = cv2.imencode(".png", marker_img)
    return png_data
    
build_page()