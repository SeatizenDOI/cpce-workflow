import os
import tqdm
import time
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from textwrap import wrap
from pypdf import PdfWriter
from natsort import natsorted

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table
from reportlab.lib.pagesizes import letter, landscape

import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles

COUNTRY_CODE_FOR_HIGH_ZOOM_LEVEL = ["REU"]
MAP_TEMP_PATH = "_map"

def evenly_select_images_on_interval(image_list):
    '''
    Function to select images evenly throughout a list based on their indexes.
    '''
    total_images = len(image_list)
    index_list = np.linspace(0, total_images, 100, dtype=int, endpoint=False)
    selected_images = [image_list[i] for i in index_list]
    return selected_images

def create_trajectory_map(metadata_path, map_path, alpha3_code):
    
    df = pd.read_csv(metadata_path)
    if "GPSLatitude" not in df or "GPSLongitude" not in df:
        print("[ERROR] Not enough gps information to draw trajectory map")
        return False

    imagery = GoogleTiles(url='https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}')

    fig = plt.figure(figsize=(2,2), dpi=300)
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    ax.set_extent([df.GPSLongitude.min()-0.005, df.GPSLongitude.max()+0.005, df.GPSLatitude.min()-0.005,df.GPSLatitude.max()+0.005])
    # ax.set_extent([df.GPSLongitude.min()-0.001, df.GPSLongitude.max()+0.001, df.GPSLatitude.min()-0.001,df.GPSLatitude.max()+0.001])
    ax.add_image(imagery, 17) # aldabra/mayotte position so we adjust the zoom level
    ax.plot(df.GPSLongitude, df.GPSLatitude, color='tab:grey', linewidth=0.1)

    fig.savefig(map_path, bbox_inches='tight',pad_inches=0, dpi=300)
    print("Trajectory map created!")
    return True

def create_predictions_map(metadata_path: Path):
    """
        Create a folder of map for each predictions.
        - predictions_path is the path to predictions file
        
        return the folder path to the images 
    """
    if not Path.exists(metadata_path):
        print(f"File {metadata_path} doesn't exist")
        return

    df = pd.read_csv(metadata_path)
    if len(df) == 0: return None # No predictions
    if "GPSLongitude" not in df or "GPSLatitude" not in df: return None # No GPS coordinate
    if round(df["GPSLatitude"].std(), 10) == 0.0 or round(df["GPSLongitude"].std(), 10) == 0.0: return None # All frames have the same gps coordinate

    imagery = GoogleTiles(url='https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}')

    # Create temp directory
    tmp_path = Path(metadata_path.parent, f"pred_{int(time.time())}_jpg")
    tmp_path.mkdir(parents=True, exist_ok=True)
    if len(list(tmp_path.iterdir())) > 0:
        for i in tmp_path.iterdir():
            i.unlink()

    classes = list(set(list(df)) - set(["OriginalFileName", "FileName", "GPSLongitude", "GPSLatitude", "DateTime", "relative_file_path", "Nb_Points"]))

    for i, category in tqdm.tqdm(enumerate(classes), total=len(classes)):
        fig = plt.figure(figsize=(8, 6), dpi=300)
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
        ax.set_extent([df.GPSLongitude.min()-0.0003, df.GPSLongitude.max()+0.0003, df.GPSLatitude.min()-0.0003, df.GPSLatitude.max()+0.0003])
        ax.add_image(imagery, 17)
        ax.plot(df[df[category] != 0].GPSLongitude, df[df[category] != 0].GPSLatitude, '.', color='tab:red', markersize=4.5, markeredgewidth=0)
        ax.plot(df[df[category] == 0].GPSLongitude, df[df[category] == 0].GPSLatitude, '.', color='tab:gray', markersize=2.0, markeredgewidth=0)
        ax.set_title(category)
        path_to_save_img = Path(tmp_path, f"multiple_page_{category.replace('/', '')}_subplots.jpg")
        plt.savefig(str(path_to_save_img), dpi=300)
        plt.close()

    return tmp_path

def create_pdf_preview(session_path: Path):

    metadata_path = Path(session_path, "METADATA", "metadata.csv")

    # PDF creation
    pdf_file = os.path.join(session_path, f"000_{session_path.name}_preview.pdf")
    map_temp_path = Path(f"{session_path.name}_map.png")
    c = canvas.Canvas(pdf_file, pagesize=letter)
    page_width, page_height = letter
    max_height = page_height - 100
    alpha3_code = session_path.name.split("_")[1].split("-")[0] # Extract MUS from 20221011_MUS-Lemorne_scuba_1
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, 730, "Session Summary")
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.blue)
    c.drawString(30, 705, session_path.name)

    # Trajectory map
    img_preview_y = 730
    if create_trajectory_map(metadata_path, map_temp_path, alpha3_code):
        print("Adding map to the PDF...")
        image_map = Image.open(map_temp_path)
        image_map_width, image_map_height = image_map.size
        x = (page_width - image_map_width) / 2
        y = (page_height - image_map_height) / 2
        c.drawImage(map_temp_path, x, y)
        os.remove(map_temp_path) # deleting map.png

        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(colors.black)
        c.drawString(30, 650, "Trajectory map")
        print("Map added!")
    
        c.showPage()
    else:
        img_preview_y = 650 # If trajectory map is not printing, we draw thumbnails on the first page

    # Thumbnails
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, img_preview_y, "Images previews")

    list_of_images = [img for img in list(Path(session_path, "DCIM").iterdir()) if img.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    selected_images = evenly_select_images_on_interval(list_of_images)
    print("Images previews selected!\n")
    x_coord = 30
    y_coord = img_preview_y - 38 # max_height = 692. Initially Images preview is draw at 730, but if draw at 650, substract 38 to get same difference 

    for i, image in enumerate(selected_images):
        if i % 5 == 0 and i != 0:
            # Start a new row of images
            x_coord = 30
            y_coord -= 110

        img = Image.open(image)
        img.thumbnail((100, 100))

        img_width, img_height = img.size

        temp_image_path = os.path.join(session_path, f'temp_{i}.jpg')
        img.save(temp_image_path)

        if y_coord - img_height < 50:
            c.showPage()
            y_coord = max_height

        c.drawImage(temp_image_path, x_coord, y_coord - img_height)

        os.remove(temp_image_path)

        x_coord += 110

    c.showPage()
    c.setPageSize(landscape(letter))

    # Metadata preview
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, 530, "Metadata preview")
    print("Loading data for metadata preview...")
  
    df = pd.read_csv(metadata_path)
    print("Data loaded!")
    preview_df = df.head(20)
    print("Preview dataframe created!")

    keys = [key for key in ["FileName", "photo_identifier", "GPSDateTime", "SubSecDateTimeOriginal", "GPSLatitude", "GPSLongitude", "FileSize", "ImageHeight", "ImageWidth"] if key in preview_df]
    try:
        preview_df = preview_df[keys]
    except KeyError:
        print("[ERROR] No key to merge gps information in metadata.")

    print("Creation of the PDF table...")
    all_cols = list(df.columns)
    table_data = [list(preview_df.columns)] + preview_df.values.tolist()
    table = Table(table_data)

    table.setStyle([
        ('TEXTCOLOR', (0, 0), (-1, 0), (0, 0, 1)),  # Header row text color (blue)
        ('FONTSIZE', (0, 1), (-1, -1), 8), # Font size of all cells
        ('BACKGROUND', (0, 0), (-1, 0), (0.7, 0.7, 0.7)),  # Header row background color (gray)
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center align all cells
        ('INNERGRID', (0, 0), (-1, -1), 0.25, (0, 0, 0)),  # Inner gridlines
        ('BOX', (0, 0), (-1, -1), 0.25, (0, 0, 0)),  # Cell borders
    ])

    table.wrapOn(c, 10, 20)
    table.drawOn(c, 30, 100)
    print("PDF table sucessfully created!")
    
    text = c.beginText(30, 70)
    text.setFont('Courier', 8)
    line = f"All metadata columns names: {all_cols}"
    wraped_text = "\n".join(wrap(line, 160))
    text.textLines(wraped_text)
    c.drawText(text)
    
    c.setFont("Courier", 8)
    c.drawString(30, 80, f"Total images: {len(df)}")

    c.save()

    # Create predictions images and pdf
    img_folder_predictions_path = create_predictions_map(metadata_path)
    # Can be None if no predictions in csv file (all images useless)
    if img_folder_predictions_path:
        pdf_predictions_path = Path(img_folder_predictions_path, "temp.pdf")
        pred_images_path = [img_name for img_name in natsorted(list(img_folder_predictions_path.iterdir())) if img_name.suffix.lower() == ".jpg"]

        images = [ Image.open(f) for f in pred_images_path ]
        images[0].save(
            pdf_predictions_path, "PDF" ,resolution=200.0, save_all=True, append_images=images[1:]
        )

        # Concat with trajectory pdf
        merger = PdfWriter()
        for pdf in [pdf_file, pdf_predictions_path]:
            merger.append(pdf)
        merger.write(pdf_file)
        merger.close()

        # Delete tmp folder
        for file in Path(img_folder_predictions_path).iterdir():
            file.unlink()
        img_folder_predictions_path.rmdir()
    print("PDF created!")


def main():
    SESSION_FOLDER = Path("/home/bioeos/Documents/Bioeos/aina_dataset")
    if not SESSION_FOLDER.exists() or not SESSION_FOLDER.is_dir():
        print(f"{SESSION_FOLDER} not found")
        return 
    
    for session in sorted(list(SESSION_FOLDER.iterdir())):
        print(f"Working with session {session}")

        metadata_path = Path(session, "METADATA", "metadata.csv")
        if not metadata_path.exists() or not metadata_path.is_file():
            print("Metadata file not found")
            continue
            
        create_pdf_preview(session)

        break


if __name__ == "__main__":
    main()