import torch
import torch.nn as nn
import random
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, Dinov2ForImageClassification, Dinov2Config


IMAGE_FOLDER = Path("dataset/test_folder")
CSV_FOLDER_OUTPUT = Path("csv/model_output")
CPCE_GRID = 7
MODEL_NAME = "groderg/Aina-large-2024_10_23-batch-size32_freeze_monolabel"
PATH_TO_MONOLABEL_DIRECTORY = "models/monolabel"

def getDynoConfig(repo_name):
    repo_path = Path(Path.cwd(), PATH_TO_MONOLABEL_DIRECTORY, repo_name)
    if not Path.exists(repo_path):
        snapshot_download(repo_id=repo_name, local_dir=Path(Path.cwd(), PATH_TO_MONOLABEL_DIRECTORY, repo_name))

    config = None
    with open(Path(repo_path, "config.json")) as f:
        config = json.load(f)
    
    return config

class NewHeadDinoV2ForImageClassification(Dinov2ForImageClassification):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)

        # Classifier head
        self.classifier = self.create_head(config.hidden_size * 2, config.num_labels)
    
    # CREATE CUSTOM MODEL
    def create_head(self, num_features , number_classes ,dropout_prob=0.5 ,activation_func = nn.ReLU):
        features_lst = [num_features , num_features//2 , num_features//4]
        layers = []
        for in_f ,out_f in zip(features_lst[:-1] , features_lst[1:]):
            layers.append(nn.Linear(in_f , out_f))
            layers.append(activation_func())
            layers.append(nn.BatchNorm1d(out_f))
            if dropout_prob != 0 : layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(features_lst[-1] , number_classes))
        return nn.Sequential(*layers)

def softmax(x):
    """Compute the softmax of a list of numbers."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum(axis=0)

class MonolabelModel():
    """Pipeline to identify mulitple class in image"""
    def __init__(self, repo_name, batch_size):

        self.image_processor = AutoImageProcessor.from_pretrained(repo_name)
        self.config = getDynoConfig(repo_name)
        self.classes_name = list(self.config["label2id"].keys())
        self.batch_size = batch_size
        self.model = NewHeadDinoV2ForImageClassification.from_pretrained(repo_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.colors = self.build_color_map_for_label()


    def build_color_map_for_label(self):
        def random_color():
            return "#{:06x}".format(random.randint(0, 0xFFFFFF))

        return {label: random_color() for label in self.classes_name}

    def predict(self, frame):
        
        inputs = self.image_processor(frame, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            model_outputs = self.model(**inputs)
        
        logit = model_outputs["logits"][0]
        scores = softmax(logit.cpu().numpy())
        
        best_score = max(scores)
        class_name = self.classes_name[list(scores).index(best_score)]
        return class_name

def vizualize(im: Image, centers: list | dict, colors: dict | None):

    im_draw = ImageDraw.Draw(im)

    if isinstance(centers, list):
        for center in centers:
            im_draw.circle(center, 5, fill=(255,255,255))
    elif isinstance(centers, dict):
        for center in centers:
            color = colors.get(centers.get(center), (255,255,255)) if colors else (255,255,255)
            im_draw.circle(center, 5, fill=color)
    
    im.show()

def generate_centers(width: int, height: int) -> list[tuple[float, float]]:
    if CPCE_GRID == 1:
        return [(width // 2, height // 2)]
    
    if CPCE_GRID == 2:
        return [(0, 0),(width, 0), (0, height), (width, height)]

    w_step, h_step = width//(CPCE_GRID-1), height//(CPCE_GRID-1)
    return [(i, j) for i in range(0, width+1, w_step) for j in range(0, height+1, h_step)]



def generate_thumbnails(im: Image, centers: list):

    map_thumbnail_by_center =  {}
    thumb_half_width, thumb_half_height = im.width // 10, im.height // 10

    for cx, cy in centers:
        left = max(cx - thumb_half_width, 0)
        top = max(cy - thumb_half_height, 0)
        right = min(cx + thumb_half_width, im.width)
        bottom = min(cy + thumb_half_height, im.height)

        map_thumbnail_by_center[(cx, cy)] = im.crop((left, top, right, bottom))

    return map_thumbnail_by_center


def load_images_from_folder(images_folder: Path) -> list[Path]:
    """ Load images from a folder of images. """
    list_images = []        
    
    if not images_folder.exists() or not images_folder.is_dir():
        raise NameError(f"Cannot found {images_folder}")


    for img_path in images_folder.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in (".jpeg", ".jpg", ".png"): continue

        # PIL open image without loading it in memory. We just test if image is not corrupt.
        try:
            with Image.open(img_path) as f:
                list_images.append(img_path)
        except:
            continue
    
    return list_images


def main():
    # Get images
    list_images = load_images_from_folder(IMAGE_FOLDER)

    model_manager = MonolabelModel(MODEL_NAME, batch_size=1)

    for img in list_images:
        im = Image.open(img)

        centers = generate_centers(im.width, im.height)
        map_thumbnails_by_center = generate_thumbnails(im, centers) 

        centers_with_pred = {}
        for center in map_thumbnails_by_center:
            label = model_manager.predict(map_thumbnails_by_center.get(center))
            centers_with_pred[center] = label
        
        vizualize(im, centers_with_pred, model_manager.colors)




if __name__ == "__main__":
    main()