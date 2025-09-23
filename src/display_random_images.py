import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import random


def display_random_images(df: pd.DataFrame, img_num: int):
    
    random_int = random.sample(range(0, len(df)), img_num)

    map_label = {0: 'Not detected', 1: 'Detected'}

    for i in random_int:
        img = df.iloc[i,0]

        image = Image.open(img)
        lb = df.iloc[i,1]
        mapped_lb = map_label.get(lb)

        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Label: {lb}, Category: {mapped_lb}')