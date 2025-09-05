import numpy as np
import matplotlib.pyplot as plt

def classify_nutrients(ndvi_array):
    """
    Simple nutrient classification from NDVI:
    0 -> Low fertility
    1 -> Medium fertility
    2 -> High fertility
    """
    fert = np.zeros_like(ndvi_array, dtype=np.uint8)
    fert[(ndvi_array >= 0.3) & (ndvi_array < 0.6)] = 1
    fert[ndvi_array >= 0.6] = 2
    return fert

nutrient_colors = {
    0: [255, 0, 0],   # Red -> Low
    1: [255, 255, 0], # Yellow -> Medium
    2: [0, 255, 0]    # Green -> High
}

def save_nutrient_map(fert_array, output_path):
    rgb_map = np.zeros((fert_array.shape[0], fert_array.shape[1], 3), dtype=np.uint8)
    for level, color in nutrient_colors.items():
        rgb_map[fert_array == level] = color
    plt.figure(figsize=(10,10))
    plt.imshow(rgb_map)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved nutrient map: {output_path}")
