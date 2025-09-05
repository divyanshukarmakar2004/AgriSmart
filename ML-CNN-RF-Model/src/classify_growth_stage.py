import numpy as np
import matplotlib.pyplot as plt

def classify_growth_stage(ndvi_array):
    """
    Classify growth stages based on NDVI thresholds.
    Returns an array with stage codes:
        0 -> Early Stage
        1 -> Mid Stage
        2 -> Late Stage
    """
    stages = np.zeros_like(ndvi_array, dtype=np.uint8)
    stages[(ndvi_array >= 0.3) & (ndvi_array < 0.6)] = 1
    stages[ndvi_array >= 0.6] = 2
    return stages

growth_stage_colors = {
    0: [255, 255, 0],   # Yellow
    1: [0, 255, 0],     # Green
    2: [0, 128, 0]      # Dark Green
}

def save_growth_stage_map(stage_array, output_path):
    """Save growth stage array as RGB PNG using predefined colors."""
    rgb_map = np.zeros((stage_array.shape[0], stage_array.shape[1], 3), dtype=np.uint8)
    for stage, color in growth_stage_colors.items():
        rgb_map[stage_array == stage] = color
    plt.figure(figsize=(10,10))
    plt.imshow(rgb_map)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved growth stage map: {output_path}")
