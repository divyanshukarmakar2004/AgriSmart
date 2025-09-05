import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling

def compute_ndvi(nir, red):
    """
    Compute Normalized Difference Vegetation Index (NDVI)
    
    Args:
        nir: Near-infrared band (B08)
        red: Red band (B04)
    
    Returns:
        NDVI array
    """
    return (nir - red) / (nir + red + 1e-10)

def compute_ndre(nir, red_edge):
    """
    Compute Normalized Difference Red Edge Index (NDRE)
    
    Args:
        nir: Near-infrared band (B08)
        red_edge: Red edge band (B05)
    
    Returns:
        NDRE array
    """
    return (nir - red_edge) / (nir + red_edge + 1e-10)

def cloud_mask_scl(scl_path, target_shape=None):
    """
    Create cloud mask using Scene Classification Layer (SCL)
    
    Args:
        scl_path: Path to SCL band (20m resolution)
        target_shape: Target shape for resampling (optional)
    
    Returns:
        Binary mask where True = clear pixels, False = cloudy pixels
    """
    try:
        with rasterio.open(scl_path) as src:
            if target_shape:
                # Resample SCL to match target resolution
                scl_data = src.read(1, out_shape=target_shape, resampling=Resampling.nearest)
            else:
                scl_data = src.read(1)
        
        # SCL values for clouds: 8-11
        # 8: Cloud medium probability
        # 9: Cloud high probability  
        # 10: Cirrus
        # 11: Snow
        cloud_mask = (scl_data >= 8) & (scl_data <= 11)
        
        # Return clear pixel mask (inverse of cloud mask)
        return ~cloud_mask
        
    except Exception as e:
        print(f"Warning: Could not load SCL band from {scl_path}: {e}")
        print("Proceeding without cloud masking...")
        return None

def apply_cloud_mask(data, cloud_mask):
    """
    Apply cloud mask to data array
    
    Args:
        data: Input data array
        cloud_mask: Binary mask (True = clear, False = cloudy)
    
    Returns:
        Masked data array with NaN for cloudy pixels
    """
    if cloud_mask is None:
        return data
    
    masked_data = data.copy()
    masked_data[~cloud_mask] = np.nan
    return masked_data

def compute_indices_with_cloud_masking(bands, scl_path=None):
    """
    Compute vegetation indices with optional cloud masking
    
    Args:
        bands: Dictionary containing band arrays
        scl_path: Path to SCL band for cloud masking
    
    Returns:
        Dictionary with NDVI, NDRE, and cloud mask
    """
    # Get cloud mask if SCL is available
    cloud_mask = None
    if scl_path:
        cloud_mask = cloud_mask_scl(scl_path, target_shape=bands['B08'].shape)
    
    # Compute NDVI
    ndvi = compute_ndvi(bands['B08'], bands['B04'])
    if cloud_mask is not None:
        ndvi = apply_cloud_mask(ndvi, cloud_mask)
    
    # Compute NDRE
    ndre = compute_ndre(bands['B08'], bands['B05'])
    if cloud_mask is not None:
        ndre = apply_cloud_mask(ndre, cloud_mask)
    
    return {
        'ndvi': ndvi,
        'ndre': ndre,
        'cloud_mask': cloud_mask
    }

def save_index_map(index_array, output_path, title, cmap='RdYlGn', vmin=-1, vmax=1):
    """
    Save vegetation index map as PNG with cloud masking support
    
    Args:
        index_array: Index array to visualize
        output_path: Output file path
        title: Plot title
        cmap: Colormap
        vmin, vmax: Value range for colormap
    """
    plt.figure(figsize=(12, 10))
    
    # Handle NaN values (cloudy pixels)
    if np.any(np.isnan(index_array)):
        # Create a masked array for better visualization
        masked_array = np.ma.masked_where(np.isnan(index_array), index_array)
        im = plt.imshow(masked_array, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Add colorbar with NaN handling
        cbar = plt.colorbar(im, extend='both')
        cbar.set_label('Index Value', rotation=270, labelpad=20)
    else:
        im = plt.imshow(index_array, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, label='Index Value')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Pixel X', fontsize=12)
    plt.ylabel('Pixel Y', fontsize=12)
    
    # Add statistics
    valid_pixels = ~np.isnan(index_array)
    if np.any(valid_pixels):
        mean_val = np.nanmean(index_array)
        std_val = np.nanstd(index_array)
        plt.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Index map saved to: {output_path}")
