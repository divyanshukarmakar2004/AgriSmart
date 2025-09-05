#!/usr/bin/env python3
"""
Google Earth Engine Integration for AgriSmart (Live + Demo fallback)
Automated weekly Sentinel-2 data collection for Tamil Nadu potato monitoring
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Optional: Live GEE imports
try:
    import ee  # earthengine-api
    EE_AVAILABLE = True
except Exception:
    EE_AVAILABLE = False

# Optional: env loader
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GEEIntegration:
    def __init__(self, output_dir="outputs/gee_data"):
        """
        Initialize GEE integration with Live mode (project) and Demo fallback

        Args:
            output_dir: Directory to save GEE reports/summary
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Mode and project from env (defaults wired for user's project)
        self.mode = os.getenv("GEE_MODE", "live").lower()
        self.project = os.getenv("GEE_PROJECT", "crested-primacy-471013-r0")

        # Sentinel-2 collection (Surface Reflectance, harmonized)
        # Switch to HARMONIZED to avoid deprecated QA60 band, rely on SCL instead
        self.s2_collection = "COPERNICUS/S2_SR_HARMONIZED"

        # Initialize Earth Engine (Live) or stay in Demo
        if self.is_live():
            try:
                ee.Initialize(project=self.project)
                logger.info(f"✅ Google Earth Engine initialized (Live) project={self.project}")
            except Exception as e:
                logger.warning(f"⚠️ Project-scoped init failed ({e}). Trying default ee.Initialize()...")
                try:
                    ee.Initialize()
                    logger.info("✅ Google Earth Engine initialized (Live, default credentials)")
                except Exception as e2:
                    logger.error(f"❌ Failed to initialize GEE Live mode: {e2}. Falling back to Demo.")
                    self.mode = "demo"

        # Define Tamil Nadu region (T44PKU/T43PGP bounding)
        if self.is_live():
            # Use real geometry in Live mode
            self.tamil_nadu_region = ee.Geometry.Rectangle([77.0, 10.0, 78.5, 11.5])
        else:
            # Minimal bounds holder in Demo mode
            self.tamil_nadu_region = {
                "bounds": [77.0, 10.0, 78.5, 11.5],
                "area_ha": 2500,
            }

        if not self.is_live():
            logger.info("✅ GEE Integration initialized (Demo Mode)")

    def is_live(self) -> bool:
        return EE_AVAILABLE and self.mode == "live"

    def get_weekly_sentinel2_data(self, start_date=None, end_date=None, max_cloud_cover=10, aoi=None):
        """
        Fetch weekly Sentinel-2 composite for Tamil Nadu region.

        Live: returns ee.Image composite with indices bands added
        Demo: returns dict with simulated metadata
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"🌍 Fetching Sentinel-2 data for {start_date} to {end_date}")

        if self.is_live():
            region = aoi if aoi is not None else self.tamil_nadu_region
            collection = (
                ee.ImageCollection(self.s2_collection)
                .filterDate(start_date, end_date)
                .filterBounds(region)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover))
            )

            # Fallback to 30% if no images
            count = collection.size().getInfo()
            if count == 0:
                collection = (
                    ee.ImageCollection(self.s2_collection)
                    .filterDate(start_date, end_date)
                    .filterBounds(region)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                )
                count = collection.size().getInfo()
            logger.info(f"📊 Images found: {count}")
            if count == 0:
                logger.warning("⚠️ No images found. Falling back to Demo mode summary.")
                return self._demo_image_data(start_date, end_date, max_cloud_cover)

            # Cloud mask via SCL + median composite + add indices
            composite = collection.map(self._mask_clouds_scl).median()
            composite = self._add_vegetation_indices_live(composite)
            return composite

        # Demo path
        return self._demo_image_data(start_date, end_date, max_cloud_cover)

    def _demo_image_data(self, start_date, end_date, max_cloud_cover):
        return {
            "collection": self.s2_collection,
            "date_range": f"{start_date} to {end_date}",
            "cloud_cover": max_cloud_cover,
            "region": "Tamil Nadu, India",
            "bands": ["B2", "B3", "B4", "B8", "B5", "B8A"],
            "resolution": "10m",
            "cloud_masked": True,
        }

    def _mask_clouds_scl(self, image):
        # Use SCL classes: exclude 8 (cloud med), 9 (cloud high), 10 (cirrus), 11 (snow)
        scl = image.select("SCL")
        clear = scl.neq(8).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return image.updateMask(clear)

    def _add_vegetation_indices_live(self, image):
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndre = image.normalizedDifference(["B8A", "B5"]).rename("NDRE")
        evi = image.expression(
            "2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)",
            {"NIR": image.select("B8"), "Red": image.select("B4"), "Blue": image.select("B2")},
        ).rename("EVI")
        savi = image.expression(
            "1.5 * (NIR - Red) / (NIR + Red + 0.5)",
            {"NIR": image.select("B8"), "Red": image.select("B4")},
        ).rename("SAVI")
        return image.addBands([ndvi, ndre, evi, savi])

    def get_weekly_summary(self, image_or_data):
        """
        Compute region-wide summary.

        Live: expects ee.Image (with NDVI band)
        Demo: expects dict
        """
        if self.is_live() and isinstance(image_or_data, type(ee.Image())):
            ndvi_stats = (
                image_or_data.select("NDVI")
                .reduceRegion(
                    reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                    geometry=self.tamil_nadu_region,
                    scale=100,
                )
                .getInfo()
            )
            # Coverage area (ha)
            area_ha = ee.Number(self.tamil_nadu_region.area()).divide(10000).getInfo()
            summary = {
                "timestamp": datetime.now().isoformat(),
                "region": "Tamil Nadu, India",
                "coverage_area_ha": round(float(area_ha), 2),
                "ndvi_mean": float(ndvi_stats.get("NDVI_mean", 0.0) or 0.0),
                "ndvi_std": float(ndvi_stats.get("NDVI_stdDev", 0.0) or 0.0),
                "bands_available": ["B2", "B3", "B4", "B8", "B5", "B8A", "NDVI", "NDRE", "EVI", "SAVI"],
                "data_quality": "cloud_masked",
                "data_source": f"Google Earth Engine (Live:{self.project})",
            }
            return summary

        # Demo summary
        ndvi_mean = 0.45 + np.random.normal(0, 0.1)
        ndvi_mean = float(np.clip(ndvi_mean, 0.1, 0.8))
        summary = {
            "timestamp": datetime.now().isoformat(),
            "region": "Tamil Nadu, India",
            "coverage_area_ha": 2500,
            "ndvi_mean": ndvi_mean,
            "ndvi_std": float(0.15 + np.random.normal(0, 0.02)),
            "bands_available": image_or_data.get("bands", []) if isinstance(image_or_data, dict) else [],
            "data_quality": "cloud_masked",
            "data_source": "Google Earth Engine (Demo)",
        }
        return summary

    def integrate_with_agrismart(self, image_or_data):
        logger.info("🔄 Integrating GEE data with AgriSmart pipeline")
        summary = self.get_weekly_summary(image_or_data)

        summary_path = self.output_dir / f"weekly_summary_{datetime.now().strftime('%Y%m%d')}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        integration_report = {
            "gee_integration": True,
            "mode": "live" if self.is_live() else "demo",
            "project": self.project if self.is_live() else None,
            "data_source": summary.get("data_source"),
            "sentinel2_collection": self.s2_collection,
            "cloud_masking": True,
            "vegetation_indices": ["NDVI", "NDRE", "EVI", "SAVI"],
            "weekly_summary": summary,
            "integration_timestamp": datetime.now().isoformat(),
        }

        report_path = self.output_dir / f"gee_integration_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_path, "w") as f:
            json.dump(integration_report, f, indent=2)

        logger.info(f"✅ GEE integration completed. Summary saved to {summary_path}")
        return integration_report


def test_gee_integration():
    try:
        mode = os.getenv("GEE_MODE", "live").lower()
        logger.info(f"🧪 Testing GEE Integration (mode={mode})")
        gee = GEEIntegration()
        image_or_data = gee.get_weekly_sentinel2_data()
        summary = gee.get_weekly_summary(image_or_data)
        gee.integrate_with_agrismart(image_or_data)
        logger.info("✅ GEE integration test completed successfully")
        logger.info(f"📊 NDVI Mean: {summary['ndvi_mean']:.3f}")
        logger.info(f"📊 Coverage Area: {summary['coverage_area_ha']} ha")
        logger.info(f"📊 Data Source: {summary['data_source']}")
        return True
    except Exception as e:
        logger.error(f"❌ GEE integration test failed: {e}")
        return False


if __name__ == "__main__":
    test_gee_integration()
