"""
End-to-End Pipeline for AgriSmart Agricultural Monitoring System
Orchestrates data processing, model training, and recommendation generation
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
from datetime import datetime

# Import our modules
from .read_band import read_band
from .compute_indices import compute_indices_with_cloud_masking, save_index_map
from .classify_growth_stage import classify_growth_stage, save_growth_stage_map
from .nutrient_mapping import classify_nutrients, save_nutrient_map
from .recommendations import generate_field_recommendations
from .utils import ensure_dir

# Import training modules
import sys
sys.path.append('training')
from training.data_loader import AgriDataLoader, save_patched_data
from training.train_cnn import train_model

# Import models
sys.path.append('models')
from models.cnn_growth_stage import build_cnn_model, build_attention_cnn_model
from models.random_forest import train_random_forest_baseline

class AgriSmartPipeline:
    """
    End-to-end pipeline for agricultural monitoring using Sentinel-2 data
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize pipeline with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.results = {}
        
        # Create output directories
        ensure_dir(self.config['paths']['outputs'])
        ensure_dir(self.config['paths']['models'])
        ensure_dir('data/patches')
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"Configuration loaded from: {config_path}")
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails"""
        return {
            'paths': {
                'img_data': 'notebooks/IMG_DATA/',
                'outputs': 'outputs/',
                'models': 'models/',
                'data_patches': 'data/patches/patched_data.npz'
            },
            'sentinel2': {
                'tile_id': 'T44PLV_20250827T045721',
                'bands': {
                    'B02': 'R10m/T44PLV_20250827T045721_B02_10m.jp2',
                    'B03': 'R10m/T44PLV_20250827T045721_B03_10m.jp2',
                    'B04': 'R10m/T44PLV_20250827T045721_B04_10m.jp2',
                    'B08': 'R10m/T44PLV_20250827T045721_B08_10m.jp2',
                    'B05': 'R20m/T44PLV_20250827T045721_B05_20m.jp2'
                },
                'cloud_mask_band': 'R20m/T44PLV_20250827T045721_SCL_20m.jp2'
            },
            'thresholds': {
                'ndvi_early_mid': 0.3,
                'ndvi_mid_late': 0.6,
                'nutrient_low_medium': 0.3,
                'nutrient_medium_high': 0.6
            },
            'cnn': {
                'epochs': 50,
                'batch_size': 32,
                'model_type': 'attention',
                'input_shape': [256, 256, 5],
                'num_classes': 3,
                'augment_data': True,
                'test_size': 0.2
            }
        }
    
    def step1_data_ingestion(self) -> Dict[str, np.ndarray]:
        """
        Step 1: Load and preprocess Sentinel-2 bands
        
        Returns:
            Dictionary containing band arrays
        """
        print("\n" + "="*60)
        print("STEP 1: DATA INGESTION")
        print("="*60)
        
        bands = {}
        img_data_path = self.config['paths']['img_data']
        tile_id = self.config['sentinel2']['tile_id']
        
        print(f"Loading Sentinel-2 data for tile: {tile_id}")
        print(f"Data path: {img_data_path}")
        
        # Load each band
        ref_profile = None
        for band_name, band_path in self.config['sentinel2']['bands'].items():
            full_path = os.path.join(img_data_path, band_path)
            print(f"Loading {band_name} from: {full_path}")
            try:
                if band_name == 'B05':
                    # Resample 20m Red Edge to 10m using B02 as reference
                    if 'B02' not in bands:
                        raise RuntimeError("B02 must be loaded before B05 for resampling reference")
                    data_b05, _ = read_band(
                        full_path,
                        out_shape=bands['B02'].shape,
                        ref_profile=ref_profile
                    )
                    bands[band_name] = data_b05
                else:
                    data, profile = read_band(full_path)
                    bands[band_name] = data
                    if band_name == 'B02':
                        ref_profile = profile
                print(f"  ✓ {band_name} loaded: {bands[band_name].shape}")
            except Exception as e:
                print(f"  ✗ Error loading {band_name}: {e}")
                # Create dummy data if band is missing
                bands[band_name] = np.zeros((10980, 10980), dtype=np.float32)
                print(f"  ⚠ Using dummy data for {band_name}")
        
        self.results['bands'] = bands
        print(f"\n✓ Data ingestion completed. Loaded {len(bands)} bands.")
        return bands
    
    def step2_cloud_masking_and_indices(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Step 2: Apply cloud masking and compute vegetation indices
        
        Args:
            bands: Dictionary containing band arrays
            
        Returns:
            Dictionary containing indices and cloud mask
        """
        print("\n" + "="*60)
        print("STEP 2: CLOUD MASKING & VEGETATION INDICES")
        print("="*60)
        
        # Get SCL path for cloud masking
        scl_path = None
        if 'cloud_mask_band' in self.config['sentinel2']:
            scl_path = os.path.join(
                self.config['paths']['img_data'],
                self.config['sentinel2']['cloud_mask_band']
            )
            print(f"Using SCL for cloud masking: {scl_path}")
        
        # Compute indices with cloud masking
        indices = compute_indices_with_cloud_masking(bands, scl_path)
        
        # Save index maps
        outputs_path = self.config['paths']['outputs']
        
        # NDVI map
        ndvi_path = os.path.join(outputs_path, 'NDVI_cloud_masked.png')
        save_index_map(
            indices['ndvi'], 
            ndvi_path, 
            'NDVI with Cloud Masking',
            cmap='RdYlGn',
            vmin=-1, vmax=1
        )
        
        # NDRE map
        ndre_path = os.path.join(outputs_path, 'NDRE_cloud_masked.png')
        save_index_map(
            indices['ndre'], 
            ndre_path, 
            'NDRE with Cloud Masking',
            cmap='RdYlGn',
            vmin=-1, vmax=1
        )
        
        # Cloud mask visualization
        if indices['cloud_mask'] is not None:
            cloud_path = os.path.join(outputs_path, 'cloud_mask.png')
            save_index_map(
                indices['cloud_mask'].astype(float), 
                cloud_path, 
                'Cloud Mask (White=Clear, Black=Cloudy)',
                cmap='gray',
                vmin=0, vmax=1
            )
        
        self.results['indices'] = indices
        print(f"\n✓ Cloud masking and index computation completed.")
        print(f"  NDVI range: {np.nanmin(indices['ndvi']):.3f} to {np.nanmax(indices['ndvi']):.3f}")
        print(f"  NDRE range: {np.nanmin(indices['ndre']):.3f} to {np.nanmax(indices['ndre']):.3f}")
        
        return indices
    
    def step3_classification(self, indices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Step 3: Classify growth stages and nutrient levels
        
        Args:
            indices: Dictionary containing vegetation indices
            
        Returns:
            Dictionary containing classification maps
        """
        print("\n" + "="*60)
        print("STEP 3: GROWTH STAGE & NUTRIENT CLASSIFICATION")
        print("="*60)
        
        # Get thresholds from config (handle different config formats)
        if 'thresholds' in self.config:
            thresholds = self.config['thresholds']
        elif 'indices' in self.config:
            # Convert indices config to thresholds format
            thresholds = {
                'ndvi_early_mid': self.config['indices']['ndvi']['low_threshold'],
                'ndvi_mid_late': self.config['indices']['ndvi']['mid_threshold'],
                'nutrient_low_medium': self.config['indices']['ndvi']['low_threshold'],
                'nutrient_medium_high': self.config['indices']['ndvi']['mid_threshold']
            }
        else:
            # Use default thresholds
            thresholds = {
                'ndvi_early_mid': 0.3,
                'ndvi_mid_late': 0.6,
                'nutrient_low_medium': 0.3,
                'nutrient_medium_high': 0.6
            }
        outputs_path = self.config['paths']['outputs']
        
        # Classify growth stages
        print("Classifying growth stages...")
        growth_stages = classify_growth_stage(indices['ndvi'])
        
        # Classify nutrient levels
        print("Classifying nutrient levels...")
        nutrient_levels = classify_nutrients(indices['ndvi'])
        
        # Save classification maps
        growth_path = os.path.join(outputs_path, 'Growth_Stages_Enhanced.png')
        save_growth_stage_map(growth_stages, growth_path)
        
        nutrient_path = os.path.join(outputs_path, 'Nutrient_Map_Enhanced.png')
        save_nutrient_map(nutrient_levels, nutrient_path)
        
        # Print statistics
        unique_stages, stage_counts = np.unique(growth_stages, return_counts=True)
        unique_nutrients, nutrient_counts = np.unique(nutrient_levels, return_counts=True)
        
        print(f"\nGrowth Stage Distribution:")
        stage_names = ['Early', 'Mid', 'Late']
        for stage, count in zip(unique_stages, stage_counts):
            print(f"  {stage_names[stage]}: {count:,} pixels ({count/len(growth_stages.flatten())*100:.1f}%)")
        
        print(f"\nNutrient Level Distribution:")
        nutrient_names = ['Low', 'Medium', 'High']
        for nutrient, count in zip(unique_nutrients, nutrient_counts):
            print(f"  {nutrient_names[nutrient]}: {count:,} pixels ({count/len(nutrient_levels.flatten())*100:.1f}%)")
        
        classifications = {
            'growth_stages': growth_stages,
            'nutrient_levels': nutrient_levels
        }
        
        self.results['classifications'] = classifications
        print(f"\n✓ Classification completed.")
        
        return classifications
    
    def step4_data_preparation(self, bands: Dict[str, np.ndarray], 
                             indices: Dict[str, np.ndarray],
                             classifications: Dict[str, np.ndarray]) -> str:
        """
        Step 4: Prepare data for machine learning training
        
        Args:
            bands: Band arrays
            indices: Vegetation indices
            classifications: Classification maps
            
        Returns:
            Path to prepared data file
        """
        print("\n" + "="*60)
        print("STEP 4: DATA PREPARATION FOR ML")
        print("="*60)
        
        print("Creating patches for CNN training...")
        
        # Create dataset with enhanced features
        # Get dataset path from config (handle different config formats)
        if 'data_patches' in self.config['paths']:
            dataset_path = self.config['paths']['data_patches']
        elif 'patches_data' in self.config['paths']:
            dataset_path = os.path.join(self.config['paths']['patches_data'], 'patched_data.npz')
        else:
            dataset_path = 'data/patches/patched_data.npz'
        
        try:
            # Build multi-channel array from available inputs (bands + indices)
            channel_list = []
            channel_names = []
            
            # Add commonly used bands if present
            for band_name in [
                'B02','B03','B04','B08','B05','B11','B12'
            ]:
                if band_name in bands and isinstance(bands[band_name], np.ndarray):
                    channel_list.append(bands[band_name])
                    channel_names.append(band_name)
            
            # Add indices if present
            for idx_name in ['NDVI','NDRE']:
                if idx_name in indices and isinstance(indices[idx_name], np.ndarray):
                    channel_list.append(indices[idx_name])
                    channel_names.append(idx_name)
            
            if not channel_list:
                raise ValueError("No valid bands/indices available to create patches")
            
            # Stack into HxWxC
            multi_channel_data = np.stack(channel_list, axis=-1)
            print(f"Channels for patching: {channel_names}")
            
            # Labels: prefer growth stages if available
            labels = classifications.get('growth_stages')
            if labels is None or not isinstance(labels, np.ndarray):
                print("Warning: growth_stages not found; creating random labels for demo")
                labels = np.random.randint(0, 3, multi_channel_data.shape[:2])
            
            # Create patches using AgriDataLoader
            patch_size = 256
            loader = AgriDataLoader(
                data_path=dataset_path,
                patch_size=patch_size
            )
            patches, patch_labels = loader.create_patches(
                multi_channel_data, labels, overlap=0.1
            )
            print(f"Created {len(patches)} patches for training")
            
            # One-hot encode labels for CNNs expecting categorical
            num_classes = int(np.max(patch_labels)) + 1 if len(patch_labels) else 3
            patch_labels_onehot = np.eye(num_classes)[patch_labels]
            
            # Save
            save_patched_data(patches, patch_labels_onehot, dataset_path)
            print(f"✓ Data preparation completed. Dataset saved to: {dataset_path}")
            return dataset_path
        except Exception as e:
            print(f"Error in data preparation: {e}")
            print("Creating sample data for demonstration...")
            self._create_sample_data(dataset_path)
            return dataset_path
    
    def step5_model_training(self, dataset_path: str) -> Dict[str, Any]:
        """
        Step 5: Train machine learning models
        
        Args:
            dataset_path: Path to prepared dataset
            
        Returns:
            Training results
        """
        print("\n" + "="*60)
        print("STEP 5: MODEL TRAINING")
        print("="*60)
        
        cnn_config = self.config['cnn']
        model_type = cnn_config['model_type']
        
        print(f"Training {model_type} CNN model...")
        print(f"Configuration: {cnn_config}")
        
        # Train CNN model
        cnn_results = train_model(
            data_path=dataset_path,
            model_type=model_type,
            epochs=cnn_config['epochs'],
            batch_size=cnn_config['batch_size'],
            test_size=cnn_config['test_size']
        )
        
        # Train Random Forest baseline
        print("\nTraining Random Forest baseline...")
        try:
            # Load data for Random Forest
            data = np.load(dataset_path)
            # Our saved dataset uses keys 'data' and 'labels'
            patches_data = data['data']
            patches_labels = data['labels']
            rf_model, rf_results = train_random_forest_baseline(
                patches_data, patches_labels,
                save_path='models/random_forest_baseline.joblib'
            )
            print(f"✓ Random Forest training completed. Accuracy: {rf_results['test_accuracy']:.4f}")
        except Exception as e:
            print(f"Random Forest training failed: {e}")
            rf_results = None
        
        training_results = {
            'cnn': cnn_results,
            'random_forest': rf_results
        }
        
        self.results['training'] = training_results
        print(f"\n✓ Model training completed.")
        
        return training_results
    
    def step6_recommendations(self, classifications: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Step 6: Generate agricultural recommendations
        
        Args:
            classifications: Classification maps
            
        Returns:
            Recommendations dictionary
        """
        print("\n" + "="*60)
        print("STEP 6: AGRICULTURAL RECOMMENDATIONS")
        print("="*60)
        
        try:
            # Generate field-level recommendations
            recommendations = generate_field_recommendations(
                classifications['growth_stages'],
                classifications['nutrient_levels']
            )
            
            # Save recommendations
            outputs_path = self.config['paths']['outputs']
            rec_path = os.path.join(outputs_path, 'recommendations.json')
            
            with open(rec_path, 'w') as f:
                json.dump(recommendations, f, indent=2)
            
            print(f"✓ Recommendations generated and saved to: {rec_path}")
            
            # Print summary (use safe defaults if keys are missing)
            print(f"\nField Summary:")
            print(f"  Dominant Growth Stage: {recommendations.get('dominant_growth_stage', 'N/A')}")
            print(f"  Dominant Nutrient Level: {recommendations.get('dominant_nutrient_level', 'N/A')}")
            health_score = recommendations.get('health_score')
            if isinstance(health_score, (int, float)):
                print(f"  Field Health Score: {health_score:.2f}/10")
            else:
                print(f"  Field Health Score: N/A")
            
            self.results['recommendations'] = recommendations
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return {}
    
    def step7_output_generation(self):
        """
        Step 7: Generate final outputs and reports
        """
        print("\n" + "="*60)
        print("STEP 7: OUTPUT GENERATION")
        print("="*60)
        
        # Create summary report
        self._create_summary_report()
        
        # Create comparison plots
        self._create_comparison_plots()
        
        print(f"\n✓ All outputs generated in: {self.config['paths']['outputs']}")
    
    def _create_sample_data(self, output_path: str):
        """Create sample data for demonstration"""
        print("Creating sample data for demonstration...")
        
        # Create sample patches
        n_patches = 200
        patch_size = 256
        n_channels = 5
        
        patches = np.random.rand(n_patches, patch_size, patch_size, n_channels).astype(np.float32)
        labels = np.random.randint(0, 3, n_patches)
        
        # Convert to one-hot encoding
        labels_onehot = np.eye(3)[labels]
        
        np.savez(output_path, data=patches, labels=labels_onehot)
        print(f"Sample data created: {n_patches} patches")
    
    def _create_summary_report(self):
        """Create a summary report of the pipeline execution"""
        report_path = os.path.join(self.config['paths']['outputs'], 'pipeline_summary.json')
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results_summary': {
                'data_ingestion': 'completed' if 'bands' in self.results else 'failed',
                'cloud_masking': 'completed' if 'indices' in self.results else 'failed',
                'classification': 'completed' if 'classifications' in self.results else 'failed',
                'model_training': 'completed' if 'training' in self.results else 'failed',
                'recommendations': 'completed' if 'recommendations' in self.results else 'failed'
            }
        }
        
        # Add performance metrics if available
        if 'training' in self.results and self.results['training']:
            training_data = self.results['training']
            summary['performance'] = {}
            
            # Handle CNN results (could be dict or tuple)
            if 'cnn' in training_data and training_data['cnn']:
                cnn_results = training_data['cnn']
                if isinstance(cnn_results, dict):
                    summary['performance']['cnn_accuracy'] = cnn_results.get('test_accuracy', 'N/A')
                else:
                    summary['performance']['cnn_accuracy'] = 'N/A'
            
            # Handle Random Forest results
            if 'random_forest' in training_data and training_data['random_forest']:
                rf_results = training_data['random_forest']
                if isinstance(rf_results, dict):
                    summary['performance']['random_forest_accuracy'] = rf_results.get('test_accuracy', 'N/A')
                else:
                    summary['performance']['random_forest_accuracy'] = 'N/A'
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary report saved to: {report_path}")
    
    def _create_comparison_plots(self):
        """Create comparison plots for different models"""
        if 'training' not in self.results:
            return
        
        # Create model comparison plot
        plt.figure(figsize=(12, 8))
        
        models = ['CNN Basic', 'CNN Attention']
        accuracies = []
        
        if self.results['training']['cnn']:
            cnn_entry = self.results['training']['cnn']
            if isinstance(cnn_entry, dict):
                accuracies.append(cnn_entry.get('test_accuracy', 0))
            elif isinstance(cnn_entry, tuple) and len(cnn_entry) >= 2:
                # cnn_entry = (model, history)
                history = cnn_entry[1]
                try:
                    val_acc = history.history.get('val_accuracy', [])
                    accuracies.append(max(val_acc) if val_acc else 0)
                except Exception:
                    accuracies.append(0)
            else:
                accuracies.append(0)
        
        if self.results['training']['random_forest']:
            models.append('Random Forest')
            accuracies.append(self.results['training']['random_forest'].get('test_accuracy', 0))
        
        if accuracies:
            plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'orange'][:len(models)])
            plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
            plt.ylabel('Test Accuracy', fontsize=12)
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for i, v in enumerate(accuracies):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
            
            plt.tight_layout()
            comparison_path = os.path.join(self.config['paths']['outputs'], 'model_comparison.png')
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Model comparison plot saved to: {comparison_path}")
    
    def run_full_pipeline(self, train_models: bool = True) -> Dict[str, Any]:
        """
        Run the complete AgriSmart pipeline
        
        Args:
            train_models: Whether to train ML models
            
        Returns:
            Complete pipeline results
        """
        print("🚀 STARTING AGRISMART PIPELINE")
        print("="*60)
        print("AI-Powered Potato Crop Growth Stage & Nutrient Health Management")
        print("="*60)
        
        try:
            # Step 1: Data Ingestion
            bands = self.step1_data_ingestion()
            
            # Step 2: Cloud Masking & Indices
            indices = self.step2_cloud_masking_and_indices(bands)
            
            # Step 3: Classification
            classifications = self.step3_classification(indices)
            
            # Step 4: Data Preparation
            dataset_path = self.step4_data_preparation(bands, indices, classifications)
            
            # Step 5: Model Training (optional)
            if train_models:
                training_results = self.step5_model_training(dataset_path)
            else:
                print("\n⏭️  Skipping model training (train_models=False)")
                training_results = {}
            
            # Step 6: Recommendations
            recommendations = self.step6_recommendations(classifications)
            
            # Step 7: Output Generation
            self.step7_output_generation()
            
            print("\n" + "="*60)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"All outputs saved to: {self.config['paths']['outputs']}")
            print(f"Trained models saved to: {self.config['paths']['models']}")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return self.results

def main():
    """Main function to run the pipeline"""
    pipeline = AgriSmartPipeline()
    results = pipeline.run_full_pipeline(train_models=True)
    return results

if __name__ == "__main__":
    main()
