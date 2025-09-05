"""
Random Forest Baseline Model for Growth Stage Classification
Provides a traditional ML baseline for comparison with CNN models
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class RandomForestGrowthStageClassifier:
    """
    Random Forest classifier for potato growth stage classification
    Uses vegetation indices and spectral bands as features
    """
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        Initialize Random Forest classifier
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        self.feature_names = None
        self.is_trained = False
        
    def prepare_features(self, patches_data, patches_labels):
        """
        Prepare features from patch data for Random Forest
        
        Args:
            patches_data: Array of shape (n_patches, height, width, channels)
            patches_labels: Array of shape (n_patches,)
        
        Returns:
            X: Feature matrix (n_patches, n_features)
            y: Target labels (n_patches,)
        """
        n_patches, height, width, channels = patches_data.shape
        
        # Extract features from each patch
        features = []
        
        for i in range(n_patches):
            patch = patches_data[i]
            
            # Statistical features for each channel
            patch_features = []
            
            for ch in range(channels):
                channel_data = patch[:, :, ch]
                
                # Remove NaN values for statistics
                valid_data = channel_data[~np.isnan(channel_data)]
                
                if len(valid_data) > 0:
                    # Statistical features
                    patch_features.extend([
                        np.mean(valid_data),      # Mean
                        np.std(valid_data),       # Standard deviation
                        np.median(valid_data),    # Median
                        np.percentile(valid_data, 25),  # 25th percentile
                        np.percentile(valid_data, 75),  # 75th percentile
                        np.min(valid_data),       # Minimum
                        np.max(valid_data),       # Maximum
                    ])
                else:
                    # If all NaN, use zeros
                    patch_features.extend([0] * 7)
            
            features.append(patch_features)
        
        X = np.array(features)
        y = patches_labels
        
        # Create feature names
        channel_names = ['B02', 'B03', 'B04', 'B08', 'B05']  # Adjust based on your channels
        stat_names = ['mean', 'std', 'median', 'p25', 'p75', 'min', 'max']
        
        self.feature_names = []
        for ch_name in channel_names[:channels]:
            for stat_name in stat_names:
                self.feature_names.append(f"{ch_name}_{stat_name}")
        
        return X, y
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the Random Forest model
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Fraction of data for testing
            random_state: Random seed
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training Random Forest on {X_train.shape[0]} samples...")
        print(f"Feature matrix shape: {X_train.shape}")
        print(f"Number of features: {X_train.shape[1]}")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nRandom Forest Results:")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Early', 'Mid', 'Late']))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance
        self.plot_feature_importance()
        
        return {
            'test_accuracy': accuracy,
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix
        
        Returns:
            predictions: Predicted class labels
            probabilities: Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to display
        """
        if not self.is_trained:
            print("Model must be trained to show feature importance")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f"Top {top_n} Feature Importances")
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [self.feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('outputs/random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to: outputs/random_forest_feature_importance.png")
    
    def save_model(self, filepath):
        """
        Save trained model to file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Random Forest model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from file
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        print(f"Random Forest model loaded from: {filepath}")

def train_random_forest_baseline(patches_data, patches_labels, save_path='models/random_forest_baseline.joblib'):
    """
    Train Random Forest baseline model
    
    Args:
        patches_data: Patch data array
        patches_labels: Patch labels array
        save_path: Path to save the trained model
    
    Returns:
        Trained RandomForestGrowthStageClassifier instance
    """
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    # Initialize classifier
    rf_classifier = RandomForestGrowthStageClassifier(
        n_estimators=200,  # More trees for better performance
        max_depth=15,      # Deeper trees
        random_state=42
    )
    
    # Prepare features
    X, y = rf_classifier.prepare_features(patches_data, patches_labels)
    
    # Train model
    results = rf_classifier.train(X, y)
    
    # Save model
    rf_classifier.save_model(save_path)
    
    return rf_classifier, results

if __name__ == "__main__":
    # Example usage
    print("Random Forest Baseline Model for Growth Stage Classification")
    print("This module provides a traditional ML baseline for comparison with CNN models")
