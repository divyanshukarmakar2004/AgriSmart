"""
Agricultural recommendations module for potato crop management.
"""

import numpy as np
from typing import Dict
import json
import os


class AgriRecommendations:
    """Generate agricultural recommendations for potato crops."""
    
    def __init__(self):
        self.growth_stages = {
            0: "Early Stage (Planting to 30 days)",
            1: "Mid Stage (30-60 days)", 
            2: "Late Stage (60+ days)"
        }
        
        self.nutrient_levels = {
            0: "Low Fertility",
            1: "Medium Fertility",
            2: "High Fertility"
        }
        
        # Irrigation recommendations (mm per application, frequency in days)
        self.irrigation_recs = {
            (0, 0): {"amount": 15, "frequency": 2, "method": "Drip irrigation"},
            (0, 1): {"amount": 12, "frequency": 3, "method": "Drip irrigation"},
            (0, 2): {"amount": 10, "frequency": 4, "method": "Drip irrigation"},
            (1, 0): {"amount": 20, "frequency": 2, "method": "Drip irrigation"},
            (1, 1): {"amount": 18, "frequency": 3, "method": "Drip irrigation"},
            (1, 2): {"amount": 15, "frequency": 4, "method": "Drip irrigation"},
            (2, 0): {"amount": 25, "frequency": 2, "method": "Drip irrigation"},
            (2, 1): {"amount": 22, "frequency": 3, "method": "Drip irrigation"},
            (2, 2): {"amount": 20, "frequency": 4, "method": "Drip irrigation"}
        }
        
        # Fertilizer recommendations (kg/ha)
        self.fertilizer_recs = {
            (0, 0): {"N": 150, "P": 80, "K": 120, "type": "NPK 19:19:19"},
            (0, 1): {"N": 120, "P": 60, "K": 100, "type": "NPK 19:19:19"},
            (0, 2): {"N": 100, "P": 40, "K": 80, "type": "NPK 19:19:19"},
            (1, 0): {"N": 180, "P": 100, "K": 140, "type": "NPK 20:20:20"},
            (1, 1): {"N": 150, "P": 80, "K": 120, "type": "NPK 20:20:20"},
            (1, 2): {"N": 120, "P": 60, "K": 100, "type": "NPK 20:20:20"},
            (2, 0): {"N": 200, "P": 120, "K": 160, "type": "NPK 20:20:20"},
            (2, 1): {"N": 170, "P": 100, "K": 140, "type": "NPK 20:20:20"},
            (2, 2): {"N": 140, "P": 80, "K": 120, "type": "NPK 20:20:20"}
        }
        
        self.additional_recs = {
            (0, 0): "Apply organic manure 2 weeks before planting. Monitor soil moisture closely.",
            (0, 1): "Apply compost during planting. Regular soil testing recommended.",
            (0, 2): "Maintain current soil conditions. Monitor for nutrient imbalances.",
            (1, 0): "Increase nitrogen application. Consider foliar feeding.",
            (1, 1): "Maintain balanced nutrition. Watch for pest and disease pressure.",
            (1, 2): "Reduce nitrogen to prevent excessive vegetative growth.",
            (2, 0): "Critical stage - increase potassium for tuber development.",
            (2, 1): "Focus on tuber bulking. Reduce irrigation before harvest.",
            (2, 2): "Optimal conditions. Prepare for harvest timing."
        }

    def get_recommendations(self, growth_stage: int, nutrient_level: int) -> Dict:
        """Get comprehensive recommendations for given growth stage and nutrient level."""
        if growth_stage not in [0, 1, 2] or nutrient_level not in [0, 1, 2]:
            raise ValueError("Growth stage and nutrient level must be 0, 1, or 2")
        
        key = (growth_stage, nutrient_level)
        
        recommendations = {
            "growth_stage": self.growth_stages[growth_stage],
            "nutrient_level": self.nutrient_levels[nutrient_level],
            "irrigation": self.irrigation_recs[key],
            "fertilizer": self.fertilizer_recs[key],
            "additional_notes": self.additional_recs[key],
            "priority": self._get_priority(growth_stage, nutrient_level)
        }
        
        return recommendations

    def _get_priority(self, growth_stage: int, nutrient_level: int) -> str:
        """Determine priority level for recommendations."""
        if nutrient_level == 0:
            return "HIGH - Immediate action required"
        elif growth_stage == 2 and nutrient_level == 0:
            return "CRITICAL - Yield at risk"
        elif growth_stage == 1:
            return "MEDIUM - Monitor closely"
        else:
            return "LOW - Maintain current practices"

    def generate_field_recommendations(self, stage_map: np.ndarray, nutrient_map: np.ndarray) -> Dict:
        """Generate recommendations for an entire field based on classified maps."""
        total_pixels = stage_map.size
        stage_counts = np.bincount(stage_map.flatten(), minlength=3)
        nutrient_counts = np.bincount(nutrient_map.flatten(), minlength=3)
        
        stage_percentages = (stage_counts / total_pixels) * 100
        nutrient_percentages = (nutrient_counts / total_pixels) * 100
        
        dominant_stage = np.argmax(stage_counts)
        dominant_nutrient = np.argmax(nutrient_counts)
        
        primary_recs = self.get_recommendations(dominant_stage, dominant_nutrient)
        
        field_recommendations = {
            "field_statistics": {
                "total_area_pixels": int(total_pixels),
                "growth_stage_distribution": {
                    "early_stage": {"count": int(stage_counts[0]), "percentage": float(stage_percentages[0])},
                    "mid_stage": {"count": int(stage_counts[1]), "percentage": float(stage_percentages[1])},
                    "late_stage": {"count": int(stage_counts[2]), "percentage": float(stage_percentages[2])}
                },
                "nutrient_distribution": {
                    "low_fertility": {"count": int(nutrient_counts[0]), "percentage": float(nutrient_percentages[0])},
                    "medium_fertility": {"count": int(nutrient_counts[1]), "percentage": float(nutrient_percentages[1])},
                    "high_fertility": {"count": int(nutrient_counts[2]), "percentage": float(nutrient_percentages[2])}
                }
            },
            "dominant_conditions": {
                "growth_stage": self.growth_stages[dominant_stage],
                "nutrient_level": self.nutrient_levels[dominant_nutrient]
            },
            "primary_recommendations": primary_recs
        }
        
        return field_recommendations

    def print_recommendations(self, recommendations: Dict):
        """Print formatted recommendations to console."""
        print("\n" + "="*60)
        print("AGRICULTURAL RECOMMENDATIONS")
        print("="*60)
        
        if "field_statistics" in recommendations:
            stats = recommendations["field_statistics"]
            print(f"\nFIELD OVERVIEW:")
            print(f"Total Area: {stats['total_area_pixels']:,} pixels")
            
            print(f"\nGROWTH STAGE DISTRIBUTION:")
            for stage, data in stats["growth_stage_distribution"].items():
                print(f"  {stage.replace('_', ' ').title()}: {data['percentage']:.1f}% ({data['count']:,} pixels)")
            
            print(f"\nNUTRIENT DISTRIBUTION:")
            for nutrient, data in stats["nutrient_distribution"].items():
                print(f"  {nutrient.replace('_', ' ').title()}: {data['percentage']:.1f}% ({data['count']:,} pixels)")
            
            print(f"\nDOMINANT CONDITIONS:")
            print(f"  Growth Stage: {recommendations['dominant_conditions']['growth_stage']}")
            print(f"  Nutrient Level: {recommendations['dominant_conditions']['nutrient_level']}")
            
            print(f"\nPRIMARY RECOMMENDATIONS:")
            primary = recommendations["primary_recommendations"]
            print(f"  Priority: {primary['priority']}")
            print(f"  Irrigation: {primary['irrigation']['amount']}mm every {primary['irrigation']['frequency']} days")
            print(f"  Fertilizer: {primary['fertilizer']['type']} - N:{primary['fertilizer']['N']}, P:{primary['fertilizer']['P']}, K:{primary['fertilizer']['K']} kg/ha")
            print(f"  Notes: {primary['additional_notes']}")
            
        else:
            print(f"\nGROWTH STAGE: {recommendations['growth_stage']}")
            print(f"NUTRIENT LEVEL: {recommendations['nutrient_level']}")
            print(f"PRIORITY: {recommendations['priority']}")
            
            print(f"\nIRRIGATION:")
            irr = recommendations['irrigation']
            print(f"  Amount: {irr['amount']}mm per application")
            print(f"  Frequency: Every {irr['frequency']} days")
            print(f"  Method: {irr['method']}")
            
            print(f"\nFERTILIZER:")
            fert = recommendations['fertilizer']
            print(f"  Type: {fert['type']}")
            print(f"  Nitrogen: {fert['N']} kg/ha")
            print(f"  Phosphorus: {fert['P']} kg/ha")
            print(f"  Potassium: {fert['K']} kg/ha")
            
            print(f"\nADDITIONAL NOTES:")
            print(f"  {recommendations['additional_notes']}")
        
        print("="*60)


# Tamil Nadu Enhanced Recommendations
class TamilNaduRecommendations(AgriRecommendations):
    """Enhanced recommendations system for Tamil Nadu agricultural data."""
    
    def __init__(self):
        super().__init__()
        
        # Tamil Nadu specific growth stages
        self.tamil_nadu_stages = {
            0: "Vegetative",
            1: "Tuber Initiation", 
            2: "Bulking",
            3: "Maturation"
        }
        
        # Enhanced recommendations with Tamil Nadu data
        self.enhanced_irrigation_recs = {
            (0, 0): {"amount": 15, "frequency": 2, "method": "Drip irrigation", "liters_per_ha": 1500},
            (0, 1): {"amount": 12, "frequency": 3, "method": "Drip irrigation", "liters_per_ha": 1500},
            (0, 2): {"amount": 10, "frequency": 3, "method": "Drip irrigation", "liters_per_ha": 1500},
            (1, 0): {"amount": 18, "frequency": 2, "method": "Drip irrigation", "liters_per_ha": 2000},
            (1, 1): {"amount": 15, "frequency": 3, "method": "Drip irrigation", "liters_per_ha": 2000},
            (1, 2): {"amount": 12, "frequency": 3, "method": "Drip irrigation", "liters_per_ha": 2000},
            (2, 0): {"amount": 20, "frequency": 2, "method": "Drip irrigation", "liters_per_ha": 2500},
            (2, 1): {"amount": 18, "frequency": 3, "method": "Drip irrigation", "liters_per_ha": 2500},
            (2, 2): {"amount": 15, "frequency": 3, "method": "Drip irrigation", "liters_per_ha": 2500},
            (3, 0): {"amount": 12, "frequency": 3, "method": "Drip irrigation", "liters_per_ha": 1800},
            (3, 1): {"amount": 10, "frequency": 4, "method": "Drip irrigation", "liters_per_ha": 1800},
            (3, 2): {"amount": 8, "frequency": 4, "method": "Drip irrigation", "liters_per_ha": 1800}
        }
        
        # Weather-specific adjustments
        self.weather_adjustments = {
            'Dry': {"irrigation_multiplier": 1.2, "advice": "Apply mulch to retain soil moisture"},
            'Hot': {"irrigation_multiplier": 1.1, "advice": "Irrigate in early morning or evening"},
            'Moderate Rain': {"irrigation_multiplier": 0.7, "advice": "Monitor for waterlogging and disease"},
            'Cloudy': {"irrigation_multiplier": 1.0, "advice": "Monitor for fungal diseases"}
        }
    
    def get_enhanced_recommendations(self, growth_stage: int, nutrient_level: int, 
                                   weather_condition: str = None, ndvi_value: float = None) -> Dict:
        """Get enhanced recommendations with Tamil Nadu data."""
        
        # Get base recommendations
        base_rec = self.get_recommendations(growth_stage, nutrient_level)
        
        # Add Tamil Nadu specific information
        if growth_stage in self.tamil_nadu_stages:
            base_rec["tamil_nadu_stage"] = self.tamil_nadu_stages[growth_stage]
        
        # Add enhanced irrigation info
        if (growth_stage, nutrient_level) in self.enhanced_irrigation_recs:
            enhanced_irr = self.enhanced_irrigation_recs[(growth_stage, nutrient_level)]
            base_rec["enhanced_irrigation"] = enhanced_irr
        
        # Add weather adjustments
        if weather_condition and weather_condition in self.weather_adjustments:
            weather_adj = self.weather_adjustments[weather_condition]
            base_rec["weather_adjustment"] = {
                "condition": weather_condition,
                "irrigation_multiplier": weather_adj["irrigation_multiplier"],
                "advice": weather_adj["advice"]
            }
        
        # Add NDVI-based recommendations
        if ndvi_value is not None:
            if ndvi_value < 0.4:
                base_rec["ndvi_advice"] = "Low vegetation vigor - check for nutrient deficiency or disease"
            elif ndvi_value > 0.7:
                base_rec["ndvi_advice"] = "High vegetation vigor - monitor for excessive growth"
            else:
                base_rec["ndvi_advice"] = "Normal vegetation vigor - maintain current practices"
        
        return base_rec


# Convenience functions
def get_recommendations(growth_stage: int, nutrient_level: int) -> Dict:
    """Get recommendations for specific growth stage and nutrient level."""
    recommender = AgriRecommendations()
    return recommender.get_recommendations(growth_stage, nutrient_level)


def get_enhanced_recommendations(growth_stage: int, nutrient_level: int, 
                               weather_condition: str = None, ndvi_value: float = None) -> Dict:
    """Get enhanced recommendations with Tamil Nadu data."""
    recommender = TamilNaduRecommendations()
    return recommender.get_enhanced_recommendations(growth_stage, nutrient_level, weather_condition, ndvi_value)


def generate_field_recommendations(stage_map: np.ndarray, nutrient_map: np.ndarray) -> Dict:
    """Generate field-level recommendations from classification maps."""
    recommender = AgriRecommendations()
    return recommender.generate_field_recommendations(stage_map, nutrient_map)