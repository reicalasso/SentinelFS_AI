"""
Comprehensive example demonstrating the advanced dataset generator features.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the enhanced SentinelFS AI components
from sentinelfs_ai import (
    AdvancedDatasetGenerator,
    AccessPatternConfig,
    UserBehaviorProfile,
    analyze_dataset_distribution,
    visualize_dataset_patterns,
    create_example_dataset
)

def run_advanced_dataset_example():
    """Run comprehensive example of advanced dataset generation."""
    print("=== Advanced Dataset Generator Example ===\n")
    
    # 1. Create generator with custom configuration
    print("1. Creating Advanced Dataset Generator with custom configuration...")
    config = AccessPatternConfig(
        sequence_length=25,  # Longer sequences
        anomaly_complexity_level='high',  # More subtle anomalies
        anomaly_subtlety_factor=0.7,  # Subtle anomalies
        business_hours_start=8,
        business_hours_end=18,
        weekend_access_probability=0.2
    )
    
    generator = AdvancedDatasetGenerator(config=config, random_seed=42)
    print(f"   Configuration: sequence_length={config.sequence_length}, "
          f"anomaly_level={config.anomaly_complexity_level}")
    
    # 2. Define user profile ratios
    print("\n2. Defining user profile distribution...")
    user_profile_ratio = {
        UserBehaviorProfile.DEVELOPER: 0.25,
        UserBehaviorProfile.ANALYST: 0.25,
        UserBehaviorProfile.EXECUTIVE: 0.15,
        UserBehaviorProfile.ADMIN: 0.15,
        UserBehaviorProfile.CONTRACTOR: 0.15,
        UserBehaviorProfile.TEMPORARY: 0.05
    }
    
    print("   User profile distribution:")
    for profile, ratio in user_profile_ratio.items():
        print(f"     {profile.value}: {ratio:.2f}")
    
    # 3. Generate comprehensive dataset
    print("\n3. Generating comprehensive dataset...")
    dataset = generator.generate_dataset(
        num_samples=3000,  # Larger dataset
        anomaly_ratio=0.2,  # 20% anomalies
        user_profile_ratio=user_profile_ratio,
        include_timestamps=True,
        seasonal_variation=True,
        temporal_dependencies=True
    )
    
    print(f"   Generated dataset with shape: {dataset['data'].shape}")
    print(f"   Labels shape: {dataset['labels'].shape}")
    print(f"   Feature names: {len(dataset['feature_names'])} features")
    
    # 4. Analyze the generated dataset
    print("\n4. Analyzing generated dataset...")
    analysis = analyze_dataset_distribution(dataset)
    print(f"   Total samples: {analysis['total_samples']}")
    print(f"   Sequence length: {analysis['sequence_length']}")
    print(f"   Anomaly ratio: {analysis['anomaly_ratio']:.3f}")
    print(f"   Normal samples: {analysis['label_distribution']['normal']}")
    print(f"   Anomaly samples: {analysis['label_distribution']['anomaly']}")
    
    # Show feature statistics for first few features
    print("\n   Feature statistics (first 3 features):")
    for i in range(3):
        feat_stats = analysis['feature_statistics'][f'feature_{i}']
        print(f"     Feature {i}: mean={feat_stats['mean']:.2f}, "
              f"std={feat_stats['std']:.2f}, range=[{feat_stats['min']:.2f}, {feat_stats['max']:.2f}]")
    
    # 5. Generate multi-scenario dataset
    print("\n5. Generating multi-scenario dataset...")
    
    scenarios = [
        {
            'num_samples': 1000,
            'anomaly_ratio': 0.15,
            'user_profile_ratio': {
                UserBehaviorProfile.DEVELOPER: 0.3,
                UserBehaviorProfile.ANALYST: 0.3,
                UserBehaviorProfile.ADMIN: 0.4
            },
            'config_overrides': {'sequence_length': 20}
        },
        {
            'num_samples': 800,
            'anomaly_ratio': 0.3,
            'user_profile_ratio': {
                UserBehaviorProfile.EXECUTIVE: 0.7,
                UserBehaviorProfile.CONTRACTOR: 0.3
            },
            'config_overrides': {'sequence_length': 15}
        },
        {
            'num_samples': 700,
            'anomaly_ratio': 0.25,
            'user_profile_ratio': {
                UserBehaviorProfile.TEMPORARY: 0.8,
                UserBehaviorProfile.DEVELOPER: 0.2
            },
            'config_overrides': {'sequence_length': 30}
        }
    ]
    
    multi_scenario_dataset = generator.generate_multi_scenario_dataset(scenarios)
    print(f"   Multi-scenario dataset shape: {multi_scenario_dataset['data'].shape}")
    print(f"   Scenario distribution: {np.bincount(multi_scenario_dataset['scenario_labels'])}")
    
    # 6. Create and analyze example dataset
    print("\n6. Creating example dataset with built-in function...")
    example_dataset = create_example_dataset()
    print(f"   Example dataset shape: {example_dataset['data'].shape}")
    
    # 7. Show sample of generated data
    print("\n7. Sample of first sequence (first 5 time steps):")
    first_sequence = example_dataset['data'][0]
    for t in range(min(5, len(first_sequence))):
        access = first_sequence[t]
        print(f"   Time {t}: [size={access[0]:.2f}, hour={access[1]:.1f}, "
              f"type={int(access[2])}, dow={int(access[3])}, freq={access[4]:.2f}, "
              f"cat={int(access[5])}, vel={access[6]:.2f}]")
    
    # 8. Show label distribution
    labels = example_dataset['labels'].flatten()
    normal_count = int((labels == 0).sum())
    anomaly_count = int((labels == 1).sum())
    print(f"\n8. Label distribution in example dataset:")
    print(f"   Normal: {normal_count} ({normal_count/len(labels):.2%})")
    print(f"   Anomaly: {anomaly_count} ({anomaly_count/len(labels):.2%})")
    
    # 9. Show feature names
    print(f"\n9. Available features ({len(example_dataset['feature_names'])}):")
    for i, name in enumerate(example_dataset['feature_names']):
        print(f"   {i}: {name}")
    
    # 10. Save the dataset
    print("\n10. Saving dataset...")
    save_path = Path("./dataset_output")
    generator.save_dataset(example_dataset, save_path)
    print(f"   Dataset saved to: {save_path}")
    
    # 11. Demonstrate different complexity levels
    print("\n11. Demonstrating different anomaly complexity levels...")
    
    for complexity in ['low', 'medium', 'high']:
        print(f"\n   Generating with {complexity} complexity:")
        temp_config = AccessPatternConfig(
            sequence_length=15,
            anomaly_complexity_level=complexity
        )
        temp_gen = AdvancedDatasetGenerator(config=temp_config, random_seed=42)
        
        temp_dataset = temp_gen.generate_dataset(
            num_samples=200,
            anomaly_ratio=0.2
        )
        
        temp_analysis = analyze_dataset_distribution(temp_dataset)
        print(f"     Anomaly ratio: {temp_analysis['anomaly_ratio']:.3f}")
        print(f"     Normal/Anomaly: {temp_analysis['label_distribution']['normal']}/"
              f"{temp_analysis['label_distribution']['anomaly']}")
    
    print("\n=== Advanced Dataset Generation Example Completed ===")
    print("\nKey features demonstrated:")
    print("- Realistic user behavior profiles (developer, analyst, executive, etc.)")
    print("- Complex attack scenarios (data exfiltration, ransomware, escalation)")
    print("- Temporal dependencies and seasonal variations")
    print("- Configurable complexity levels")
    print("- Multi-scenario dataset generation")
    print("- Comprehensive dataset analysis")
    print("- Proper data persistence and export")


if __name__ == "__main__":
    run_advanced_dataset_example()