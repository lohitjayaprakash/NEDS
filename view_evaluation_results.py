#!/usr/bin/env python3
"""
Script to visualize NEDS evaluation results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path

def load_and_summarize_results(results_dir):
    """Load and summarize evaluation results"""
    results = {}
    
    # Load spike evaluation results
    spike_dir = os.path.join(results_dir, "eval_spike")
    if os.path.exists(spike_dir):
        spike_results = {}
        
        # Load BPS (bits per spike)
        bps_file = os.path.join(spike_dir, "bps.npy")
        if os.path.exists(bps_file):
            spike_results['bps'] = np.load(bps_file, allow_pickle=True)
            
        # Load R2
        r2_file = os.path.join(spike_dir, "r2.npy")
        if os.path.exists(r2_file):
            spike_results['r2'] = np.load(r2_file, allow_pickle=True)
            
        # Load spike data
        spike_data_file = os.path.join(spike_dir, "spike_data.npy")
        if os.path.exists(spike_data_file):
            spike_results['spike_data'] = np.load(spike_data_file, allow_pickle=True)
            
        results['spike'] = spike_results
    
    # Load behavior evaluation results
    behavior_dir = os.path.join(results_dir, "eval_behavior")
    if os.path.exists(behavior_dir):
        behavior_results = {}
        
        # Load accuracy
        acc_file = os.path.join(behavior_dir, "acc.npy")
        if os.path.exists(acc_file):
            behavior_results['acc'] = np.load(acc_file, allow_pickle=True)
            
        # Load R2
        r2_file = os.path.join(behavior_dir, "r2.npy")
        if os.path.exists(r2_file):
            behavior_results['r2'] = np.load(r2_file, allow_pickle=True)
            
        # Load wheel data
        wheel_file = os.path.join(behavior_dir, "wheel_data.npy")
        if os.path.exists(wheel_file):
            behavior_results['wheel_data'] = np.load(wheel_file, allow_pickle=True)
            
        # Load whisker data
        whisker_file = os.path.join(behavior_dir, "whisker_data.npy")
        if os.path.exists(whisker_file):
            behavior_results['whisker_data'] = np.load(whisker_file, allow_pickle=True)
            
        results['behavior'] = behavior_results
    
    return results

def print_summary_stats(results):
    """Print summary statistics"""
    print("=" * 60)
    print("NEDS EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    
    if 'spike' in results:
        print("\n🧠 SPIKE RECONSTRUCTION RESULTS:")
        spike_results = results['spike']
        
        if 'bps' in spike_results:
            bps = spike_results['bps']
            print(f"  Bits per Spike (BPS):")
            print(f"    Mean: {np.mean(bps):.4f}")
            print(f"    Std:  {np.std(bps):.4f}")
            print(f"    Min:  {np.min(bps):.4f}")
            print(f"    Max:  {np.max(bps):.4f}")
            print(f"    Number of neurons: {len(bps)}")
            
        if 'r2' in spike_results:
            r2 = spike_results['r2']
            print(f"  R² (Variance Explained):")
            print(f"    Mean: {np.mean(r2):.4f}")
            print(f"    Std:  {np.std(r2):.4f}")
            print(f"    Min:  {np.min(r2):.4f}")
            print(f"    Max:  {np.max(r2):.4f}")
            print(f"    Number of neurons: {len(r2)}")
            
        if 'spike_data' in spike_results:
            spike_data = spike_results['spike_data']
            print(f"  Spike Data Shape: {spike_data.shape}")
    
    if 'behavior' in results:
        print("\n🎯 BEHAVIOR PREDICTION RESULTS:")
        behavior_results = results['behavior']
        
        if 'acc' in behavior_results:
            acc = behavior_results['acc']
            print(f"  Accuracy:")
            if isinstance(acc, dict):
                print(f"    Type: Dictionary with keys {list(acc.keys())}")
                for key, val in acc.items():
                    try:
                        if hasattr(val, '__len__') and hasattr(val, 'shape') and len(val.shape) > 0:
                            print(f"    {key}: Mean={np.mean(val):.4f}, Std={np.std(val):.4f}")
                        else:
                            print(f"    {key}: {val}")
                    except:
                        print(f"    {key}: {val}")
            else:
                try:
                    if hasattr(acc, 'shape') and len(acc.shape) > 0:
                        print(f"    Mean: {np.mean(acc):.4f}")
                        print(f"    Std:  {np.std(acc):.4f}")
                        print(f"    Shape: {acc.shape}")
                    else:
                        print(f"    Value: {acc}")
                except:
                    print(f"    Type: {type(acc)}, Value: {acc}")
            
        if 'r2' in behavior_results:
            r2 = behavior_results['r2']
            print(f"  R² (Behavior Prediction):")
            if isinstance(r2, dict):
                print(f"    Type: Dictionary with keys {list(r2.keys())}")
                for key, val in r2.items():
                    try:
                        if hasattr(val, '__len__') and hasattr(val, 'shape') and len(val.shape) > 0:
                            print(f"    {key}: Mean={np.mean(val):.4f}, Std={np.std(val):.4f}")
                        else:
                            print(f"    {key}: {val}")
                    except:
                        print(f"    {key}: {val}")
            else:
                try:
                    if hasattr(r2, 'shape') and len(r2.shape) > 0:
                        print(f"    Mean: {np.mean(r2):.4f}")
                        print(f"    Std:  {np.std(r2):.4f}")
                        print(f"    Shape: {r2.shape}")
                    else:
                        print(f"    Value: {r2}")
                except:
                    print(f"    Type: {type(r2)}, Value: {r2}")
            
        if 'wheel_data' in behavior_results:
            wheel_data = behavior_results['wheel_data']
            try:
                if hasattr(wheel_data, 'shape'):
                    print(f"  Wheel Data Shape: {wheel_data.shape}")
                else:
                    print(f"  Wheel Data Type: {type(wheel_data)}")
            except:
                print(f"  Wheel Data: {wheel_data}")
            
        if 'whisker_data' in behavior_results:
            whisker_data = behavior_results['whisker_data']
            try:
                if hasattr(whisker_data, 'shape'):
                    print(f"  Whisker Data Shape: {whisker_data.shape}")
                else:
                    print(f"  Whisker Data Type: {type(whisker_data)}")
            except:
                print(f"  Whisker Data: {whisker_data}")

def create_visualizations(results, save_dir=None):
    """Create visualizations of the results"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    plot_idx = 1
    
    # Spike reconstruction plots
    if 'spike' in results:
        spike_results = results['spike']
        
        # BPS histogram
        if 'bps' in spike_results:
            plt.subplot(2, 3, plot_idx)
            bps = spike_results['bps']
            plt.hist(bps, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Bits per Spike (BPS)')
            plt.ylabel('Number of Neurons')
            plt.title(f'Distribution of BPS\n(Mean: {np.mean(bps):.3f} ± {np.std(bps):.3f})')
            plt.grid(True, alpha=0.3)
            plot_idx += 1
            
        # R2 histogram for spikes
        if 'r2' in spike_results:
            plt.subplot(2, 3, plot_idx)
            r2 = spike_results['r2']
            plt.hist(r2, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.xlabel('R² (Spike Reconstruction)')
            plt.ylabel('Number of Neurons')
            plt.title(f'Distribution of Spike R²\n(Mean: {np.mean(r2):.3f} ± {np.std(r2):.3f})')
            plt.grid(True, alpha=0.3)
            plot_idx += 1
            
        # BPS vs R2 scatter plot
        if 'bps' in spike_results and 'r2' in spike_results:
            plt.subplot(2, 3, plot_idx)
            plt.scatter(spike_results['r2'], spike_results['bps'], alpha=0.6, color='purple')
            plt.xlabel('R² (Spike Reconstruction)')
            plt.ylabel('Bits per Spike (BPS)')
            plt.title('BPS vs R² Correlation')
            plt.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = np.corrcoef(spike_results['r2'], spike_results['bps'])[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            plot_idx += 1
    
    # Behavior prediction plots
    if 'behavior' in results:
        behavior_results = results['behavior']
        
        # Accuracy plot
        if 'acc' in behavior_results:
            plt.subplot(2, 3, plot_idx)
            acc = behavior_results['acc']
            if acc.ndim > 0:
                plt.hist(acc.flatten(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                plt.xlabel('Accuracy')
                plt.ylabel('Frequency')
                plt.title(f'Behavior Prediction Accuracy\n(Mean: {np.mean(acc):.3f} ± {np.std(acc):.3f})')
            else:
                plt.bar(['Accuracy'], [acc], color='lightgreen', alpha=0.7)
                plt.ylabel('Accuracy')
                plt.title(f'Behavior Prediction Accuracy: {acc:.3f}')
            plt.grid(True, alpha=0.3)
            plot_idx += 1
            
        # Behavior R2
        if 'r2' in behavior_results:
            plt.subplot(2, 3, plot_idx)
            r2 = behavior_results['r2']
            if r2.ndim > 0:
                plt.hist(r2.flatten(), bins=30, alpha=0.7, color='orange', edgecolor='black')
                plt.xlabel('R² (Behavior)')
                plt.ylabel('Frequency')
                plt.title(f'Behavior R² Distribution\n(Mean: {np.mean(r2):.3f} ± {np.std(r2):.3f})')
            else:
                plt.bar(['R²'], [r2], color='orange', alpha=0.7)
                plt.ylabel('R²')
                plt.title(f'Behavior R²: {r2:.3f}')
            plt.grid(True, alpha=0.3)
            plot_idx += 1
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "evaluation_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="View NEDS evaluation results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Path to results directory")
    parser.add_argument("--save_plots", type=str, default=None,
                       help="Directory to save plots (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist!")
        return
    
    print(f"Loading results from: {args.results_dir}")
    results = load_and_summarize_results(args.results_dir)
    
    if not results:
        print("No evaluation results found in the specified directory!")
        return
    
    # Print summary statistics
    print_summary_stats(results)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_visualizations(results, save_dir=args.save_plots)

if __name__ == "__main__":
    main()
