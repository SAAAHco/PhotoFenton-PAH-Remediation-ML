"""
Advanced Visualization Module for PAH Removal Analysis
Author: [Your Name]
Date: 2025
Description: 3D plots, violin plots, surface plots, and other advanced visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class AdvancedVisualizer:
    """
    Advanced visualization tools for PAH removal analysis.
    """
    
    def __init__(self, style='seaborn'):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.figures = []
    
    def plot_3d_surface(self, x, y, z, xlabel='X', ylabel='Y', zlabel='Z', 
                        title='3D Surface Plot', colormap='viridis'):
        """
        Create 3D surface plot.
        
        Args:
            x, y, z: Data arrays
            xlabel, ylabel, zlabel: Axis labels
            title: Plot title
            colormap: Colormap to use
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate z values
        zi = griddata((x, y), z, (xi, yi), method='cubic')
        
        # Create surface plot
        surf = ax.plot_surface(xi, yi, zi, cmap=colormap, 
                               linewidth=0, antialiased=True, alpha=0.8)
        
        # Add scatter points
        ax.scatter(x, y, z, c='red', marker='o', s=20, alpha=0.5)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        self.figures.append(fig)
        return fig
    
    def plot_interactive_3d(self, df, x_col, y_col, z_col, color_col=None,
                           title='Interactive 3D Plot'):
        """
        Create interactive 3D scatter plot using Plotly.
        
        Args:
            df: DataFrame with data
            x_col, y_col, z_col: Column names for axes
            color_col: Column for color coding
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        if color_col:
            fig.add_trace(go.Scatter3d(
                x=df[x_col],
                y=df[y_col],
                z=df[z_col],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df[color_col],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=color_col)
                ),
                text=df.index,
                hovertemplate='<b>%{text}</b><br>' +
                             f'{x_col}: %{{x:.2f}}<br>' +
                             f'{y_col}: %{{y:.2f}}<br>' +
                             f'{z_col}: %{{z:.2f}}<br>' +
                             '<extra></extra>'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=df[x_col],
                y=df[y_col],
                z=df[z_col],
                mode='markers',
                marker=dict(size=8, color='blue'),
                text=df.index
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def plot_violin_comparison(self, data_dict, ylabel='Value', 
                               title='Violin Plot Comparison'):
        """
        Create violin plots for comparing distributions.
        
        Args:
            data_dict: Dictionary of {label: data_array}
            ylabel: Y-axis label
            title: Plot title
            
        Returns:
            Figure object
        """
        # Prepare data for plotting
        plot_data = []
        for label, data in data_dict.items():
            df_temp = pd.DataFrame({
                'value': data,
                'category': label
            })
            plot_data.append(df_temp)
        
        df_plot = pd.concat(plot_data, ignore_index=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create violin plot
        sns.violinplot(data=df_plot, x='category', y='value', ax=ax, inner='box')
        
        # Add swarm plot overlay
        sns.swarmplot(data=df_plot, x='category', y='value', ax=ax, 
                     color='black', alpha=0.3, size=3)
        
        ax.set_xlabel('Category')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Rotate x labels if needed
        if len(data_dict) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_parameter_optimization_surface(self, param1_range, param2_range, 
                                          performance_matrix, param1_name='Parameter 1',
                                          param2_name='Parameter 2', 
                                          performance_name='Performance'):
        """
        Create optimization surface plot for two parameters.
        
        Args:
            param1_range: Range of first parameter
            param2_range: Range of second parameter
            performance_matrix: 2D array of performance values
            param1_name: Name of first parameter
            param2_name: Name of second parameter
            performance_name: Name of performance metric
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=[go.Surface(
            x=param1_range,
            y=param2_range,
            z=performance_matrix,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=performance_name)
        )])
        
        # Add contour projections
        fig.update_traces(
            contours_z=dict(show=True, usecolormap=True,
                          highlightcolor="limegreen", project_z=True)
        )
        
        fig.update_layout(
            title=f'Parameter Optimization Surface',
            scene=dict(
                xaxis_title=param1_name,
                yaxis_title=param2_name,
                zaxis_title=performance_name,
                camera_eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def plot_time_series_comparison(self, time_data_dict, xlabel='Time', 
                                   ylabel='Value', title='Time Series Comparison'):
        """
        Plot multiple time series for comparison.
        
        Args:
            time_data_dict: Dictionary of {label: (time_array, value_array)}
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(time_data_dict)))
        
        for (label, (time, values)), color in zip(time_data_dict.items(), colors):
            ax.plot(time, values, label=label, color=color, linewidth=2, marker='o', 
                   markersize=4, alpha=0.8)
            
            # Add confidence interval if values is 2D (mean and std)
            if isinstance(values, tuple) and len(values) == 2:
                mean_values, std_values = values
                ax.fill_between(time, mean_values - std_values, 
                              mean_values + std_values, 
                              color=color, alpha=0.2)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_heatmap_grid(self, data_matrices, titles, main_title='Heatmap Grid',
                         cmap='coolwarm', figsize=(15, 10)):
        """
        Create grid of heatmaps for comparing multiple conditions.
        
        Args:
            data_matrices: List of 2D arrays
            titles: List of subplot titles
            main_title: Main figure title
            cmap: Colormap
            figsize: Figure size
            
        Returns:
            Figure object
        """
        n_plots = len(data_matrices)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (data, title) in enumerate(zip(data_matrices, titles)):
            if idx < n_plots:
                im = axes[idx].imshow(data, cmap=cmap, aspect='auto')
                axes[idx].set_title(title)
                plt.colorbar(im, ax=axes[idx])
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(main_title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_parallel_coordinates(self, df, class_column, cols_to_plot=None,
                                 title='Parallel Coordinates Plot'):
        """
        Create parallel coordinates plot for multi-dimensional data.
        
        Args:
            df: DataFrame with data
            class_column: Column name for classification
            cols_to_plot: List of columns to plot (if None, uses all numeric)
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if cols_to_plot is None:
            cols_to_plot = df.select_dtypes(include=[np.number]).columns.tolist()
            if class_column in cols_to_plot:
                cols_to_plot.remove(class_column)
        
        # Normalize data for better visualization
        df_norm = df.copy()
        for col in cols_to_plot:
            df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(color=df[class_column],
                         colorscale='Viridis',
                         showscale=True),
                dimensions=[dict(range=[df_norm[col].min(), df_norm[col].max()],
                               label=col, values=df_norm[col])
                          for col in cols_to_plot]
            )
        )
        
        fig.update_layout(
            title=title,
            width=1200,
            height=500
        )
        
        return fig
    
    def plot_radar_chart(self, categories, values_dict, title='Radar Chart'):
        """
        Create radar chart for comparing multiple metrics.
        
        Args:
            categories: List of category names
            values_dict: Dictionary of {label: values_array}
            title: Chart title
            
        Returns:
            Figure object
        """
        fig = go.Figure()
        
        for label, values in values_dict.items():
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=label
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([max(v) for v in values_dict.values()])]
                )),
            showlegend=True,
            title=title,
            width=700,
            height=600
        )
        
        return fig
    
    def plot_boxplot_with_points(self, data_groups, labels=None, 
                                ylabel='Value', title='Box Plot with Data Points'):
        """
        Create box plots with overlaid data points.
        
        Args:
            data_groups: List of data arrays
            labels: List of labels for each group
            ylabel: Y-axis label
            title: Plot title
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if labels is None:
            labels = [f'Group {i+1}' for i in range(len(data_groups))]
        
        # Create box plot
        bp = ax.boxplot(data_groups, labels=labels, patch_artist=True)
        
        # Customize box plot colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_groups)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Overlay data points
        for i, (data, color) in enumerate(zip(data_groups, colors)):
            x = np.random.normal(i + 1, 0.04, len(data))
            ax.scatter(x, data, color=color, alpha=0.6, s=20)
        
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def create_dashboard(self, data_dict, title='Analysis Dashboard'):
        """
        Create interactive dashboard with multiple plots.
        
        Args:
            data_dict: Dictionary containing various data for plotting
            title: Dashboard title
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Performance Over Time', 'Feature Importance',
                          'Parameter Distribution', 'Correlation Matrix'],
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                  [{'type': 'violin'}, {'type': 'heatmap'}]]
        )
        
        # Add traces for each subplot
        # This is a template - customize based on your specific data
        
        # Performance over time (example)
        if 'time_series' in data_dict:
            fig.add_trace(
                go.Scatter(x=data_dict['time_series']['x'],
                          y=data_dict['time_series']['y'],
                          mode='lines+markers',
                          name='Performance'),
                row=1, col=1
            )
        
        # Feature importance (example)
        if 'feature_importance' in data_dict:
            fig.add_trace(
                go.Bar(x=data_dict['feature_importance']['features'],
                      y=data_dict['feature_importance']['importance'],
                      name='Importance'),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text=title,
            showlegend=True,
            height=800,
            width=1400
        )
        
        return fig
    
    def save_all_figures(self, save_dir='figures', dpi=300, format='png'):
        """
        Save all created figures.
        
        Args:
            save_dir: Directory to save figures
            dpi: Resolution for saved figures
            format: File format ('png', 'pdf', 'svg')
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for idx, fig in enumerate(self.figures):
            filepath = os.path.join(save_dir, f'figure_{idx+1}.{format}')
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")


def create_comprehensive_visualizations(results_dict):
    """
    Create comprehensive visualization suite from results.
    
    Args:
        results_dict: Dictionary containing various analysis results
        
    Returns:
        Dictionary of created figures
    """
    visualizer = AdvancedVisualizer()
    figures = {}
    
    # Example visualizations - customize based on your data
    
    # 3D surface plot for parameter optimization
    if 'optimization_data' in results_dict:
        data = results_dict['optimization_data']
        fig = visualizer.plot_3d_surface(
            data['x'], data['y'], data['z'],
            xlabel='EDTA Concentration',
            ylabel='H2O2 Concentration',
            zlabel='Removal Rate (%)',
            title='Parameter Optimization Surface'
        )
        figures['3d_surface'] = fig
    
    # Violin plots for PAH comparison
    if 'pah_removal_rates' in results_dict:
        fig = visualizer.plot_violin_comparison(
            results_dict['pah_removal_rates'],
            ylabel='Removal Rate (%)',
            title='PAH Removal Rate Distribution by Compound'
        )
        figures['violin_plot'] = fig
    
    # Time series comparison
    if 'time_series_data' in results_dict:
        fig = visualizer.plot_time_series_comparison(
            results_dict['time_series_data'],
            xlabel='Time (minutes)',
            ylabel='Removal Rate (%)',
            title='Removal Rate Over Time'
        )
        figures['time_series'] = fig
    
    return figures


def main():
    """Main execution function for visualization."""
    logger.info("Creating advanced visualizations...")
    
    # Generate example data
    np.random.seed(42)
    
    # Example 3D data
    n_points = 100
    x = np.random.uniform(10, 20, n_points)
    y = np.random.uniform(5, 15, n_points)
    z = 70 + 2*x + 1.5*y + np.random.normal(0, 5, n_points)
    
    # Initialize visualizer
    viz = AdvancedVisualizer()
    
    # Create 3D surface plot
    fig1 = viz.plot_3d_surface(x, y, z,
                               xlabel='EDTA Concentration (g/L)',
                               ylabel='Fe Concentration (mM)',
                               zlabel='Removal Rate (%)',
                               title='3D Parameter Optimization')
    
    # Create violin plot example
    violin_data = {
        'Naphthalene': np.random.normal(95, 3, 50),
        'Anthracene': np.random.normal(92, 4, 50),
        'Pyrene': np.random.normal(88, 5, 50),
        'Fluorene': np.random.normal(90, 3.5, 50)
    }
    fig2 = viz.plot_violin_comparison(violin_data,
                                      ylabel='Removal Rate (%)',
                                      title='PAH Removal Distribution')
    
    # Save figures
    viz.save_all_figures(save_dir='results/figures')
    
    logger.info("Visualizations completed!")
    
    return viz


if __name__ == "__main__":
    main()