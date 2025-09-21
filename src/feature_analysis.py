"""
Feature Importance and Statistical Analysis Module
Author: [Your Name]
Date: 2025
Description: Comprehensive feature analysis including Random Forest, statistical testing, and clustering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """
    Comprehensive feature analysis for PAH removal prediction models.
    """
    
    def __init__(self):
        """Initialize the feature analyzer."""
        self.rf_model = None
        self.feature_importance_df = None
        self.statistical_results = None
        self.clustering_results = None
        
    def random_forest_importance(self, X, y, feature_names=None, n_estimators=100):
        """
        Calculate feature importance using Random Forest.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            n_estimators: Number of trees in the forest
            
        Returns:
            DataFrame with feature importances
        """
        logger.info("Calculating Random Forest feature importance...")
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X, y)
        
        # Get feature importances
        importances = self.rf_model.feature_importances_
        
        # Create DataFrame
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        self.feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 5 features: {self.feature_importance_df.head()['feature'].tolist()}")
        
        return self.feature_importance_df
    
    def permutation_importance_analysis(self, model, X, y, feature_names=None, n_repeats=10):
        """
        Calculate permutation importance for any model.
        
        Args:
            model: Trained model with predict method
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            n_repeats: Number of permutation repeats
            
        Returns:
            DataFrame with permutation importances
        """
        logger.info("Calculating permutation importance...")
        
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        perm_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return perm_importance_df
    
    def statistical_analysis(self, X, y, feature_names=None, alpha=0.05):
        """
        Perform comprehensive statistical analysis.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            alpha: Significance level
            
        Returns:
            Dictionary with statistical results
        """
        logger.info("Performing statistical analysis...")
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        results = {
            'ols_results': None,
            'correlation_matrix': None,
            'feature_statistics': None,
            'significant_features': []
        }
        
        # OLS Regression
        X_with_const = sm.add_constant(X)
        model_ols = sm.OLS(y, X_with_const)
        results['ols_results'] = model_ols.fit()
        
        # Extract p-values and coefficients
        p_values = results['ols_results'].pvalues[1:]  # Exclude constant
        coefficients = results['ols_results'].params[1:]
        
        # Identify significant features
        significant_mask = p_values < alpha
        results['significant_features'] = [
            feature_names[i] for i, is_sig in enumerate(significant_mask) if is_sig
        ]
        
        # Correlation analysis
        df_analysis = pd.DataFrame(X, columns=feature_names)
        df_analysis['target'] = y
        results['correlation_matrix'] = df_analysis.corr()
        
        # Feature statistics
        results['feature_statistics'] = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'p_value': p_values,
            'significant': significant_mask,
            'correlation_with_target': [
                results['correlation_matrix'].loc[feat, 'target'] 
                for feat in feature_names
            ]
        })
        
        logger.info(f"Found {len(results['significant_features'])} significant features")
        
        self.statistical_results = results
        return results
    
    def hierarchical_clustering(self, X, feature_names=None, n_clusters=None, 
                               method='ward', metric='euclidean'):
        """
        Perform hierarchical clustering analysis.
        
        Args:
            X: Feature matrix or distance matrix
            feature_names: Names of features/samples
            n_clusters: Number of clusters (if None, uses silhouette analysis)
            method: Linkage method ('ward', 'complete', 'average', 'single')
            metric: Distance metric
            
        Returns:
            Dictionary with clustering results
        """
        logger.info("Performing hierarchical clustering...")
        
        if feature_names is None:
            feature_names = [f'Sample_{i}' for i in range(X.shape[0])]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate linkage matrix
        linkage_matrix = linkage(X_scaled, method=method, metric=metric)
        
        # Find optimal number of clusters if not specified
        if n_clusters is None:
            silhouette_scores = []
            for k in range(2, min(10, X.shape[0])):
                clusters = fcluster(linkage_matrix, k, criterion='maxclust')
                score = silhouette_score(X_scaled, clusters)
                silhouette_scores.append((k, score))
            
            n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
            logger.info(f"Optimal number of clusters: {n_clusters}")
        
        # Get cluster assignments
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Calculate cluster statistics
        cluster_stats = []
        for i in range(1, n_clusters + 1):
            mask = clusters == i
            cluster_stats.append({
                'cluster': i,
                'size': np.sum(mask),
                'members': [feature_names[j] for j, m in enumerate(mask) if m]
            })
        
        self.clustering_results = {
            'linkage_matrix': linkage_matrix,
            'clusters': clusters,
            'n_clusters': n_clusters,
            'cluster_stats': cluster_stats,
            'feature_names': feature_names
        }
        
        return self.clustering_results
    
    def pca_analysis(self, X, feature_names=None, n_components=None):
        """
        Perform PCA analysis for dimensionality reduction.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            n_components: Number of components (if None, keeps 95% variance)
            
        Returns:
            Dictionary with PCA results
        """
        logger.info("Performing PCA analysis...")
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        if n_components is None:
            pca = PCA(n_components=0.95)  # Keep 95% variance
        else:
            pca = PCA(n_components=n_components)
        
        X_pca = pca.fit_transform(X_scaled)
        
        # Get component loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=feature_names
        )
        
        # Calculate feature contributions to each PC
        feature_contributions = {}
        for i in range(pca.n_components_):
            contributions = abs(loadings[f'PC{i+1}'])
            feature_contributions[f'PC{i+1}'] = contributions.sort_values(ascending=False)
        
        results = {
            'pca_model': pca,
            'transformed_data': X_pca,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'loadings': loadings,
            'feature_contributions': feature_contributions,
            'n_components': pca.n_components_
        }
        
        logger.info(f"PCA with {pca.n_components_} components explains "
                   f"{results['cumulative_variance'][-1]:.2%} of variance")
        
        return results
    
    def plot_feature_importance_comparison(self, rf_importance, perm_importance, top_n=20):
        """
        Plot comparison of different feature importance methods.
        
        Args:
            rf_importance: Random Forest importance DataFrame
            perm_importance: Permutation importance DataFrame
            top_n: Number of top features to display
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Random Forest importance
        top_rf = rf_importance.head(top_n)
        ax1.barh(range(len(top_rf)), top_rf['importance'].values)
        ax1.set_yticks(range(len(top_rf)))
        ax1.set_yticklabels(top_rf['feature'].values)
        ax1.set_xlabel('Importance')
        ax1.set_title(f'Top {top_n} Features - Random Forest')
        ax1.invert_yaxis()
        
        # Permutation importance
        top_perm = perm_importance.head(top_n)
        ax2.barh(range(len(top_perm)), 
                top_perm['importance_mean'].values,
                xerr=top_perm['importance_std'].values)
        ax2.set_yticks(range(len(top_perm)))
        ax2.set_yticklabels(top_perm['feature'].values)
        ax2.set_xlabel('Importance')
        ax2.set_title(f'Top {top_n} Features - Permutation')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_correlation_heatmap(self, correlation_matrix, figsize=(12, 10)):
        """
        Plot correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   annot=False)
        
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    
    def plot_dendrogram(self, linkage_matrix, labels=None, figsize=(12, 8)):
        """
        Plot hierarchical clustering dendrogram.
        
        Args:
            linkage_matrix: Linkage matrix from hierarchical clustering
            labels: Labels for leaves
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        dendrogram(linkage_matrix,
                  labels=labels,
                  leaf_rotation=90,
                  leaf_font_size=10)
        
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample/Feature')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()
    
    def plot_pca_variance(self, pca_results):
        """
        Plot PCA explained variance.
        
        Args:
            pca_results: Results from PCA analysis
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scree plot
        components = range(1, len(pca_results['explained_variance_ratio']) + 1)
        ax1.bar(components, pca_results['explained_variance_ratio'])
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('PCA Scree Plot')
        ax1.set_xticks(components[::max(1, len(components)//10)])
        
        # Cumulative variance
        ax2.plot(components, pca_results['cumulative_variance'], 'bo-')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Variance Explained')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(components[::max(1, len(components)//10)])
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_feature_report(self, X, y, feature_names=None, save_path=None):
        """
        Create comprehensive feature analysis report.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            save_path: Path to save the report
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Creating comprehensive feature analysis report...")
        
        report = {}
        
        # Random Forest importance
        report['rf_importance'] = self.random_forest_importance(X, y, feature_names)
        
        # Permutation importance
        report['perm_importance'] = self.permutation_importance_analysis(
            self.rf_model, X, y, feature_names
        )
        
        # Statistical analysis
        report['statistical'] = self.statistical_analysis(X, y, feature_names)
        
        # PCA analysis
        report['pca'] = self.pca_analysis(X, feature_names)
        
        # Hierarchical clustering
        report['clustering'] = self.hierarchical_clustering(X.T, feature_names)
        
        # Create visualizations
        self.plot_feature_importance_comparison(
            report['rf_importance'], 
            report['perm_importance']
        )
        
        self.plot_correlation_heatmap(report['statistical']['correlation_matrix'])
        
        self.plot_pca_variance(report['pca'])
        
        self.plot_dendrogram(
            report['clustering']['linkage_matrix'],
            report['clustering']['feature_names']
        )
        
        # Save report if path provided
        if save_path:
            # Save important DataFrames
            report['rf_importance'].to_csv(f"{save_path}_rf_importance.csv", index=False)
            report['perm_importance'].to_csv(f"{save_path}_perm_importance.csv", index=False)
            report['statistical']['feature_statistics'].to_csv(
                f"{save_path}_statistical.csv", index=False
            )
            
            # Save summary
            with open(f"{save_path}_summary.txt", 'w') as f:
                f.write("Feature Analysis Report Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total features analyzed: {X.shape[1]}\n")
                f.write(f"Total samples: {X.shape[0]}\n\n")
                
                f.write("Top 10 Important Features (Random Forest):\n")
                for idx, row in report['rf_importance'].head(10).iterrows():
                    f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
                
                f.write(f"\nSignificant features (p < 0.05): {len(report['statistical']['significant_features'])}\n")
                f.write(f"PCA components for 95% variance: {report['pca']['n_components']}\n")
                f.write(f"Optimal number of clusters: {report['clustering']['n_clusters']}\n")
            
            logger.info(f"Report saved to {save_path}")
        
        return report


def main():
    """Main execution function for feature analysis."""
    # Example usage
    logger.info("Starting feature analysis...")
    
    # Generate example data (replace with actual data)
    n_samples = 100
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # Initialize analyzer
    analyzer = FeatureAnalyzer()
    
    # Create comprehensive report
    report = analyzer.create_feature_report(
        X, y, 
        feature_names=feature_names,
        save_path='results/feature_analysis'
    )
    
    logger.info("Feature analysis completed!")
    
    return report


if __name__ == "__main__":
    main()