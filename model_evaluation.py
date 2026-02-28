import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, 
                             confusion_matrix, classification_report)
import pickle
import xgboost as xgb

class ModelEvaluator:
    """
    Comprehensive model evaluation with visualizations and metrics
    """
    
    def __init__(self, model_file='models/objective_model.pkl', 
                 data_file='data/objective_training_data.csv'):
        
        # Load model
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
        
        # Load and prepare data
        self.data = pd.read_csv(data_file)
        self.X, self.y = self._prepare_data()
        
        # Split data (same random state as training for consistency)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        print(f"Loaded model and {len(self.data)} samples")
        print(f"Test set: {len(self.X_test)} samples\n")
    
    def _prepare_data(self):
        """Prepare features and labels"""
        # One-hot encode drake types
        drake_dummies = pd.get_dummies(self.data['drake_type'], prefix='drake')
        
        # Core features
        feature_cols = ['gold_diff', 'level_diff', 'ally_alive', 
                       'enemy_alive', 'game_time_minutes']
        
        # Advanced features
        advanced = ['cs_diff', 'ally_total_cs', 'enemy_total_cs',
                   'ally_completed_items', 'enemy_completed_items', 'item_advantage',
                   'kills_last_2min_ally', 'kills_last_2min_enemy', 'kill_diff_recent',
                   'ally_towers', 'enemy_towers', 'tower_diff', 'time_since_last_fight']
        
        for feat in advanced:
            if feat in self.data.columns:
                feature_cols.append(feat)
        
        X = self.data[feature_cols].copy()
        
        # Add drake dummies
        for drake in ['INFERNAL', 'MOUNTAIN', 'OCEAN', 'CLOUD', 'HEXTECH', 'CHEMTECH']:
            col = f'drake_{drake}'
            X[col] = drake_dummies[col] if col in drake_dummies else 0
        
        X['is_baron'] = self.data['is_baron'] if 'is_baron' in self.data.columns else 0
        
        # Ensure column order matches training
        X = X[self.feature_names]
        y = self.data['objective_secured']
        
        return X, y
    
    def calculate_metrics(self):
        """Calculate all performance metrics"""
        print("="*70)
        print("MODEL PERFORMANCE METRICS")
        print("="*70)
        
        # Basic metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        print(f"\nClassification Metrics:")
        print(f"  Accuracy:  {accuracy:.3f}  (How often model is correct)")
        print(f"  Precision: {precision:.3f}  (When it predicts success, how often right?)")
        print(f"  Recall:    {recall:.3f}  (Of all successes, how many did it catch?)")
        print(f"  F1-Score:  {f1:.3f}  (Balance of precision and recall)")
        print(f"  ROC-AUC:   {roc_auc:.3f}  (Overall ranking ability, 0.5=random, 1.0=perfect)")
        
        # What these mean
        print(f"\nInterpretation:")
        if accuracy > 0.75:
            print(f"  ✓ Excellent accuracy - model is very reliable")
        elif accuracy > 0.65:
            print(f"  ✓ Good accuracy - model is useful")
        elif accuracy > 0.55:
            print(f"  ⚠ Fair accuracy - model is better than guessing")
        else:
            print(f"  ✗ Poor accuracy - model needs improvement")
        
        if roc_auc > 0.80:
            print(f"  ✓ Excellent discrimination - model ranks predictions well")
        elif roc_auc > 0.70:
            print(f"  ✓ Good discrimination")
        elif roc_auc > 0.60:
            print(f"  ⚠ Fair discrimination")
        else:
            print(f"  ✗ Poor discrimination")
        
        # Detailed classification report
        print(f"\nDetailed Report:")
        print(classification_report(self.y_test, self.y_pred, 
                                   target_names=['Failure', 'Success']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted Failure', 'Predicted Success'],
                   yticklabels=['Actual Failure', 'Actual Success'])
        plt.title('Confusion Matrix\n(How often model is right/wrong)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add interpretation
        tn, fp, fn, tp = cm.ravel()
        plt.text(0.5, -0.15, 
                f'True Negatives: {tn} | False Positives: {fp}\n'
                f'False Negatives: {fn} | True Positives: {tp}',
                ha='center', transform=plt.gca().transAxes, fontsize=9)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: confusion_matrix.png")
    
    def plot_roc_curve(self):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Guess (AUC = 0.500)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve\n(Higher curve = better model)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: roc_curve.png")
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title('Top 15 Most Important Features\n(What matters most for predictions?)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: feature_importance.png")
        
        return importance_df
    
    def plot_probability_distribution(self):
        """Plot distribution of predicted probabilities"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of probabilities
        ax1.hist(self.y_pred_proba[self.y_test == 0], bins=20, alpha=0.5, 
                label='Actual Failures', color='red')
        ax1.hist(self.y_pred_proba[self.y_test == 1], bins=20, alpha=0.5, 
                label='Actual Successes', color='green')
        ax1.set_xlabel('Predicted Success Probability')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Predicted Probabilities')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Calibration-like plot
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_counts = []
        bin_successes = []
        
        for i in range(len(bins) - 1):
            mask = (self.y_pred_proba >= bins[i]) & (self.y_pred_proba < bins[i+1])
            bin_counts.append(mask.sum())
            if mask.sum() > 0:
                bin_successes.append(self.y_test[mask].mean())
            else:
                bin_successes.append(0)
        
        ax2.scatter(bin_centers, bin_successes, s=[c*10 for c in bin_counts], alpha=0.6)
        ax2.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Actual Success Rate')
        ax2.set_title('Calibration Plot\n(Are probabilities accurate?)')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('probability_analysis.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: probability_analysis.png")
    
    def plot_performance_by_game_time(self):
        """Analyze performance at different game times"""
        test_indices = self.X_test.index
        game_times = self.data.loc[test_indices, 'game_time_minutes']
        
        # Bin by game time
        time_bins = [0, 10, 15, 20, 25, 100]
        labels = ['0-10min', '10-15min', '15-20min', '20-25min', '25+min']
        
        df_analysis = pd.DataFrame({
            'game_time': game_times,
            'predicted': self.y_pred,
            'actual': self.y_test,
            'correct': self.y_pred == self.y_test
        })
        
        df_analysis['time_bin'] = pd.cut(df_analysis['game_time'], 
                                         bins=time_bins, labels=labels)
        
        accuracy_by_time = df_analysis.groupby('time_bin')['correct'].mean()
        counts_by_time = df_analysis.groupby('time_bin').size()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy by time
        accuracy_by_time.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_xlabel('Game Time')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy by Game Time')
        ax1.axhline(y=0.5, color='r', linestyle='--', label='Random guess')
        ax1.legend()
        ax1.set_ylim([0, 1])
        
        # Sample counts
        counts_by_time.plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_xlabel('Game Time')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Data Distribution by Game Time')
        
        plt.tight_layout()
        plt.savefig('performance_by_time.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: performance_by_time.png")
    
    def analyze_errors(self):
        """Analyze where model makes mistakes"""
        print("\n" + "="*70)
        print("ERROR ANALYSIS")
        print("="*70)
        
        # False positives (predicted success but failed)
        fp_mask = (self.y_pred == 1) & (self.y_test == 0)
        fp_indices = self.X_test[fp_mask].index
        
        # False negatives (predicted failure but succeeded)
        fn_mask = (self.y_pred == 0) & (self.y_test == 1)
        fn_indices = self.X_test[fn_mask].index
        
        print(f"\nFalse Positives: {fp_mask.sum()} cases")
        print("  (Model said GO FOR IT but team failed)")
        if fp_mask.sum() > 0:
            fp_data = self.data.loc[fp_indices]
            print(f"  Avg gold diff: {fp_data['gold_diff'].mean():.0f}")
            print(f"  Avg alive: {fp_data['ally_alive'].mean():.1f} vs {fp_data['enemy_alive'].mean():.1f}")
        
        print(f"\nFalse Negatives: {fn_mask.sum()} cases")
        print("  (Model said SKIP but team succeeded)")
        if fn_mask.sum() > 0:
            fn_data = self.data.loc[fn_indices]
            print(f"  Avg gold diff: {fn_data['gold_diff'].mean():.0f}")
            print(f"  Avg alive: {fn_data['ally_alive'].mean():.1f} vs {fn_data['enemy_alive'].mean():.1f}")
    
    def generate_full_report(self):
        """Generate complete evaluation report"""
        print("\n" + "="*70)
        print("GENERATING FULL EVALUATION REPORT")
        print("="*70 + "\n")
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Generate all plots
        print("\nGenerating visualizations...")
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        importance_df = self.plot_feature_importance()
        self.plot_probability_distribution()
        self.plot_performance_by_game_time()
        
        # Error analysis
        self.analyze_errors()
        
        print("\n" + "="*70)
        print("REPORT COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  - confusion_matrix.png")
        print("  - roc_curve.png")
        print("  - feature_importance.png")
        print("  - probability_analysis.png")
        print("  - performance_by_time.png")
        
        print("\n📊 Open these images to see visualizations!")
        
        return metrics, importance_df


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    print("Model Evaluation Suite")
    print("="*70 + "\n")
    
    evaluator = ModelEvaluator()
    metrics, importance = evaluator.generate_full_report()