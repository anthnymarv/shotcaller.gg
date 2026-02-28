import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, log_loss)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

class ModelComparison:
    """
    Compare multiple ML models with comprehensive metrics including AIC
    """
    
    def __init__(self, data_file='data/objective_training_data.csv'):
        print("Loading data...")
        self.data = pd.read_csv(data_file)
        self.X, self.y = self._prepare_data()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}\n")
        
        self.models = {}
        self.results = []
    
    def _prepare_data(self):
        """Prepare features and labels"""
        drake_dummies = pd.get_dummies(self.data['drake_type'], prefix='drake')
        
        feature_cols = ['gold_diff', 'level_diff', 'ally_alive', 
                       'enemy_alive', 'game_time_minutes']
        
        advanced = ['cs_diff', 'ally_total_cs', 'enemy_total_cs',
                   'ally_completed_items', 'enemy_completed_items', 'item_advantage',
                   'kills_last_2min_ally', 'kills_last_2min_enemy', 'kill_diff_recent',
                   'ally_towers', 'enemy_towers', 'tower_diff', 'time_since_last_fight']
        
        for feat in advanced:
            if feat in self.data.columns:
                feature_cols.append(feat)
        
        X = self.data[feature_cols].copy()
        
        for drake in ['INFERNAL', 'MOUNTAIN', 'OCEAN', 'CLOUD', 'HEXTECH', 'CHEMTECH']:
            col = f'drake_{drake}'
            X[col] = drake_dummies[col] if col in drake_dummies else 0
        
        X['is_baron'] = self.data['is_baron'] if 'is_baron' in self.data.columns else 0
        X = X.fillna(0)
        
        y = self.data['objective_secured']
        
        return X, y
    
    def calculate_aic(self, model, X, y):
        """
        Calculate AIC (Akaike Information Criterion)
        AIC = 2k - 2ln(L)
        where k = number of parameters, L = likelihood
        
        Lower AIC = better model (balances fit and complexity)
        """
        try:
            # Get predictions
            y_pred_proba = model.predict_proba(X)
            
            # Calculate log-likelihood
            epsilon = 1e-15  # Small value to avoid log(0)
            y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
            
            # Log-likelihood for binary classification
            log_likelihood = 0
            for i, true_label in enumerate(y):
                prob = y_pred_proba[i][int(true_label)]
                log_likelihood += np.log(prob)
            
            # Count parameters
            if hasattr(model, 'coef_'):
                # Linear models
                n_params = model.coef_.shape[1] + 1  # weights + bias
            elif hasattr(model, 'n_features_in_'):
                # Tree-based models (rough estimate)
                n_params = model.n_features_in_ * 10  # Rough heuristic
            else:
                n_params = X.shape[1]  # Fallback
            
            # AIC formula
            aic = 2 * n_params - 2 * log_likelihood
            
            return aic
            
        except Exception as e:
            return None
    
    def calculate_bic(self, model, X, y):
        """
        Calculate BIC (Bayesian Information Criterion)
        BIC = k*ln(n) - 2ln(L)
        where k = params, n = sample size, L = likelihood
        
        Similar to AIC but penalizes complexity more heavily
        """
        try:
            y_pred_proba = model.predict_proba(X)
            epsilon = 1e-15
            y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
            
            log_likelihood = 0
            for i, true_label in enumerate(y):
                prob = y_pred_proba[i][int(true_label)]
                log_likelihood += np.log(prob)
            
            if hasattr(model, 'coef_'):
                n_params = model.coef_.shape[1] + 1
            elif hasattr(model, 'n_features_in_'):
                n_params = model.n_features_in_ * 10
            else:
                n_params = X.shape[1]
            
            n_samples = len(X)
            bic = n_params * np.log(n_samples) - 2 * log_likelihood
            
            return bic
            
        except Exception as e:
            return None
    
    def train_and_evaluate_model(self, name, model):
        """Train model and calculate all metrics"""
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        logloss = log_loss(self.y_test, y_pred_proba)
        
        # AIC and BIC
        aic = self.calculate_aic(model, self.X_test, self.y_test)
        bic = self.calculate_bic(model, self.X_test, self.y_test)
        
        # Cross-validation score (5-fold)
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  ROC-AUC:  {roc_auc:.3f}")
        print(f"  AIC:      {aic:.1f}" if aic else "  AIC:      N/A")
        
        # Store results
        self.models[name] = model
        self.results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Log Loss': logloss,
            'AIC': aic,
            'BIC': bic,
            'CV Mean': cv_mean,
            'CV Std': cv_std
        })
        
        return model
    
    def compare_all_models(self):
        """Train and compare all models"""
        print("="*70)
        print("MODEL COMPARISON SUITE")
        print("="*70)
        
        # 1. XGBoost
        xgb_model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.train_and_evaluate_model('XGBoost', xgb_model)
        
        # 2. Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        self.train_and_evaluate_model('Random Forest', rf_model)
        
        # 3. Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.train_and_evaluate_model('Gradient Boosting', gb_model)
        
        # 4. Logistic Regression
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        self.train_and_evaluate_model('Logistic Regression', lr_model)
        
        # 5. Neural Network
        nn_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            max_iter=500,
            random_state=42
        )
        self.train_and_evaluate_model('Neural Network', nn_model)
        
        print("\n" + "="*70)
        print("COMPARISON COMPLETE")
        print("="*70)
    
    def print_comparison_table(self):
        """Print comprehensive comparison table"""
        df_results = pd.DataFrame(self.results)
        
        print("\n" + "="*100)
        print("DETAILED METRICS COMPARISON")
        print("="*100)
        
        # Sort by ROC-AUC (primary metric)
        df_results = df_results.sort_values('ROC-AUC', ascending=False)
        
        print(df_results.to_string(index=False))
        
        print("\n" + "="*100)
        print("RANKING BY METRIC")
        print("="*100)
        
        # Best model for each metric
        print(f"\nBest Accuracy:  {df_results.loc[df_results['Accuracy'].idxmax(), 'Model']:20s} "
              f"({df_results['Accuracy'].max():.3f})")
        print(f"Best ROC-AUC:   {df_results.loc[df_results['ROC-AUC'].idxmax(), 'Model']:20s} "
              f"({df_results['ROC-AUC'].max():.3f})")
        print(f"Best F1-Score:  {df_results.loc[df_results['F1-Score'].idxmax(), 'Model']:20s} "
              f"({df_results['F1-Score'].max():.3f})")
        
        # AIC/BIC (lower is better)
        df_with_aic = df_results[df_results['AIC'].notna()]
        if len(df_with_aic) > 0:
            print(f"Lowest AIC:     {df_with_aic.loc[df_with_aic['AIC'].idxmin(), 'Model']:20s} "
                  f"({df_with_aic['AIC'].min():.1f}) - simpler model")
            print(f"Lowest BIC:     {df_with_aic.loc[df_with_aic['BIC'].idxmin(), 'Model']:20s} "
                  f"({df_with_aic['BIC'].min():.1f}) - penalizes complexity more")
        
        print("\n💡 Lower AIC/BIC = better balance of accuracy and simplicity")
        
        return df_results
    
    def plot_comparison(self, df_results):
        """Create comparison visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        models = df_results['Model']
        accuracy = df_results['Accuracy']
        colors = ['gold' if a == accuracy.max() else 'skyblue' for a in accuracy]
        ax1.barh(models, accuracy, color=colors)
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison')
        ax1.axvline(x=0.5, color='r', linestyle='--', alpha=0.3, label='Random Guess')
        ax1.set_xlim([0, 1])
        for i, v in enumerate(accuracy):
            ax1.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # 2. ROC-AUC comparison
        ax2 = axes[0, 1]
        roc_auc = df_results['ROC-AUC']
        colors = ['gold' if r == roc_auc.max() else 'lightcoral' for r in roc_auc]
        ax2.barh(models, roc_auc, color=colors)
        ax2.set_xlabel('ROC-AUC')
        ax2.set_title('ROC-AUC Comparison')
        ax2.axvline(x=0.5, color='r', linestyle='--', alpha=0.3, label='Random')
        ax2.set_xlim([0, 1])
        for i, v in enumerate(roc_auc):
            ax2.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # 3. Precision vs Recall
        ax3 = axes[1, 0]
        precision = df_results['Precision']
        recall = df_results['Recall']
        ax3.scatter(recall, precision, s=200, alpha=0.6)
        for i, model in enumerate(models):
            ax3.annotate(model, (recall.iloc[i], precision.iloc[i]), 
                        fontsize=8, ha='right')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision vs Recall Trade-off')
        ax3.grid(alpha=0.3)
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1])
        
        # 4. AIC/BIC comparison (if available)
        ax4 = axes[1, 1]
        df_with_aic = df_results[df_results['AIC'].notna()]
        if len(df_with_aic) > 0:
            x = np.arange(len(df_with_aic))
            width = 0.35
            ax4.bar(x - width/2, df_with_aic['AIC'], width, label='AIC', alpha=0.8)
            ax4.bar(x + width/2, df_with_aic['BIC'], width, label='BIC', alpha=0.8)
            ax4.set_xlabel('Model')
            ax4.set_ylabel('Information Criterion (lower = better)')
            ax4.set_title('AIC/BIC Comparison\n(Lower = Better Balance of Fit and Complexity)')
            ax4.set_xticks(x)
            ax4.set_xticklabels(df_with_aic['Model'], rotation=45, ha='right')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'AIC/BIC not available for all models', 
                    ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved: model_comparison.png")
    
    def plot_cv_comparison(self, df_results):
        """Plot cross-validation scores"""
        plt.figure(figsize=(10, 6))
        
        models = df_results['Model']
        cv_means = df_results['CV Mean']
        cv_stds = df_results['CV Std']
        
        plt.barh(models, cv_means, xerr=cv_stds, capsize=5, alpha=0.7, color='mediumseagreen')
        plt.xlabel('Cross-Validation Accuracy (5-fold)')
        plt.title('Model Stability (Cross-Validation)\nLower error bars = more stable')
        plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.3, label='Random Guess')
        plt.xlim([0, 1])
        
        for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
            plt.text(mean + 0.01, i, f'{mean:.3f}±{std:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('cv_comparison.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: cv_comparison.png")
    
    def save_best_model(self, df_results):
        """Save the best performing model"""
        best_model_name = df_results.loc[df_results['ROC-AUC'].idxmax(), 'Model']
        best_model = self.models[best_model_name]
        
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump({
                'model': best_model,
                'model_name': best_model_name,
                'feature_names': list(self.X.columns),
                'metrics': df_results[df_results['Model'] == best_model_name].to_dict('records')[0]
            }, f)
        
        print(f"\n✓ Saved best model: {best_model_name}")
        print(f"  Location: models/best_model.pkl")
        print(f"  ROC-AUC: {df_results['ROC-AUC'].max():.3f}")
    
    def generate_report(self):
        """Generate full comparison report"""
        self.compare_all_models()
        df_results = self.print_comparison_table()
        
        print("\nGenerating visualizations...")
        self.plot_comparison(df_results)
        self.plot_cv_comparison(df_results)
        
        self.save_best_model(df_results)
        
        print("\n" + "="*70)
        print("REPORT COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  - model_comparison.png (4 comparison charts)")
        print("  - cv_comparison.png (stability analysis)")
        print("  - models/best_model.pkl (best model saved)")
        
        print("\n🏆 WINNER: " + df_results.iloc[0]['Model'])
        print(f"   Accuracy: {df_results.iloc[0]['Accuracy']:.1%}")
        print(f"   ROC-AUC: {df_results.iloc[0]['ROC-AUC']:.3f}")


if __name__ == '__main__':
    print("Model Comparison Suite with AIC/BIC")
    print("="*70 + "\n")
    
    comparison = ModelComparison()
    comparison.generate_report()