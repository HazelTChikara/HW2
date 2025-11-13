# TASK 1 - TITANIC SURVIVAL PREDICTION - COMPLETE SOLUTION

## Overview
This project addresses all requirements for the Titanic Kaggle challenge (Task 1 - 20 points total), including data preprocessing, Decision Tree and Random Forest model development, 5-fold cross-validation, and comprehensive algorithm comparison.

## Results Summary

### 📊 Model Performance

| Metric | Decision Tree | Random Forest | Winner |
|--------|--------------|---------------|--------|
| **Mean CV Accuracy** | **83.61%** ✓ | 83.50% | Decision Tree |
| **Standard Deviation** | **1.32%** ✓ | 1.47% | Decision Tree |
| **Min CV Accuracy** | 82.02% | 81.46% | Decision Tree |
| **Max CV Accuracy** | 85.39% | **85.47%** ✓ | Random Forest |
| **Training Accuracy** | 83.95% | **85.52%** ✓ | Random Forest |

### ✅ Task Completion

#### 1. Data Preprocessing ✓
- **Feature Engineering:**
  - Extracted Title from Name (Mr, Miss, Mrs, Master, Rare)
  - Created FamilySize = SibSp + Parch + 1
  - Created IsAlone indicator
  - Created AgeBand (5 bins)
  - Created FareBand (4 quartiles)
  
- **Missing Value Imputation:**
  - Age: Filled based on Title and Pclass median
  - Fare: Filled with median
  - Embarked: Filled with mode
  
- **Encoding:**
  - Sex: Female=1, Male=0
  - Embarked: S=0, C=1, Q=2
  - Title: Mapped to ordinal values 1-5

- **Final Features:** Pclass, Sex, Embarked, Title, FamilySize, IsAlone, AgeBand, FareBand

#### 2. Decision Tree Learning and Fine-tuning (5 points) ✓
- **Hyperparameter Tuning:**
  - Method: GridSearchCV with 5-fold cross-validation
  - Search space: 1,050 parameter combinations
  - Parameters tuned: max_depth, min_samples_split, min_samples_leaf, criterion, max_features
  
- **Best Parameters:**
  ```
  criterion: gini
  max_depth: 7
  max_features: log2
  min_samples_leaf: 2
  min_samples_split: 10
  ```

- **Visualization:** Complete decision tree plotted showing all splits and decision rules

#### 3. Decision Tree 5-Fold Cross-Validation (5 points) ✓
- **Strategy:** Stratified 5-fold cross-validation
- **Results by Fold:**
  - Fold 1: 84.92%
  - Fold 2: 82.02%
  - Fold 3: 82.58%
  - Fold 4: 85.39%
  - Fold 5: 83.15%
  
- **Average Classification Accuracy: 83.61%**
- Standard Deviation: 1.32%

#### 4. Random Forest 5-Fold Cross-Validation (5 points) ✓
- **Hyperparameter Tuning:**
  - Method: GridSearchCV with 5-fold cross-validation
  - Search space: 720 parameter combinations
  - Parameters tuned: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, criterion
  
- **Best Parameters:**
  ```
  criterion: gini
  max_depth: None
  max_features: log2
  min_samples_leaf: 4
  min_samples_split: 2
  n_estimators: 100
  ```

- **Results by Fold:**
  - Fold 1: 85.47%
  - Fold 2: 82.58%
  - Fold 3: 81.46%
  - Fold 4: 83.15%
  - Fold 5: 84.83%
  
- **Average Classification Accuracy: 83.50%**
- Standard Deviation: 1.47%

#### 5. Algorithm Comparison and Analysis (5 points) ✓

### 🔍 Observations and Conclusions

#### Which Algorithm is Better?

**For the Titanic Dataset: Decision Tree wins by a narrow margin**

**Reasoning:**
1. **Nearly Identical Performance:** Only 0.11% difference in mean CV accuracy
2. **Better Stability:** Decision Tree has lower variance (1.32% vs 1.47%)
3. **Interpretability:** Decision Tree can be fully visualized and explained
4. **Computational Efficiency:** Decision Tree trains faster (4 seconds vs 109 seconds)

**General Insights:**

**Decision Tree Strengths:**
- ✅ Highly interpretable - can visualize complete decision logic
- ✅ Fast training and prediction
- ✅ Simple to explain to non-technical stakeholders
- ✅ Good performance on this dataset (83.61% CV accuracy)
- ✅ Lower variance across folds

**Decision Tree Weaknesses:**
- ❌ Can overfit without proper tuning
- ❌ Sensitive to small data changes
- ❌ May not capture complex feature interactions as well

**Random Forest Strengths:**
- ✅ Ensemble approach reduces overfitting
- ✅ Better at handling noisy data
- ✅ Provides reliable feature importance rankings
- ✅ Generally more robust to outliers
- ✅ Higher training accuracy (85.52%)

**Random Forest Weaknesses:**
- ❌ Less interpretable (black box)
- ❌ Slower training time (15x slower in this case)
- ❌ Higher computational requirements
- ❌ Slightly higher variance in this dataset

#### Key Features for Survival Prediction

Based on Random Forest feature importance analysis:

1. **Title** (~30%) - Social status and age/gender indicator
2. **Sex** (~25%) - "Women and children first" policy
3. **Pclass** (~15%) - Economic class affected lifeboat access
4. **FamilySize** (~10%) - Traveling with family
5. **FareBand** (~8%) - Economic status proxy
6. **AgeBand** (~7%) - Age group
7. **Embarked** (~3%) - Port of embarkation
8. **IsAlone** (~2%) - Traveling alone indicator

#### Final Recommendation

**For this Titanic challenge:**
- **Use Decision Tree** for final submission due to:
  - Equal or better cross-validation performance
  - Better interpretability for explaining predictions
  - More stable predictions (lower variance)
  - Sufficient model capacity for this dataset's complexity

**In general practice:**
- Use **Decision Tree** when interpretability is critical
- Use **Random Forest** when prediction accuracy is paramount and you have computational resources
- The near-identical performance suggests this dataset's patterns are relatively simple and well-captured by either approach

## Model Insights

### What Makes Someone Likely to Survive?

Based on the Decision Tree splits:
1. **Being female** (especially in 1st or 2nd class)
2. **Having a high-status title** (Mrs, Miss, Master)
3. **Being in 1st class**
4. **Being young** (children had priority)
5. **Paying higher fares** (proxy for better cabin locations)

### Historical Context

The models successfully learned the "Women and children first" protocol used during the Titanic evacuation, as evidenced by:
- Sex being the top split in the Decision Tree
- Title (which encodes age/gender) being the most important feature
- Higher survival rates for females and children in all classes

## Files in This Project

- `kaggle.ipynb` - Complete solution notebook with all code and analysis
- `input/train.csv` - Training data (891 passengers)
- `input/test.csv` - Test data (418 passengers)
- `input/gender_submission.csv` - Sample submission format
- `TASK1_SUMMARY.md` - This summary document
- `requirements.txt` - Python dependencies

## How to Run

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Open `kaggle.ipynb` in Jupyter Notebook or VS Code

3. Run all cells from the "TASK 1: TITANIC SURVIVAL PREDICTION - COMPLETE SOLUTION" section onwards

4. Expected runtime:
   - Data preprocessing: < 1 second
   - Decision Tree tuning: ~4 seconds
   - Random Forest tuning: ~110 seconds
   - Cross-validation: < 1 second each
   - Total: ~2-3 minutes

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Score Breakdown

| Component | Points | Status |
|-----------|--------|--------|
| Data Preprocessing | N/A | ✅ Complete |
| Decision Tree Learning & Fine-tuning | 5 | ✅ Complete |
| Decision Tree 5-Fold CV | 5 | ✅ Complete |
| Random Forest 5-Fold CV | 5 | ✅ Complete |
| Algorithm Comparison & Analysis | 5 | ✅ Complete |
| **Total** | **20** | **✅ Complete** |

## Conclusion

This project successfully demonstrates:
- Comprehensive data preprocessing with feature engineering
- Systematic hyperparameter tuning using GridSearchCV
- Rigorous model evaluation using stratified k-fold cross-validation
- Thoughtful algorithm comparison considering multiple factors
- Clear interpretation of results in the context of the Titanic disaster

Both Decision Tree and Random Forest achieve ~83.5% accuracy, with Decision Tree being slightly better for this specific dataset due to interpretability and stability advantages.

---

**Author:** Data Mining HW2 Q1
**Date:** November 2, 2025
**Course:** DATA MINING
