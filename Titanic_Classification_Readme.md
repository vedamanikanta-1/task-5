# Titanic Classification – Decision Tree, Random Forest & Cross-Validation

This project demonstrates a complete machine learning workflow on the **Titanic dataset**, including preprocessing, Decision Tree training, Random Forest training, overfitting analysis, feature importance interpretation, and cross-validation evaluation.

---

## 1. Project Overview

The goal is to predict whether a passenger **survived (1)** or **did not survive (0)** using:

- Decision Tree Classifier  
- Random Forest Classifier  

Machine learning tasks performed:

- Data preprocessing  
- Train/test split  
- Decision Tree training & visualization  
- Overfitting analysis  
- Random Forest training  
- Feature importance analysis  
- Cross-validation evaluation  

---

## 2. Dataset

We use the Titanic dataset from seaborn.

### Selected Features
- pclass  
- age  
- sibsp  
- parch  
- fare  
- sex  
- embarked  

### Target
- survived (0/1)

---

## 3. Data Preprocessing

Steps:
1. Load dataset  
2. Drop rows with missing target  
3. Fill missing values  
4. One-hot encode categorical variables  
5. Prepare X and y  

Decision Trees & Random Forests **do not require scaling**.

---

## 4. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

---

## 5. Decision Tree Classifier

### Training
```python
clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
clf.fit(X_train, y_train)
```

### Visualization
```python
plot_tree(clf, feature_names=X.columns, filled=True, rounded=True)
```

### Evaluation
Includes:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

---

## 6. Overfitting Analysis

Compare training vs testing accuracy:

```python
train_acc = clf_overfit.score(X_train, y_train)
test_acc  = clf_overfit.score(X_test, y_test)
```

### Controlling Overfitting
- max_depth  
- min_samples_leaf  
- min_samples_split  
- max_leaf_nodes  

---

## 7. Random Forest Classifier

### Training
```python
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
rf.fit(X_train, y_train)
```

### Evaluation
Random Forest typically gives **higher accuracy** and less overfitting.

---

## 8. Feature Importance Interpretation

```python
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.sort_values(ascending=False)
```

### Typical Importance Order
1. sex_male  
2. pclass  
3. age  
4. fare  
5. embarked_S / embarked_Q  
6. sibsp, parch  

---

## 9. Cross-Validation Evaluation

### Accuracy CV
```python
scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
```

### Precision, Recall, F1
```python
cross_val_score(rf, X, y, cv=5, scoring=make_scorer(precision_score))
```

### Interpretation
- Higher mean score → better generalization  
- Lower std deviation → more stable model  

---

## 10. Summary

| Task | Completed |
|------|-----------|
| Data Preprocessing | ✔ |
| Decision Tree Training | ✔ |
| Tree Visualization | ✔ |
| Overfitting Analysis | ✔ |
| Random Forest Training | ✔ |
| Feature Importances | ✔ |
| Cross-Validation | ✔ |

**Final insights:**
- Random Forest outperforms Decision Tree  
- Gender, class, and age are top predictors  
- Tree depth control reduces overfitting  
- Cross-validation gives reliable performance estimate  
