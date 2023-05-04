
# Results

## KNN

### Best parameters

```python
metric  = manhattan
k       = 5
```

### Best score

```python
score   = 0.8629629629629629
```

## Decision Tree

### Best parameters

```python
max_depth           = 29
min_samples_split   = 29
min_samples_leaf    = 27
```

### Best score

```python
score              = 0.5222222222222223
```

## Random Forest

### Best parameters

The random forest best parameter depends
on some randomness, so the result may vary.
Our way of determining the best parameter
is $`score / (depth * 10 + split + est * 5)`$.
since we want to minimize the depth and estimators,
and don't really care about the split but want to maximize the score.


```python
max_depth           = 14
min_samples_split   = 2
n_estimators        = 31 
```

### Best score

```python
score               = 1.0
```
