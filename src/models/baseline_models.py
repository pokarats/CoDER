

"""
rule-base model
input: doc as concepts
process: concepts found that are also in the icd9 concepts == labels
output: labels

tfidf + classifier
input: vectorized concepts found in each doc -->tfidf input features
linear classifier output to |unique labels in semantic_types_mimic|
output: labels

label cluster (optional add to baseline) by
1) semantic types
2)
"""