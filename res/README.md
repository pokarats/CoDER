## Baseline Results
### Rule-Based

#### Top-50
| Model               | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | Acc (Macro) | Acc (Micro) |
|---------------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|
| No Extension (None) | **45.02** |   18.88   |   19.58    |   26.60    |    12.57    |    15.34    |
| ALL                 |   16.86   | **26.54** |   18.11    |   20.62    |    9.89     |    11.50    |
| Best                |   40.71   |   20.34   | **23.75**  | **27.13**  |  **13.07**  |  **15.69**  |

(09/16/2022)

#### Full
| Model               | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | Acc (Macro) | Acc (Micro) |
|---------------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|
| No Extension (None) |   8.83    |   14.47   |    2.19    |   10.97    |    1.13     |    5.80     |
| ALL                 |   8.83    |   14.47   |    2.19    |   10.97    |    1.13     |    5.80     |
| Best                |   8.83    |   14.47   |    2.19    |   10.97    |    1.13     |    5.80     |

(09/16/2022)

**NOTE:** that for the Full Version, _the extension criteria (finding < 1 ICD label corresponding to input CUIs)_ were
never met. Hence, identical results: i.e. No extensions were triggered.

### Non-Deep-Learning Models

Overall, using 2-gram feature option improves performance across all models in both **Top-50** and **Full** versions. 
Our best LR model with **UMLS CUIS** input type shows
comparable performance with LR results from **text** input type reported in 
[Vu et al. (2020)](https://arxiv.org/abs/2007.06351). Our best SGD and LR results are also comparable with BiGRU 
results reported also in [Vu et al. (2020)](https://arxiv.org/abs/2007.06351). This is noted in both **Top-50** and 
**Full** versions.

#### Top-50 Version

| Model                    | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | AUC (Macro) | AUC (Micro) |    P@5    |
|--------------------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:---------:|
| LR[^lr] 1-gram           |   51.87   |   47.97   |   46.38    |   49.84    |    81.26    |    84.09    |   48.86   |
| LR[^lr] 2-gram           |   63.61   |   47.29   |  _49.56_   |  _54.25_   |   _84.89_   |   _87.84_   |  _54.59_  |
| SGD[^sgd] 1-gram         |   50.14   | **57.87** |   50.95    |   53.73    |    83.75    |    86.56    |   52.16   |
| SGD[^sgd] 2-gram         |   60.08   |   52.12   | **51.59**  | **55.82**  |  **85.27**  |    88.16    |   54.93   |
| SVM[^svm] 1-gram         |   51.45   |   43.72   |   43.55    |   47.28    |    79.70    |    81.85    |   46.92   |
| SVM[^svm] 2-gram         |   67.52   |   41.66   |   45.90    |   51.53    |    84.02    |    86.92    |   53.90   |
| Stacked[^stacked] 2-gram | **73.79** |   36.34   |   41.80    |   48.70    |    84.94    |  **88.30**  | **55.28** |

(09/17/2022)  


[^lr]: With default settings for [scikit-learn 1.0.2](https://scikit-learn.org/1.0/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regression#sklearn.linear_model.LogisticRegression) 
except `class_weight=='balanced'`, `C==1000` and `max_iter==1000`.

[^sgd]: With default settings for [scikit-learn 1.0.2](https://scikit-learn.org/1.0/modules/generated/sklearn.linear_model.SGDClassifier.html?highlight=sgd#sklearn.linear_model.SGDClassifier) 
except `class_weight=='balanced'`, `eta0=0.01`, `learning_rate='adaptive'`, and `loss='modified_huber'`.

[^svm]: With default settings for [scikit-learn 1.0.2](https://scikit-learn.org/1.0/modules/generated/sklearn.svm.LinearSVC.html?highlight=svc#sklearn.svm.LinearSVC) 
except `class_weight=='balanced'` and `C==1000`.

[^stacked]: With defaults settings for [OneVsRestClassifier](https://scikit-learn.org/1.0/modules/generated/sklearn.multiclass.OneVsRestClassifier.html?highlight=onevsrestclassifier#sklearn.multiclass.OneVsRestClassifier) 
and changes to each component as above.


#### Full Version
 
| Model             | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro)  | AUC (Macro) | AUC (Micro) |     P@5     |
|-------------------|:---------:|:---------:|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| LR[^lrf] 1-gram   |   36.51   |   34.58   |    6.37    |    35.52    |    84.54    |    91.25    |    56.70    |
| LR[^lrf] 2-gram   |  _58.93_  |  _32.36_  |   _5.55_   | **_41.78_** | **_86.41_** |   _96.04_   | **_67.63_** |
| SGD[^sgdf] 1-gram |   21.90   | **56.59** |    8.06    |    31.58    |    84.69    |  **96.06**  |    30.77    |
| SGD[^sgdf] 2-gram |   30.29   |   49.19   |  **8.09**  |    37.50    |    84.72    |    95.94    |    42.08    |
| SVM[^svmf] 1-gram |   38.43   |   28.66   |    4.63    |    32.83    |    81.54    |    67.49    |    50.81    |
| SVM[^svmf] 2-gram | **69.51** |   24.95   |    3.61    |    36.72    |    82.91    |    86.43    |    63.71    |

(09/17/2022, 09/20/2022) 


[^lrf]: With default settings for [scikit-learn 1.0.2](https://scikit-learn.org/1.0/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regression#sklearn.linear_model.LogisticRegression) 
except `class_weight=='balanced'`, `C==1000` and `max_iter==5000`.

[^sgdf]: With default settings for [scikit-learn 1.0.2](https://scikit-learn.org/1.0/modules/generated/sklearn.linear_model.SGDClassifier.html?highlight=sgd#sklearn.linear_model.SGDClassifier) 
except `class_weight=='balanced'`, `eta0=0.01`, `learning_rate='adaptive'`, and `loss='modified_huber'`.

[^svmf]: With default settings for [scikit-learn 1.0.2](https://scikit-learn.org/1.0/modules/generated/sklearn.svm.LinearSVC.html?highlight=svc#sklearn.svm.LinearSVC) 
except `class_weight=='balanced'` and `C==1000`.


### LAAT Results

Our implementation reproduced the results for the **Top-50** and **Full** versions of the dataset for the **text input** 
type as reported in [Vu et al. (2020)](https://arxiv.org/abs/2007.06351). We report the results from using **UMLS CUIs** 
as input tokens below.

| Model      | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro)  | AUC (Macro) | AUC (Micro) |  P@5  |
|------------|:---------:|:---------:|:----------:|:-----------:|:-----------:|:-----------:|:-----:|
| Text Top50 |   75.60   |   66.95   |   66.55    |  **71.01**  |    92.79    |    94.6     | 67.28 |
| CUI Top50  |   68.75   |   47.38   |   50.55    |    56.10    |    86.16    |    89.26    | 57.50 |
| Text Full  |   65.70   |   50.64   |    9.87    |  **57.20**  |    89.84    |    98.56    | 80.91 |
| CUI Full   |   64.93   |   36.59   |    6.25    |    46.8     |    84.38    |    97.74    | 73.90 |

(Results from 10/15/2022 run on [Git Commit@36dda76](https://github.com/pokarats/CoDER/commit/36dda76d28e2a9606688016a770d0bf1129104fe))


## Extension Results

### KGE

All models reported here were with CUI input type. Baseline W2V were re-run with pruned CUIs for comparison; hence,
the results are different from what was reported in the earlier section.

| Model               | P (Micro)  | R (Micro) | F1 (Macro) | F1 (Micro) | AUC (Macro) | AUC (Micro) |  P@5  |
|---------------------|:----------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:-----:|
| Base Top50          | **69.71**  |   52.05   |   54.02    |   59.60    |    87.17    |    90.47    | 59.68 |
| Case4 Top50         |   68.46    |   59.01   |   58.94    | **63.38**  |    88.22    |    91.13    | 60.69 |
| W2V Top50           |   64.90    |   55.03   |   53.53    |   59.56    |    86.07    |    89.40    | 58.06 |
| Base Full[^basef]   |   62.11    |   34.84   |    5.53    |   44.64    |    86.55    |    97.99    | 71.73 |
| Case4 Full[^case4f] |   63.79    |   37.61   |    6.50    | **47.32**  |    85.60    |    97.89    | 73.51 |
| W2V Full            | **65.78**  |   35.44   |    5.70    |   46.07    |    84.92    |    97.77    | 73.31 |

(Results from 12/26/2022 run on [Git Commit@c658090](https://github.com/pokarats/CoDER/commit/c658090f63bd706a28319ef7eac15dfb81082c5e))

[^basef]: With `u==256` and `da==256`.
[^case4f]: With `u==256` and `da==256`.

### Combined KGE + Text

We experimented on with the case4 KGE as this is the best-performming KGE for the CUI input type.

#### Late-Fusion 

##### Shared LSTM Encoder

| Model       | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | AUC (Macro) | AUC (Micro) |  P@5  |
|-------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:-----:|
| Case4 Top50 |   71.52   |   68.23   |   65.52    |   69.83    |    91.11    |    98.68    | 65.69 |
| Case4 Full  |   63.13   |   50.10   |    6.25    |   55.87    |    84.38    |    97.74    | 78.76 |

(Results from 12/26/2022 run on [Git Commit@c658090](https://github.com/pokarats/CoDER/commit/c658090f63bd706a28319ef7eac15dfb81082c5e))

##### Separate LSTM Encoder

Post-LAAT Aggregator layer was used to combine the two input representations.

| Model       | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | AUC (Macro) | AUC (Micro) |  P@5  |
|-------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:-----:|
| Case4 Top50 |   72.20   |   68.15   |   65.20    |   70.11    |    92.04    |    94.12    | 66.47 |
| Case4 Full  |   60.21   |   51.07   |    9.92    |   55.27    |    90.35    |    98.57    | 76.94 |

(Results from 08/10-12/2023 run on [Git Commit@fc63f24](https://github.com/pokarats/CoDER/commit/fc63f249055069cceba27dee23bc1c09303f2829))

LAAT Layer was used to combine the two input represetnations.

| Model       | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | AUC (Macro) | AUC (Micro) |  P@5  |
|-------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:-----:|
| Case4 Top50 |   73.97   |   68.71   |   67.35    |   71.25    |    92.87    |    94.63    | 67.02 |
| Case4 Full  |   65.60   |   50.56   |    9.99    |   57.11    |    88.06    |    98.38    | 80.47 |

(Results from 08/10-12/2023 run on [Git Commit@fc63f24](https://github.com/pokarats/CoDER/commit/fc63f249055069cceba27dee23bc1c09303f2829))


### GNN: 2-layer GCN Baseline Results

Baseline 2-layer GCN model as encoder without the LAAT Attention mechanism. Preliminary experiments indicated that using
LAAT layers in-lieu of 'mean' / 'sum' readout functions was not beneficial to model's performance.
All hypermarameters remain the same as in the baseline LAAT model.

Baseline graph construction approaches rely on shared TUI or relations from KG.

#### Preliminary Baseline: CUIs connected if they belong to the same TUI

'mean' readout function was used for this experiment.

| Model       | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | AUC (Macro) | AUC (Micro) |    P@5    |
|-------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:---------:|
| Base Top50  | **63.19** |   35.40   |   36.25    |   45.38    |    80.96    |    84.79    |   49.82   |
| Case4 Top50 |   62.11   | **36.01** | **36.42**  | **45.59**  |  **81.88**  |  **85.43**  | **50.02** |
| Base Full   |   52.56   |   18.89   |    2.88    |   27.79    |    80.12    |    96.71    |   54.74   |
| Case4 Full  | **53.40** | **19.19** |    2.88    | **28.24**  |  **80.96**  |  **96.86**  | **55.72** |

(Results from 5/22-24/2023 run on [Git Commit@ac47a9](https://github.com/pokarats/CoDER/commit/ac47a9da0124f4dcd6d155f634217a31bb685c91))

#### KG Graph Construction Method: CUIs connected if they have shared relations

Same as the basic GNN approach above, but baseline graph construction utilizes KGE relation information.
CUIs are connected if they have a relation between them in the KG.

##### 'mean' aggregator readout

| Model       | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | AUC (Macro) | AUC (Micro) |    P@5    |
|-------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:---------:|
| Base Top50  |   67.03   |   36.18   |   38.26    |   46.99    |    83.85    |    86.83    |   52.62   |
| Case4 Top50 |   67.67   |   37.72   |   38.68    |   48.44    |    84.46    |    87.28    |   53.70   |
| W2V Top50   |   64.54   |   21.99   |   21.01    |   32.80    |    76.77    |    80.89    |   43.09   |
| Base Full   |   64.66   |   18.59   |    2.41    |   28.87    |    83.56    |    97.31    |   62.73   |
| Case4 Full  |   63.68   | **19.77** |  **2.82**  | **30.17**  |  **84.65**  |  **97.46**  | **63.60** |
| W2V Full    | **67.78** |   16.69   |    1.18    |   26.79    |    84.47    |    97.40    |   60.82   |

##### 'sum' aggregator readout

| Model       | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | AUC (Macro) | AUC (Micro) |    P@5    |
|-------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:---------:|
| Base Top50  |   66.52   |   42.64   |   42.98    |   51.97    |    83.77    |    86.82    |   53.95   |
| Case4 Top50 | **67.81** | **45.02** | **47.33**  | **54.12**  |  **84.55**  |  **87.41**  | **56.00** |
| W2V Top50   |   65.69   |   38.68   |   38.76    |   48.68    |    82.98    |    86.26    |   51.97   |
| Base Full   |   62.03   |   16.75   |    1.47    |   26.38    |    76.48    |    96.38    |   57.08   |
| Case4 Full  |   55.53   |   17.81   |    1.80    |   26.97    |    77.19    |    96.08    |   55.60   |
| W2V Full    |   62.94   |   19.26   |    1.93    |   29.50    |    76.67    |    96.44    |   60.27   |

(Results from 8/5-8/2023 run on [Git Commit@2bcb60](https://github.com/pokarats/CoDER/commit/2bcb60ab8ebde0998e98c0f2eefb07b7a9baeabf)

### Graph Construction Method with EHR Structure/Clinical Reasoning 

Following insight from manual annotation and the idea in  [Learning EHR Graphical Structures with Graph Convolutional Transformer](https://ojs.aaai.org/index.php/AAAI/article/view/5400),
we construction each graph representing an input document following the following heuristic:

- CUI entities are grouped based on whether they are considered diagnostic, procedure, concept, or laboratory entities.
- Conditional probabilities of the co-occurrences of CUIs across these groups are pre-calculated from the training set
- During graph construction, edges are connected if the conditional probability between the CUIs exceed a threshold
  - **experiment thresholds: 0.3, 0.5, 0.7, 0.8**

An additional experiment was performed to combine the EHR-guided approach with the baseline KG relations. The motivation
is to see if combining the two approaches, resultig in more edges and fewer disjointed subgraphs, will help model learn
more useful information. Results from combining the two approaches seem to have mixed results.

Best results for the Top50 version are from the experiment with 0.7 threshold and with 0.5 for the Full version.

#### 'sum' aggregator readout, comparing results from 0.7 and 0.5 threshold for Top50 and Full

| Model                | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | AUC (Macro) | AUC (Micro) |    P@5    |
|----------------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:---------:|
| Case4 Top50 0.7      |   65.24   | **48.41** | **48.71**  | **55.58**  |  **84.72**  |  **87.71**  |   56.22   |
| Case4 Top50 0.7[^kg] | **69.34** |   43.99   |   46.58    |   53.83    |    84.44    |    87.64    | **56.33** |
| Case4 Top50 0.5      |   66.25   |   41.57   |   43.76    |   51.09    |    83.90    |    86.88    |   55.60   |
| Case4 Full 0.7       | **61.17** |   17.88   |    2.12    |   26.67    |    75.08    |    96.20    |   58.56   |
| Case4 Full 0.5[^kg]  |     -     |     -     |     -      |     -      |      -      |      -      |     -     |
| Case4 Full 0.5       |   60.69   | **18.91** |  **2.23**  | **28.84**  |  **76.10**  |  **96.40**  | **59.32** |

[^kg] with connections between CUIs also when they shre relations in KG .

#### 'mean' aggregator readout, comparing results from 0.7 and 0.5 threshold for Top50 and Full

| Model           | P (Micro) | R (Micro) | F1 (Macro) | F1 (Micro) | AUC (Macro) | AUC (Micro) | P@5 |
|-----------------|:---------:|:---------:|:----------:|:----------:|:-----------:|:-----------:|:---:|
| Case4 Top50 0.7 |     -     |     -     |     -      |     -      |      -      |      -      |  -  |
| Case4 Full 0.5  |     -     |     -     |     -      |     -      |      -      |      -      |  -  |

(Results from 9/14-9/18/2023 run on [Git Commit@fdec63](https://github.com/pokarats/CoDER/commit/fdec630303cdda2d99cd5fce98bad0c86c7ce6bf)