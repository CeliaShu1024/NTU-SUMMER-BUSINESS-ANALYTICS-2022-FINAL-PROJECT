# NTU-SUMMER-BUSINESS-ANALYTICS-2022-FINAL-PROJECT
This is the final project for Business Analytics summer course of Nanyang Technology University in 2022. </br>
The learning outcomes of this project are:
1. Implement classification of samples by using machine learnining techniques.
2. Compare the performance of models and suggest the best solution for business banks to evaluate credit of potential clients.</br>

To finish this project, I applied `Python@3.9` and `Jupyter Notebook`.</br>

# Content of Repository
`Credit Card Default II.csv`: Dataset.</br>
`Project.ipynb`: Project source code and output of each code blocks.</br>
Note that this repository does not include the report document because of the copyright.</br>

# Configuration of Programming Environment
Programming IDE: `Virtual Studio Code`</br>
Programming Language: `Python@3.9.0`</br>
Liraries: `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `seaborn`, `matplotlib`</br>

# Dataset Info
A small dataset provided by NTU. This dataset contains 2000 samples randomly select from a credit card default record of a bank. It is a small dataset contains 5 attributes. The first four attributes are used as feature inputs to predict prospective clients and the 5th attribute is used as the predict target. The legend of this dataset is shown below.</br>
<table>
    <tr>
        <th>Attribute</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>clientid</td>
        <td>An integer number that represents the corresponding client.</td>
    </tr>
    <tr>
        <td>income</td>
        <td>A float number that represents the income of a client (measured in US Dollars).</td>
    </tr>
    <tr>
        <td>age</td>
        <td>Age of a client; a higher value is further old.</td>
    </tr>
    <tr>
        <td>loan</td>
        <td>Loan of a client (measured in US Dollars).</td>
    </tr>
    <tr>
        <td>default</td>
        <td>A Boolean variable represents whether the client settles the loan or not; 0 means “settled” and 1 means “unsettled”.</td>
    </tr>
</table>
</br>

# Data Pre-Processing
Through observation, some negative values are in the `age` attribute. Therefore, before formal data pre-processing stage, I took the absolute values of the whole attribute. Then I got the statistical description of the dataset, which is shown below.
<table>
    <tr>
        <th>#</th>
        <th>Column</th>
        <th>Non-Null Count</th>
        <th>Dtype</th>
    </tr>
    <tr>
        <td>0</td>
        <td>clientid</td>
        <td>2000 non-null</td>
        <td>int64</td>
    </tr>
    <tr>
        <td>1</td>
        <td>income</td>
        <td>2000 non-null</td>
        <td>float64</td>
    </tr>
    <tr>
        <td>2</td>
        <td>age</td>
        <td>1997 non-null</td>
        <td>float64</td>
    </tr>
    <tr>
        <td>3</td>
        <td>loan</td>
        <td>2000 non-null</td>
        <td>float64</td>
    </tr>
    <tr>
        <td>4</td>
        <td>default</td>
        <td>2000 non-null</td>
        <td>int64</td>
    </tr>
</table>
</br>
According to the above, there are 3 .Since the dataset is literally small, I did not drop the samples with null value but filled the null values with mean value of `age` attribute.</br>
I applied boxplot to determine the existence of outlier. The graph shows that there are several outliers in `loan` attribute.</br>
![boxplot](https://github.com/CeliaShu1024/NTU-SUMMER-BUSINESS-ANALYTICS-2022-FINAL-PROJECT/blob/main/Plots/boxplot.png)
</br>
To drop outliers, I removed samples with z-score less than 2.5. After that I sketched the client characteristics. Based on this sketch, the income levels of settled and unsettled clients are basically same. The age of clients who have settled their loan falls between 18 and 35, and peaks in the range [25, 30]. In contrast, the age of unsettled clients falls between 20 and 65, and is mainly concentrated in the [35, 65] interval. In terms of loan amount, the loan amount of settled clients is approximately normally distributed in the interval [2000, 13000], while unsettled clients are more likely to apply small loan.</br>
![client characteristics](https://github.com/CeliaShu1024/NTU-SUMMER-BUSINESS-ANALYTICS-2022-FINAL-PROJECT/blob/main/Plots/statistics.png)
</br>
In order to make each attribute to have the same effect in the regression analysis, I normalized the dataset by `StandardScaler()` after the above steps. </br>

# Attribute Relationship
I sketched the relation plot of the attributes. It could be observed easily that “income” attribute has the greatest influence on “default” attribute.</br>
![relplot](https://github.com/CeliaShu1024/NTU-SUMMER-BUSINESS-ANALYTICS-2022-FINAL-PROJECT/blob/main/Plots/relplot.png)
</br>

# Metrics & Model Evaluation Function
Since this study involved a simple binary classification problem, I used `confusion matrix` and `receiver operating characteristic (ROC)` as metrics for model evaluation.</br>
A template of confusion matrix is shown below.</br>
![confusion_matrix](https://github.com/CeliaShu1024/NTU-SUMMER-BUSINESS-ANALYTICS-2022-FINAL-PROJECT/blob/main/Plots/confusion%20matrix.png)
According to these data, several derived metrics could be calculated:</br>
1. $accuracy=\frac{TP+TN}{T+N}$</br>
2. $precision=\frac{TP}{P}$</br>
3. $recall=\frac{TP}{TP+FN}$</br>
4. F1-Score: $f1=\frac{2 \times precision \times recall}{precision+recall}$</br>
5. True-Positive Rate: $TPR=\frac{TP}{P}$</br>
6. False-Positive Rate: $FPR=\frac{FP}{N}$</br>

Receiver operating characteristic (ROC) is a curve drawn with FPR and TPR as the x and y axes. Compared with F1-score, another metrics used to evaluate the classification model, the ROC curve can better reflect the performance of the binary classifier. The ROC-AUC score is the integral of the ROC curve over the FPR, and the closer to 1, the better the model performance. Therefore, ROC-AUC score can be calculated as:</br>

7. ROC-AUC: $auc_{m}=\int_{0}^{1}{FPR}d{TPR}$</br>

Grounded in these two metrics, I proposed a evaluation function to represent the performance of the model. The expression of this fuction is $P(m)=0.3 \times accuracy_{m}+0.2 \times f1_{m}+0.5 \times auc_{m}$, where $m$ refers to the current model configuration.

# Parameter Tuning
Based on the model evaluaiton function mentioned above, I tuned parameters of each model to get the fairest model comparison. After that, I sketched the model performance curves. The following are the performance curves of logistic regression, CART decision tree, gradient boosting decision tree and extreme gradient boosting (xgboost).</br>
![logistic_regression](https://github.com/CeliaShu1024/NTU-SUMMER-BUSINESS-ANALYTICS-2022-FINAL-PROJECT/blob/main/Plots/LogisticRegression.png)
![CART_Decision_Tree](https://github.com/CeliaShu1024/NTU-SUMMER-BUSINESS-ANALYTICS-2022-FINAL-PROJECT/blob/main/Plots/DecisionTree.png)
![Gradient_Boosting_Decision_Tree](https://github.com/CeliaShu1024/NTU-SUMMER-BUSINESS-ANALYTICS-2022-FINAL-PROJECT/blob/main/Plots/GradientBoosting.png)
![eXtreme_Gradient_Boosting](https://github.com/CeliaShu1024/NTU-SUMMER-BUSINESS-ANALYTICS-2022-FINAL-PROJECT/blob/main/Plots/xgb.png)
</br>

# Experiment Results
The performance-related results are listed below.</br>
<table>
    <tr>
        <th>Model</th>
        <th>Logistic Regression</th>
        <th>Naive Bayes</th>
        <th>CART Decision Tree</th>
        <th>Gradient Boosting Decision Tree</th>
        <th>eXtreme Gradient Boost</th>
    </tr>
    <tr>
        <th>Accuracy (%)</th>
        <td>94.17</td>
        <td>93.10</td>
        <td>98.16</td>
        <td>98.93</td>
        <td>98.93</td>
    </tr>
    <tr>
        <th>F1-Score (%)</th>
        <td>96.61</td>
        <td>96.03</td>
        <td>98.91</td>
        <td>99.37</td>
        <td>99.37</td>
    </tr>
    <tr>
        <th>ROC-AUC Score (%)</th>
        <td>98.02</td>
        <td>97.12</td>
        <td>98.24</td>
        <td>99.93</td>
        <td>99.93</td>
    </tr>
    <tr>
        <th>Model Score</th>
        <td>96.58</td>
        <td>95.69</td>
        <td>98.42</td>
        <td>99.52</td>
        <td>99.52</td>
    </tr>
</table>
</br>
Due to the small size of the dataset and fewer dimensions, all of our selected models can achieve over 90% accuracy. In terms of performance, the two gradient boosting trees have the most advantages. Therefore, we recommend that commercial banks use gradient boosting trees or extreme gradient boosting trees to evaluate credits of prospective clients. However, as mentioned earlier, the dataset used in this study is not large enough to capture the difference in performance between the two gradient boosted trees. Theoretically, we recommend the extreme gradient boosting tree algorithm as an auxiliary tool for credit evaluation based on larger user datasets.</br>

# Conclusion
In this study, several AI models are used to find out the potential influencing factors of the credit cards defaults and to predict the credit cards defaults, the models including logistic regression, Naïve Bayes, classic decision tree, gradient boosting decision tree and extreme gradient boosting decision tree. Having taken accuracy, F1-Score and ROC-AUC Score into consideration, we concluded that gradient boosting decision tree and extreme gradient boosting decision tree have the best overall performance and are the most appropriate models, which prediction score approximately approached 100%, especially since data like credit card default is usually imbalanced.</br>
Thus, commercially, we recommend banks to use these two algorithms for client credit classification, the company can use the previous credit card records to make prediction models to identify potential credit card defaults. When processing credit card applications and setting appropriate credit limits for cardholders, they should also pay more attention to the salient features leading to default and be more cautious.</br>
Nevertheless, due to the limitation of dataset size, the differences between these two gradient boosting algorithms were not captured. Concerning with theories and principles behind the algorithms, we also suggest commercial banks to use extreme gradient boosting
decision tree for large-dataset-based client classification. In the future, we may expand the data source and establish more prediction features, such as different types of credit payment experience and credit history length of customers.
