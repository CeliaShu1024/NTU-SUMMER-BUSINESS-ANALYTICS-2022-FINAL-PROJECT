# NTU-SUMMER-BUSINESS-ANALYTICS-2022-FINAL-PROJECT
This is the final project for Business Analytics summer course of Nanyang Technology University in 2022. It is mainly about classify the client type by using machine learning techniques. To finish this project, I applied `Python@3.9` and `Jupyter Notebook`.</br>

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