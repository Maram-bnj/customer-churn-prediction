# Data

This project uses the public **IBM Telco Customer Churn** dataset: 7,043 customers of a
fictional telecom company, 21 columns (demographics, subscribed services, account
information and a `Churn` label).

The CSV is not stored in this repository. To get it:

1. Download it from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   (or with the Kaggle CLI: `kaggle datasets download -d blastchar/telco-customer-churn --unzip -p data/`).
   Alternatively, IBM publishes the same file (identical columns and 7,043 rows) on GitHub:
   ```bash
   curl -L -o data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
     https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
   ```
2. Place the file here, keeping the Kaggle name (rename the IBM file as in the command above):

```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Another location can be used with `python -m churn.train --data path/to/file.csv`.

## Columns

| Group | Columns |
|---|---|
| ID | `customerID` (dropped before training) |
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Target | `Churn` (Yes / No) |

Known quirk: `TotalCharges` is stored as text and is blank for the 11 customers with
`tenure = 0`. `churn.data.clean` converts it to a number and sets those values to 0.
