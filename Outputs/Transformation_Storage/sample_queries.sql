SELECT * FROM churn_local LIMIT 10;
SELECT * FROM churn_kaggle LIMIT 10;
SELECT CustomerId, TotalBalance FROM churn_local WHERE TotalBalance > 50000;
SELECT AVG(ChargePerMonth) FROM churn_kaggle;
