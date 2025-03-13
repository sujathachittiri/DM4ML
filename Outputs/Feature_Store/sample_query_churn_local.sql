
        SELECT CustomerId, TotalBalance, TenurePerProduct, ActivityScore
        FROM churn_local
        WHERE TotalBalance > 50000;
        