-- Churn rate and customer volume by contract type.
SELECT
    contract_type,
    COUNT(*) AS customers,
    ROUND(AVG(churn), 4) AS churn_rate,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM customer_churn
GROUP BY contract_type
ORDER BY churn_rate DESC;

-- Churn rate by recent support demand.
SELECT
    CASE
        WHEN support_tickets = 0 THEN '0'
        WHEN support_tickets BETWEEN 1 AND 2 THEN '1-2'
        ELSE '3+'
    END AS support_ticket_band,
    COUNT(*) AS customers,
    ROUND(AVG(churn), 4) AS churn_rate
FROM customer_churn
GROUP BY support_ticket_band
ORDER BY churn_rate DESC;

-- High-value customers who may deserve proactive retention outreach.
SELECT
    customer_id,
    tenure_months,
    monthly_charges,
    support_tickets,
    late_payments,
    contract_type
FROM customer_churn
WHERE monthly_charges >= 85
  AND contract_type = 'month-to-month'
  AND (support_tickets >= 2 OR late_payments >= 2)
ORDER BY monthly_charges DESC, support_tickets DESC
LIMIT 25;
