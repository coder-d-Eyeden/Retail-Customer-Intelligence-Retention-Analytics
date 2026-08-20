-- ============================================================
-- Retail Customer Intelligence & Retention Analytics
-- SQL Analysis Queries
--
-- Dataset: clean_transactions.csv loaded into `transactions` table
-- Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate,
--           UnitPrice, CustomerID, Country, Revenue
-- ============================================================

-- ============================================================
-- 1. BASIC REVENUE METRICS
-- ============================================================

-- Total revenue
SELECT SUM(Revenue) AS total_revenue FROM transactions;

-- Total customers, orders, products
SELECT
    COUNT(DISTINCT CustomerID) AS unique_customers,
    COUNT(DISTINCT InvoiceNo) AS unique_orders,
    COUNT(DISTINCT StockCode) AS unique_products,
    ROUND(SUM(Revenue), 2) AS total_revenue,
    ROUND(AVG(Revenue), 2) AS avg_line_revenue
FROM transactions;

-- ============================================================
-- 2. MONTHLY REVENUE TREND
-- ============================================================

SELECT
    strftime('%Y-%m', InvoiceDate) AS month,
    SUM(Revenue) AS revenue,
    COUNT(DISTINCT InvoiceNo) AS orders,
    COUNT(DISTINCT CustomerID) AS active_customers
FROM transactions
GROUP BY 1
ORDER BY 1;

-- ============================================================
-- 3. REVENUE BY COUNTRY
-- ============================================================

SELECT
    Country,
    COUNT(DISTINCT CustomerID) AS customers,
    COUNT(DISTINCT InvoiceNo) AS orders,
    ROUND(SUM(Revenue), 2) AS revenue,
    ROUND(SUM(Revenue) / COUNT(DISTINCT CustomerID), 2) AS revenue_per_customer
FROM transactions
GROUP BY Country
ORDER BY revenue DESC;

-- ============================================================
-- 4. TOP 20 CUSTOMERS BY REVENUE (PARETO)
-- ============================================================

WITH customer_revenue AS (
    SELECT
        CustomerID,
        COUNT(DISTINCT InvoiceNo) AS orders,
        ROUND(SUM(Revenue), 2) AS revenue
    FROM transactions
    GROUP BY CustomerID
),
ranked AS (
    SELECT *,
        RANK() OVER (ORDER BY revenue DESC) AS revenue_rank,
        ROUND(SUM(revenue) OVER (ORDER BY revenue DESC) / SUM(revenue) OVER () * 100, 1) AS cumulative_pct
    FROM customer_revenue
)
SELECT * FROM ranked WHERE revenue_rank <= 20;

-- ============================================================
-- 5. REPEAT vs ONE-TIME CUSTOMERS
-- ============================================================

WITH orders AS (
    SELECT CustomerID, COUNT(DISTINCT InvoiceNo) AS n_orders
    FROM transactions
    GROUP BY CustomerID
)
SELECT
    CASE WHEN n_orders = 1 THEN 'One-Time' ELSE 'Repeat' END AS customer_type,
    COUNT(*) AS n_customers,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 1) AS pct
FROM orders
GROUP BY 1;

-- ============================================================
-- 6. PRODUCT ANALYSIS: TOP PRODUCTS BY REVENUE
-- ============================================================

SELECT
    StockCode,
    MAX(Description) AS Description,
    COUNT(DISTINCT CustomerID) AS unique_buyers,
    SUM(Quantity) AS total_qty,
    ROUND(SUM(Revenue), 2) AS revenue
FROM transactions
GROUP BY StockCode
ORDER BY revenue DESC
LIMIT 20;

-- ============================================================
-- 7. PRODUCT CATEGORY REVENUE DENSITY
--     (Products appearing in >100 orders)
-- ============================================================

SELECT
    StockCode,
    MAX(Description) AS Description,
    COUNT(DISTINCT InvoiceNo) AS order_appearances,
    SUM(Quantity) AS total_qty,
    ROUND(SUM(Revenue), 2) AS revenue
FROM transactions
GROUP BY StockCode
HAVING order_appearances > 100
ORDER BY revenue DESC
LIMIT 30;

-- ============================================================
-- 8. CUSTOMER RANKING WITH WINDOW FUNCTIONS
-- ============================================================

WITH customer_stats AS (
    SELECT
        CustomerID,
        COUNT(DISTINCT InvoiceNo) AS orders,
        ROUND(SUM(Revenue), 2) AS total_revenue,
        ROUND(AVG(Revenue), 2) AS avg_revenue,
        MIN(InvoiceDate) AS first_purchase,
        MAX(InvoiceDate) AS last_purchase
    FROM transactions
    GROUP BY CustomerID
)
SELECT *,
    RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank,
    NTILE(5) OVER (ORDER BY total_revenue DESC) AS revenue_quintile,
    julianday('2011-12-09') - julianday(last_purchase) AS days_since_last_purchase
FROM customer_stats
ORDER BY revenue_rank
LIMIT 30;

-- ============================================================
-- 9. HOURLY PURCHASE PATTERNS
-- ============================================================

SELECT
    CAST(strftime('%H', InvoiceDate) AS INTEGER) AS hour,
    COUNT(*) AS transactions,
    ROUND(SUM(Revenue), 2) AS revenue
FROM transactions
GROUP BY 1
ORDER BY 1;

-- ============================================================
-- 10. MONTHLY COHORT SIZE (proxy for retention SQL analysis)
-- ============================================================

WITH first_purchase AS (
    SELECT CustomerID, MIN(InvoiceDate) AS first_date
    FROM transactions
    GROUP BY CustomerID
)
SELECT
    strftime('%Y-%m', fp.first_date) AS cohort_month,
    COUNT(DISTINCT t.CustomerID) AS cohort_size
FROM transactions t
JOIN first_purchase fp ON t.CustomerID = fp.CustomerID
GROUP BY 1
ORDER BY 1;

-- ============================================================
-- 11. PARETO SUMMARY: TOP 20% REVENUE CONTRIBUTION
-- ============================================================

WITH customer_revenue AS (
    SELECT CustomerID, ROUND(SUM(Revenue), 2) AS revenue
    FROM transactions
    GROUP BY CustomerID
),
ranked AS (
    SELECT *,
        NTILE(5) OVER (ORDER BY revenue DESC) AS quintile
    FROM customer_revenue
)
SELECT
    quintile,
    COUNT(*) AS customers,
    ROUND(SUM(revenue), 2) AS total_revenue,
    ROUND(SUM(revenue) * 100.0 / (SELECT SUM(revenue) FROM customer_revenue), 1) AS pct_revenue
FROM ranked
GROUP BY 1
ORDER BY 1;

-- ============================================================
-- 12. CUSTOMER LIFETIME VALUE PROXY (RFM-style in SQL)
-- ============================================================

WITH customer_stats AS (
    SELECT
        CustomerID,
        CAST(julianday('2011-12-09') - julianday(MAX(InvoiceDate)) AS INTEGER) AS recency_days,
        COUNT(DISTINCT InvoiceNo) AS frequency,
        ROUND(SUM(Revenue), 2) AS monetary
    FROM transactions
    GROUP BY CustomerID
)
SELECT *,
    CASE
        WHEN recency_days <= 30 AND frequency >= 10 AND monetary >= 1000 THEN 'Champion'
        WHEN recency_days <= 60 AND frequency >= 5 THEN 'Loyal'
        WHEN recency_days <= 90 THEN 'Potential Loyalist'
        WHEN recency_days <= 180 THEN 'At Risk'
        ELSE 'Lost'
    END AS segment
FROM customer_stats
ORDER BY monetary DESC;
