-- ============================================================================
-- schema.sql
--
-- Purpose: Defines the normalized relational schema for the Retail Sales
-- Analytics & Forecasting Platform.
--
-- Design Notes
-- ------------
-- The source data (data/cleaned/retail_sales_cleaned.csv) is a single flat
-- file where customer, product, and order information is repeated on every
-- row. This schema normalizes that into four related tables, following
-- Third Normal Form (3NF):
--
--   customers   -- one row per unique customer
--   products    -- one row per unique product
--   orders      -- one row per unique order (the "order header")
--   order_items -- one row per product line within an order (the "order lines")
--
-- Why this matters: a customer's name is now stored exactly once, no matter
-- how many orders they place. A product's category is stored exactly once,
-- no matter how many times it's sold. This eliminates data duplication and
-- the update-inconsistency risk that comes with it.
-- ============================================================================

CREATE DATABASE IF NOT EXISTS retail_analytics
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE retail_analytics;

-- Drop tables in reverse dependency order if re-running this script
-- (order_items depends on orders and products, so it must be dropped first)
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

-- ----------------------------------------------------------------------------
-- Table: customers
-- One row per unique customer. customer_id is a natural key coming from our
-- source system (e.g. CUST-00001), not an auto-increment surrogate, because
-- it's already guaranteed unique and meaningful across the pipeline.
-- ----------------------------------------------------------------------------
CREATE TABLE customers (
    customer_id    VARCHAR(20)  NOT NULL,
    customer_name  VARCHAR(150) NOT NULL,
    segment        VARCHAR(30)  NOT NULL,
    country        VARCHAR(60)  NOT NULL,
    city           VARCHAR(80)  NOT NULL,
    state          VARCHAR(80)  NOT NULL,
    region         VARCHAR(20)  NOT NULL,
    created_at     TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (customer_id),
    -- Indexes: we will frequently filter/group by region and segment in
    -- Module 7 (Business Analytics) and Module 10 (Customer Segmentation),
    -- so we index them now rather than waiting for a slow query to force it.
    INDEX idx_customers_region (region),
    INDEX idx_customers_segment (segment)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- Table: products
-- One row per unique product.
-- ----------------------------------------------------------------------------
CREATE TABLE products (
    product_id    VARCHAR(20)  NOT NULL,
    product_name  VARCHAR(150) NOT NULL,
    category      VARCHAR(50)  NOT NULL,
    sub_category  VARCHAR(50)  NOT NULL,
    PRIMARY KEY (product_id),
    INDEX idx_products_category (category),
    INDEX idx_products_sub_category (sub_category)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- Table: orders
-- One row per unique order (the "header" -- who ordered, when, how it ships).
-- Line-item details live in order_items, not here.
-- ----------------------------------------------------------------------------
CREATE TABLE orders (
    order_id     VARCHAR(20)  NOT NULL,
    customer_id  VARCHAR(20)  NOT NULL,
    order_date   DATE         NOT NULL,
    ship_date    DATE         NOT NULL,
    ship_mode    VARCHAR(30)  NOT NULL,
    PRIMARY KEY (order_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    -- order_date is indexed because nearly every analytical query in this
    -- project (monthly sales, forecasting, seasonality) filters or groups
    -- by date.
    INDEX idx_orders_order_date (order_date),
    INDEX idx_orders_customer_id (customer_id),
    -- Business rule enforced at the database level, not just in Python:
    -- a ship date can never be earlier than its order date.
    CONSTRAINT chk_ship_after_order CHECK (ship_date >= order_date)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- Table: order_items
-- One row per product line within an order. This is where sales, quantity,
-- discount, and profit actually live -- these are facts about a specific
-- product within a specific order, not about the order or product alone.
-- ----------------------------------------------------------------------------
CREATE TABLE order_items (
    order_item_id  INT           NOT NULL AUTO_INCREMENT,
    order_id       VARCHAR(20)   NOT NULL,
    product_id     VARCHAR(20)   NOT NULL,
    sales          DECIMAL(12,2) NOT NULL,
    quantity       INT           NOT NULL,
    discount       DECIMAL(4,2)  NOT NULL DEFAULT 0.00,
    profit         DECIMAL(12,2) NOT NULL,
    PRIMARY KEY (order_item_id),
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    INDEX idx_order_items_order_id (order_id),
    INDEX idx_order_items_product_id (product_id),
    -- Business rules enforced at the database level:
    CONSTRAINT chk_sales_positive CHECK (sales > 0),
    CONSTRAINT chk_quantity_positive CHECK (quantity > 0),
    CONSTRAINT chk_discount_range CHECK (discount >= 0.00 AND discount <= 1.00)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- View: vw_order_summary
-- A denormalized VIEW that joins everything back together for convenient
-- analytical querying -- this gives us the best of both worlds: normalized
-- storage (no duplication, safe updates) AND a flat, easy-to-query surface
-- for BI tools like Power BI to connect to directly.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_order_summary AS
SELECT
    oi.order_item_id,
    o.order_id,
    o.order_date,
    o.ship_date,
    o.ship_mode,
    c.customer_id,
    c.customer_name,
    c.segment,
    c.country,
    c.city,
    c.state,
    c.region,
    p.product_id,
    p.product_name,
    p.category,
    p.sub_category,
    oi.sales,
    oi.quantity,
    oi.discount,
    oi.profit
FROM order_items oi
JOIN orders o     ON oi.order_id = o.order_id
JOIN customers c  ON o.customer_id = c.customer_id
JOIN products p   ON oi.product_id = p.product_id;
