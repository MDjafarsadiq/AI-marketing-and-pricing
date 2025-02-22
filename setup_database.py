# The system will automatically create the database
# No additional setup needed, but you can pre-populate with test data:

from market_pricing_system import MarketPricingSystem, Database, Config
import sqlite3

def setup_test_data():
    config = Config()
    db = Database(config)
    
    # Add some test purchase history
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute("""
        INSERT INTO purchase_history (customer_id, product_id, price, quantity, timestamp)
        VALUES 
            ('CUST001', 'TEST001', 99.99, 1, datetime('now', '-30 days')),
            ('CUST002', 'TEST001', 95.99, 2, datetime('now', '-20 days')),
            ('CUST003', 'TEST001', 97.99, 1, datetime('now', '-10 days'))
        """)

setup_test_data()