# examples/test_pricing.py

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_pricing_system import MarketPricingSystem

async def test_basic_pricing():
    # Initialize system
    system = MarketPricingSystem()
    
    # Test product
    product_id = "TEST001"
    current_price = 99.99
    
    # Add competitor URL for tracking
    system.competitor_tracker.add_competitor_product(
        product_id,
        "https://competitor.com/product/TEST001"
    )
    
    # Test price optimization
    print("\nTesting price optimization...")
    result = await system.optimize_product_price(product_id, current_price)
    print("Optimization result:", result)
    
    # Generate report
    print("\nGenerating comprehensive report...")
    report = await system.generate_comprehensive_report(product_id)
    print("Report summary:", report.get('summary', {}))

if __name__ == "__main__":
    asyncio.run(test_basic_pricing())