import asyncio
from market_pricing_system import MarketPricingSystem

async def main():
    # Initialize system
    system = MarketPricingSystem()
    
    # Example: Optimize price for a product
    product_id = "TEST001"
    current_price = 99.99
    
    # Get optimization
    optimization = await system.optimize_product_price(product_id, current_price)
    print("Price Optimization:", optimization)
    
    # Generate report
    report = await system.generate_comprehensive_report(product_id)
    print("\nComprehensive Report:", report)
    
    # Analyze customer segments
    segments = await system.analyze_customer_segments(product_id)
    print("\nCustomer Segments:", segments)

if __name__ == "__main__":
    asyncio.run(main())