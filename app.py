# market_pricing_system.py

import asyncio
import json
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
from dataclasses import dataclass
import os
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration settings for the Market Pricing System"""
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "your-api-key-here")
    DB_PATH: str = "pricing_history.db"
    COMPETITOR_SCAN_INTERVAL: int = 3600  # seconds
    MAX_PRICE_VARIANCE: float = 0.20  # 20% max price adjustment
    MIN_CONFIDENCE_THRESHOLD: float = 0.75
    PRICE_UPDATE_COOLDOWN: int = 300  # seconds
    REPORT_CACHE_DURATION: int = 86400  # 24 hours
    MAX_HISTORY_DAYS: int = 90

class Database:
    """Database management for pricing history"""
    def __init__(self, config: Config):
        self.db_path = config.DB_PATH
        self._init_db()

    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    product_id TEXT,
                    price REAL,
                    timestamp DATETIME,
                    source TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS purchase_history (
                    customer_id TEXT,
                    product_id TEXT,
                    price REAL,
                    quantity INTEGER,
                    timestamp DATETIME
                )
            """)

    def save_price(self, product_id: str, price: float, source: str):
        """Save price data to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO price_history (product_id, price, timestamp, source) VALUES (?, ?, ?, ?)",
                (product_id, price, datetime.utcnow(), source)
            )

    def get_price_history(self, product_id: str, days: int = 30) -> List[Dict]:
        """Retrieve price history for a product"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT price, timestamp, source 
                FROM price_history 
                WHERE product_id = ? 
                AND timestamp >= datetime('now', ?)
                ORDER BY timestamp DESC
                """,
                (product_id, f'-{days} days')
            )
            return [{'price': row[0], 'timestamp': row[1], 'source': row[2]} 
                   for row in cursor.fetchall()]

class GroqAI:
    """Handle interactions with Groq AI API"""
    def __init__(self, config: Config):
        self.api_key = config.GROQ_API_KEY
        self.base_url = "https://api.groq.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.config = config

    async def analyze_market_trends(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market trends using Groq AI"""
        prompt = self._create_market_analysis_prompt(market_data)
        return await self._make_groq_request("market_analysis", prompt)

    async def optimize_pricing(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-optimized pricing recommendations"""
        prompt = self._create_pricing_prompt(product_data)
        return await self._make_groq_request("pricing_optimization", prompt)

    async def predict_purchase_probability(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict likelihood of purchase at different price points"""
        prompt = self._create_purchase_prediction_prompt(customer_data)
        return await self._make_groq_request("purchase_prediction", prompt)

    def _create_market_analysis_prompt(self, market_data: Dict[str, Any]) -> str:
        return json.dumps({
            "task": "market_analysis",
            "data": market_data,
            "parameters": {
                "confidence_threshold": self.config.MIN_CONFIDENCE_THRESHOLD,
                "max_variance": self.config.MAX_PRICE_VARIANCE
            }
        })

    def _create_pricing_prompt(self, product_data: Dict[str, Any]) -> str:
        return json.dumps({
            "task": "price_optimization",
            "data": product_data,
            "parameters": {
                "confidence_threshold": self.config.MIN_CONFIDENCE_THRESHOLD,
                "max_adjustment": self.config.MAX_PRICE_VARIANCE
            }
        })

    def _create_purchase_prediction_prompt(self, customer_data: Dict[str, Any]) -> str:
        return json.dumps({
            "task": "purchase_prediction",
            "data": customer_data,
            "parameters": {
                "confidence_threshold": self.config.MIN_CONFIDENCE_THRESHOLD
            }
        })

    async def _make_groq_request(self, endpoint: str, prompt: str) -> Dict[str, Any]:
        """Make API request to Groq"""
        try:
            response = requests.post(
                f"{self.base_url}/{endpoint}",
                headers=self.headers,
                json={"prompt": prompt}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {str(e)}")
            raise

class CompetitorTracker:
    """Track and analyze competitor prices"""
    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.competitor_urls = {}

    def add_competitor_product(self, product_id: str, url: str):
        """Add a competitor's product URL to track"""
        self.competitor_urls[product_id] = url

    async def track_competitor_prices(self) -> Dict[str, List[Dict]]:
        """Fetch and store competitor prices"""
        results = {}
        for product_id, url in self.competitor_urls.items():
            try:
                price = await self._scrape_price(url)
                if price:
                    self.db.save_price(product_id, price, 'competitor')
                    results[product_id] = price
            except Exception as e:
                logger.error(f"Error tracking competitor price for {product_id}: {str(e)}")
        return results

    async def _scrape_price(self, url: str) -> Optional[float]:
        """Scrape price from competitor website"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Add custom price extraction logic here
            # This is a placeholder that should be customized based on the target website
            price_element = soup.find('span', {'class': 'price'})
            if price_element:
                return float(price_element.text.strip().replace('$', ''))
        except Exception as e:
            logger.error(f"Error scraping price from {url}: {str(e)}")
        return None

class PurchasePredictor:
    """Predict customer purchase behavior"""
    def __init__(self, groq_ai: GroqAI, db: Database):
        self.groq_ai = groq_ai
        self.db = db

    async def predict_purchase_likelihood(self, customer_id: str, product_id: str, 
                                       price: float) -> Dict[str, Any]:
        """Predict likelihood of purchase at given price"""
        # Gather customer history
        purchase_history = self._get_customer_history(customer_id)
        
        # Prepare data for AI analysis
        customer_data = {
            'customer_id': customer_id,
            'product_id': product_id,
            'proposed_price': price,
            'purchase_history': purchase_history,
            'market_context': await self._get_market_context(product_id)
        }

        # Get AI prediction
        prediction = await self.groq_ai.predict_purchase_probability(customer_data)
        
        return {
            'customer_id': customer_id,
            'product_id': product_id,
            'price': price,
            'purchase_probability': prediction.get('probability', 0),
            'recommended_price': prediction.get('recommended_price'),
            'factors': prediction.get('contributing_factors', [])
        }

    def _get_customer_history(self, customer_id: str) -> List[Dict]:
        """Retrieve customer's purchase history"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT product_id, price, quantity, timestamp 
                FROM purchase_history 
                WHERE customer_id = ? 
                ORDER BY timestamp DESC
                """,
                (customer_id,)
            )
            return [{'product_id': row[0], 'price': row[1], 
                    'quantity': row[2], 'timestamp': row[3]} 
                   for row in cursor.fetchall()]

    async def _get_market_context(self, product_id: str) -> Dict[str, Any]:
        """Get current market context for the product"""
        price_history = self.db.get_price_history(product_id)
        return {
            'price_history': price_history,
            'market_trends': self._analyze_price_trends(price_history)
        }

    def _analyze_price_trends(self, price_history: List[Dict]) -> Dict[str, Any]:
        """Analyze price trends from historical data"""
        if not price_history:
            return {}

        prices = [p['price'] for p in price_history]
        return {
            'avg_price': sum(prices) / len(prices),
            'min_price': min(prices),
            'max_price': max(prices),
            'price_volatility': pd.Series(prices).std() if len(prices) > 1 else 0
        }

class ReportGenerator:
    """Generate pricing and market analysis reports"""
    def __init__(self, db: Database):
        self.db = db

    def generate_pricing_report(self, product_id: str, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive pricing report"""
        price_history = self.db.get_price_history(product_id, days)
        
        if not price_history:
            return {"error": "No data available for the specified period"}

        # Convert to pandas DataFrame for analysis
        df = pd.DataFrame(price_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Generate visualizations
        self._create_price_trend_plot(df, product_id)

        # Calculate statistics
        stats = self._calculate_statistics(df)

        return {
            'product_id': product_id,
            'period': f'Last {days} days',
            'statistics': stats,
            'charts': {
                'price_trend': f'price_trend_{product_id}.png'
            },
            'recommendations': self._generate_recommendations(stats)
        }

    def _create_price_trend_plot(self, df: pd.DataFrame, product_id: str):
        """Create price trend visualization"""
        plt.figure(figsize=(10, 6))
        df['price'].plot(kind='line')
        plt.title(f'Price Trends for Product {product_id}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.savefig(f'price_trend_{product_id}.png')
        plt.close()

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key statistics from price data"""
        return {
            'current_price': df['price'].iloc[-1],
            'average_price': df['price'].mean(),
            'min_price': df['price'].min(),
            'max_price': df['price'].max(),
            'price_volatility': df['price'].std(),
            'price_change_30d': (
                (df['price'].iloc[-1] - df['price'].iloc[0]) / 
                df['price'].iloc[0] * 100
            ),
            'competitor_stats': {
                'avg_competitor_price': df[df['source'] == 'competitor']['price'].mean(),
                'price_difference': (
                    df[df['source'] != 'competitor']['price'].mean() -
                    df[df['source'] == 'competitor']['price'].mean()
                )
            }
        }

    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate pricing recommendations based on statistics"""
        recommendations = []
        
        # Price position relative to average
        if stats['current_price'] > stats['average_price'] * 1.1:
            recommendations.append(
                "Current price is significantly above average. Consider price reduction "
                "if sales velocity has decreased."
            )
        elif stats['current_price'] < stats['average_price'] * 0.9:
            recommendations.append(
                "Current price is significantly below average. Consider price increase "
                "if demand is strong."
            )

        # Competitor pricing
        if stats['competitor_stats']['price_difference'] > 0:
            recommendations.append(
                "Our prices are higher than competitors. Monitor market share and "
                "consider price adjustments if losing sales."
            )

        # Volatility
        if stats['price_volatility'] > stats['average_price'] * 0.1:
            recommendations.append(
                "High price volatility detected. Consider implementing more stable "
                "pricing strategy."
            )

        return recommendations

class MarketPricingSystem:
    """Main system that coordinates all pricing components"""
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.db = Database(self.config)
        self.groq_ai = GroqAI(self.config)
        self.competitor_tracker = CompetitorTracker(self.config, self.db)
        self.purchase_predictor = PurchasePredictor(self.groq_ai, self.db)
        self.report_generator = ReportGenerator(self.db)

    async def optimize_product_price(self, product_id: str, current_price: float) -> Dict[str, Any]:
        """Optimize price for a product considering all factors"""
        try:
            # Get competitor prices
            competitor_prices = await self.competitor_tracker.track_competitor_prices()
            
            # Get market trends
            market_data = {
                'competitor_prices': competitor_prices,
                'price_history': self.db.get_price_history(product_id),
                'current_price': current_price
            }
            
            market_analysis = await self.groq_ai.analyze_market_trends(market_data)
            
            # Get price optimization
            optimization_result = await self.groq_ai.optimize_pricing({
                'product_id': product_id,
                'current_price': current_price,
                'market_analysis': market_analysis,
                'competitor_prices': competitor_prices
            })
            
            # Store new price in database
            if optimization_result.get('recommended_price'):
                self.db.save_price(
                    product_id, 
                    optimization_result['recommended_price'], 
                    'system'
                )
            
            return {
                'status': 'success',
                'product_id': product_id,
                'current_price': current_price,
                'recommended_price': optimization_result.get('recommended_price'),
                'market_analysis': market_analysis,
                'competitor_data': competitor_prices,
                'confidence_score': optimization_result.get('confidence_score', 0),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing price for product {product_id}: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'product_id': product_id
            }

    async def analyze_customer_segments(self, product_id: str) -> Dict[str, Any]:
        """Analyze customer segments and their price sensitivity"""
        try:
            # Get purchase history for the product
            with sqlite3.connect(self.db.db_path) as conn:
                df = pd.read_sql_query(
                    """
                    SELECT customer_id, price, quantity, timestamp
                    FROM purchase_history
                    WHERE product_id = ?
                    AND timestamp >= datetime('now', '-90 days')
                    """,
                    conn,
                    params=(product_id,)
                )

            if df.empty:
                return {"status": "error", "message": "No purchase data available"}

            # Analyze price sensitivity by segment
            segments = self._analyze_price_sensitivity(df)

            return {
                "status": "success",
                "product_id": product_id,
                "segments": segments,
                "recommendations": self._generate_segment_recommendations(segments)
            }

        except Exception as e:
            logger.error(f"Error analyzing customer segments: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _analyze_price_sensitivity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price sensitivity for different customer segments"""
        df['price_bracket'] = pd.qcut(df['price'], q=4, labels=['low', 'medium-low', 'medium-high', 'high'])
        
        sensitivity_analysis = {}
        for bracket in df['price_bracket'].unique():
            bracket_data = df[df['price_bracket'] == bracket]
            sensitivity_analysis[str(bracket)] = {
                'avg_quantity': bracket_data['quantity'].mean(),
                'total_customers': bracket_data['customer_id'].nunique(),
                'total_revenue': (bracket_data['price'] * bracket_data['quantity']).sum(),
                'avg_price': bracket_data['price'].mean()
            }
            
        return sensitivity_analysis

    def _generate_segment_recommendations(self, segments: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on segment analysis"""
        recommendations = []
        
        # Analyze price sensitivity patterns
        high_end = segments.get('high', {})
        low_end = segments.get('low', {})
        
        if high_end and low_end:
            revenue_ratio = high_end.get('total_revenue', 0) / low_end.get('total_revenue', 1)
            
            if revenue_ratio > 2:
                recommendations.append(
                    "High-end segment shows strong revenue performance. Consider premium pricing "
                    "strategy and exclusive offerings."
                )
            elif revenue_ratio < 0.5:
                recommendations.append(
                    "Lower-price segments dominate revenue. Focus on volume-based pricing and "
                    "competitive positioning."
                )

        return recommendations

    async def generate_comprehensive_report(self, product_id: str) -> Dict[str, Any]:
        """Generate a comprehensive pricing and market analysis report"""
        try:
            # Get basic pricing report
            pricing_report = self.report_generator.generate_pricing_report(product_id)
            
            # Get customer segment analysis
            segment_analysis = await self.analyze_customer_segments(product_id)
            
            # Get latest market optimization
            current_price = pricing_report['statistics']['current_price']
            optimization_data = await self.optimize_product_price(product_id, current_price)
            
            return {
                "status": "success",
                "product_id": product_id,
                "timestamp": datetime.utcnow().isoformat(),
                "pricing_analysis": pricing_report,
                "segment_analysis": segment_analysis,
                "market_optimization": optimization_data,
                "summary": self._generate_executive_summary(
                    pricing_report,
                    segment_analysis,
                    optimization_data
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _generate_executive_summary(self, pricing_report: Dict, 
                                  segment_analysis: Dict,
                                  optimization_data: Dict) -> Dict[str, Any]:
        """Generate an executive summary of all analyses"""
        return {
            "current_market_position": {
                "price_position": self._analyze_price_position(pricing_report),
                "market_trend": self._analyze_market_trend(pricing_report),
                "competitive_status": self._analyze_competitive_status(optimization_data)
            },
            "key_recommendations": self._compile_key_recommendations(
                pricing_report,
                segment_analysis,
                optimization_data
            ),
            "action_items": self._generate_action_items(
                pricing_report,
                optimization_data
            )
        }

    def _analyze_price_position(self, pricing_report: Dict) -> str:
        """Analyze current price position in market"""
        stats = pricing_report.get('statistics', {})
        current = stats.get('current_price', 0)
        avg = stats.get('average_price', 0)
        
        if current > avg * 1.1:
            return "Premium position"
        elif current < avg * 0.9:
            return "Value position"
        return "Mid-market position"

    def _analyze_market_trend(self, pricing_report: Dict) -> str:
        """Analyze market trend direction"""
        stats = pricing_report.get('statistics', {})
        change = stats.get('price_change_30d', 0)
        
        if change > 5:
            return "Upward trend"
        elif change < -5:
            return "Downward trend"
        return "Stable market"

    def _analyze_competitive_status(self, optimization_data: Dict) -> str:
        """Analyze competitive position"""
        competitor_data = optimization_data.get('competitor_data', {})
        if not competitor_data:
            return "Insufficient competitive data"
            
        our_price = optimization_data.get('current_price', 0)
        comp_prices = [p for p in competitor_data.values() if p]
        
        if not comp_prices:
            return "No competitor price data"
            
        avg_comp_price = sum(comp_prices) / len(comp_prices)
        
        if our_price > avg_comp_price * 1.1:
            return "Premium to competition"
        elif our_price < avg_comp_price * 0.9:
            return "Discount to competition"
        return "At market"

    def _compile_key_recommendations(self, pricing_report: Dict,
                                   segment_analysis: Dict,
                                   optimization_data: Dict) -> List[str]:
        """Compile key recommendations from all analyses"""
        recommendations = []
        
        # Add pricing recommendations
        if pricing_report.get('recommendations'):
            recommendations.extend(pricing_report['recommendations'])
            
        # Add segment-based recommendations
        if segment_analysis.get('recommendations'):
            recommendations.extend(segment_analysis['recommendations'])
            
        # Add optimization-based recommendations
        if optimization_data.get('recommended_price'):
            current_price = optimization_data.get('current_price', 0)
            recommended_price = optimization_data.get('recommended_price', 0)
            
            if abs(recommended_price - current_price) / current_price > 0.05:
                recommendations.append(
                    f"Consider price {'increase' if recommended_price > current_price else 'decrease'} "
                    f"to {recommended_price:.2f} based on market optimization analysis."
                )
                
        return recommendations

    def _generate_action_items(self, pricing_report: Dict,
                             optimization_data: Dict) -> List[Dict[str, str]]:
        """Generate specific action items based on analysis"""
        action_items = []
        
        # Price adjustment actions
        if optimization_data.get('recommended_price'):
            action_items.append({
                "action": "Price Adjustment",
                "description": "Update product price based on optimization analysis",
                "priority": "High" if optimization_data.get('confidence_score', 0) > 0.8 else "Medium"
            })
            
        # Competitive monitoring actions
        if not optimization_data.get('competitor_data'):
            action_items.append({
                "action": "Competitor Monitoring",
                "description": "Set up automated competitor price tracking",
                "priority": "High"
            })
            
        # Market analysis actions
        if pricing_report.get('statistics', {}).get('price_volatility', 0) > 0.1:
            action_items.append({
                "action": "Market Analysis",
                "description": "Conduct detailed market analysis to understand price volatility",
                "priority": "Medium"
            })
            
        return action_items

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize the system
        system = MarketPricingSystem()
        
        # Example product ID and current price
        product_id = "SAMPLE001"
        current_price = 99.99
        
        # Get price optimization
        result = await system.optimize_product_price(product_id, current_price)
        print("Price Optimization Result:", json.dumps(result, indent=2))
        
        # Generate comprehensive report
        report = await system.generate_comprehensive_report(product_id)
        print("Comprehensive Report:", json.dumps(report, indent=2))

    # Run the example
    asyncio.run(main())