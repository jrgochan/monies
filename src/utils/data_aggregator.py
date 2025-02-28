import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from src.models.database import DataSource, UserDataSource, User

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAggregator:
    """Class to aggregate financial data from multiple sources"""
    
    @staticmethod
    def get_user_data_sources(db: Session, user_id: int, category: str = None) -> List[Dict[str, Any]]:
        """
        Get user's preferred data sources in priority order
        
        Args:
            db: Database session
            user_id: User ID
            category: Optional filter for data source category
            
        Returns:
            List of data source dictionaries with settings
        """
        # Get user's custom preferences
        query = (
            db.query(DataSource, UserDataSource)
            .join(
                UserDataSource, 
                UserDataSource.data_source_id == DataSource.id, 
                isouter=True
            )
            .filter(
                (UserDataSource.user_id == user_id) | (UserDataSource.id == None),
                DataSource.enabled == True
            )
        )
        
        # Apply category filter if provided
        if category:
            query = query.filter(DataSource.category == category)
            
        results = query.all()
        
        # Process the results into a list of dictionaries
        data_sources = []
        for ds, uds in results:
            # Determine if source is enabled
            enabled = True
            if uds and uds.enabled is not None:
                enabled = uds.enabled
                
            # Determine priority
            priority = ds.priority
            if uds and uds.priority is not None and uds.priority > 0:
                priority = uds.priority
                
            data_sources.append({
                "id": ds.id,
                "name": ds.name,
                "display_name": ds.display_name,
                "category": ds.category,
                "api_required": ds.api_required,
                "api_key_field": ds.api_key_field,
                "enabled": enabled,
                "priority": priority
            })
        
        # Sort by priority (lower number = higher priority)
        data_sources.sort(key=lambda x: x["priority"])
        
        return data_sources
    
    @staticmethod
    def get_available_data_sources(db: Session, category: str = None) -> List[Dict[str, Any]]:
        """
        Get all available data sources
        
        Args:
            db: Database session
            category: Optional filter for data source category
            
        Returns:
            List of data source dictionaries
        """
        query = db.query(DataSource).filter(DataSource.enabled == True)
        
        if category:
            query = query.filter(DataSource.category == category)
            
        data_sources = query.order_by(DataSource.priority).all()
        
        return [
            {
                "id": ds.id,
                "name": ds.name,
                "display_name": ds.display_name,
                "category": ds.category,
                "api_required": ds.api_required,
                "api_key_field": ds.api_key_field,
                "priority": ds.priority
            }
            for ds in data_sources
        ]
    
    @staticmethod
    def update_user_data_source_preferences(
        db: Session, 
        user_id: int, 
        preferences: List[Dict[str, Any]]
    ) -> bool:
        """
        Update user's data source preferences
        
        Args:
            db: Database session
            user_id: User ID
            preferences: List of preference dictionaries with data_source_id, enabled, and priority
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing user preferences
            existing_prefs = db.query(UserDataSource).filter(
                UserDataSource.user_id == user_id
            ).all()
            
            # Create a map of existing preferences by data_source_id
            existing_prefs_map = {p.data_source_id: p for p in existing_prefs}
            
            # Update or create preferences
            for pref in preferences:
                data_source_id = pref.get("data_source_id")
                if not data_source_id:
                    continue
                
                if data_source_id in existing_prefs_map:
                    # Update existing preference
                    user_pref = existing_prefs_map[data_source_id]
                    if "enabled" in pref:
                        user_pref.enabled = pref["enabled"]
                    if "priority" in pref:
                        user_pref.priority = pref["priority"]
                else:
                    # Create new preference
                    user_pref = UserDataSource(
                        user_id=user_id,
                        data_source_id=data_source_id,
                        enabled=pref.get("enabled", True),
                        priority=pref.get("priority", 0)
                    )
                    db.add(user_pref)
            
            db.commit()
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating user data source preferences: {str(e)}")
            return False
    
    @staticmethod
    def aggregate_stock_data(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate stock data from multiple sources
        
        Args:
            results: List of dictionaries with stock data from different sources
            
        Returns:
            Dictionary with aggregated stock data
        """
        if not results:
            return {}
            
        if len(results) == 1:
            # If only one source, just return it
            return results[0]
            
        # Start with a copy of the first result as the base
        aggregated = results[0].copy()
        aggregated['data_sources'] = [r.get('data_source', 'unknown') for r in results]
        
        # Aggregate numerical data by averaging values from all sources
        numerical_fields = ['start_price', 'end_price', 'high', 'low', 'volume_avg']
        for field in numerical_fields:
            values = []
            for result in results:
                value = result.get('data', {}).get(field)
                if value is not None:
                    # Convert string numbers to float
                    try:
                        if isinstance(value, str):
                            value = float(value.replace('$', '').replace(',', ''))
                        values.append(value)
                    except ValueError:
                        pass
                        
            if values:
                avg_value = sum(values) / len(values)
                aggregated['data'][field] = f"{avg_value:.2f}"
        
        # Recalculate percent_change based on aggregated start and end prices
        try:
            start_price = float(aggregated['data']['start_price'])
            end_price = float(aggregated['data']['end_price'])
            percent_change = ((end_price - start_price) / start_price) * 100
            aggregated['data']['percent_change'] = f"{percent_change:.2f}"
        except (ValueError, KeyError):
            pass
            
        # Use most common non-empty values for categorical fields
        categorical_fields = ['company_name', 'sector', 'industry']
        for field in categorical_fields:
            values = [r.get('data', {}).get(field) for r in results]
            values = [v for v in values if v and v != 'Unknown']
            if values:
                # Use the most common value
                from collections import Counter
                most_common = Counter(values).most_common(1)[0][0]
                aggregated['data'][field] = most_common
        
        # Take the most recent analysis
        analyses = [r.get('analysis', '') for r in results if r.get('analysis')]
        if analyses:
            # Take the longest analysis as it's likely the most detailed
            aggregated['analysis'] = max(analyses, key=len)
        
        # Mark as aggregated
        aggregated['aggregated'] = True
        
        return aggregated
    
    @staticmethod
    def aggregate_crypto_data(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate cryptocurrency data from multiple sources
        
        Args:
            results: List of dictionaries with crypto data from different sources
            
        Returns:
            Dictionary with aggregated crypto data
        """
        # Similar approach to stock data aggregation
        return DataAggregator.aggregate_stock_data(results)
        
    @staticmethod
    def aggregate_etf_data(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate ETF data from multiple sources
        
        Args:
            results: List of dictionaries with ETF data from different sources
            
        Returns:
            Dictionary with aggregated ETF data
        """
        # Similar approach to stock data aggregation with ETF-specific fields
        if not results:
            return {}
            
        aggregated = DataAggregator.aggregate_stock_data(results)
        
        # ETF-specific fields like expense ratio
        etf_fields = ['expense_ratio', 'assets_under_management', 'yield']
        for field in etf_fields:
            values = []
            for result in results:
                value = result.get('data', {}).get(field)
                if value is not None:
                    try:
                        if isinstance(value, str):
                            value = float(value.replace('%', '').replace('$', '').replace(',', ''))
                        values.append(value)
                    except ValueError:
                        pass
                        
            if values:
                avg_value = sum(values) / len(values)
                # Format appropriately
                if field == 'expense_ratio' or field == 'yield':
                    aggregated['data'][field] = f"{avg_value:.2f}%"
                elif field == 'assets_under_management':
                    aggregated['data'][field] = f"${avg_value:.2f}M"
                else:
                    aggregated['data'][field] = f"{avg_value:.2f}"
        
        return aggregated