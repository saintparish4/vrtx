from typing import List, Dict, Optional
from datetime import datetime
import logging
import numpy as np
from sklearn.ensemble import IsolationForest

from ..schemas import (
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    Anomaly,
    MetricPoint
)
from ..models.ensemble import EnsemblePredictor
from ..db import db

logger = logging.getLogger(__name__)


class FailureDetectorService:
    """Service for detecting anomalies and potential failures"""
    
    def __init__(self):
        self.anomaly_models: Dict[str, any] = {}
        self.isolation_forests: Dict[str, IsolationForest] = {}
        
        # Configuration
        self.contamination_rate = 0.1  # Expected proportion of outliers
        self.history_window_hours = 168  # 7 days
        self.severe_threshold_multiplier = 3.0
    
    def _get_or_create_ensemble(self, resource_id: str) -> EnsemblePredictor:
        """Get or create ensemble model for anomaly detection"""
        if resource_id not in self.anomaly_models:
            self.anomaly_models[resource_id] = EnsemblePredictor()
        return self.anomaly_models[resource_id]
    
    def _get_or_create_isolation_forest(self, resource_id: str) -> IsolationForest:
        """Get or create Isolation Forest for unsupervised anomaly detection"""
        if resource_id not in self.isolation_forests:
            self.isolation_forests[resource_id] = IsolationForest(
                contamination=self.contamination_rate,
                random_state=42,
                n_estimators=100
            )
        return self.isolation_forests[resource_id]
    
    async def detect_anomalies(
        self,
        request: AnomalyDetectionRequest
    ) -> AnomalyDetectionResponse:
        """Detect anomalies in resource metrics"""
        try:
            logger.info(f"Detecting anomalies for resource {request.resource_id}")
            
            # Extract timestamps and values
            timestamps = [point.timestamp for point in request.current_metrics]
            values = [point.value for point in request.current_metrics]
            
            if len(values) < 10:
                logger.warning(f"Insufficient data for anomaly detection: {len(values)} points")
                return AnomalyDetectionResponse(
                    resource_id=request.resource_id,
                    anomalies=[],
                    is_anomalous=False,
                    risk_score=0.0
                )
            
            # Method 1: Statistical anomaly detection using ensemble model
            ensemble_anomalies = await self._detect_with_ensemble(
                request.resource_id,
                timestamps,
                values,
                request.threshold
            )
            
            # Method 2: Unsupervised anomaly detection using Isolation Forest
            isolation_anomalies = self._detect_with_isolation_forest(
                request.resource_id,
                timestamps,
                values
            )
            
            # Combine anomalies from both methods
            combined_anomalies = self._combine_anomalies(
                ensemble_anomalies,
                isolation_anomalies
            )
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(
                combined_anomalies,
                values
            )
            
            # Save anomalies to database
            if combined_anomalies:
                severity = self._determine_overall_severity(combined_anomalies)
                await db.save_anomaly(
                    request.resource_id,
                    {'anomalies': [a.dict() for a in combined_anomalies]},
                    severity,
                    risk_score
                )
            
            logger.info(
                f"Detected {len(combined_anomalies)} anomalies "
                f"with risk score {risk_score:.3f}"
            )
            
            return AnomalyDetectionResponse(
                resource_id=request.resource_id,
                anomalies=combined_anomalies,
                is_anomalous=len(combined_anomalies) > 0,
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise
    
    async def _detect_with_ensemble(
        self,
        resource_id: str,
        timestamps: List[datetime],
        values: List[float],
        threshold: float
    ) -> List[Anomaly]:
        """Detect anomalies using ensemble model"""
        try:
            model = self._get_or_create_ensemble(resource_id)
            
            # Train model if not trained
            if not model.is_trained:
                logger.info("Training ensemble model for anomaly detection")
                model.train(timestamps, values)
            
            # Detect anomalies
            anomaly_data = model.detect_anomalies(timestamps, values, threshold)
            
            # Convert to Anomaly objects
            anomalies = []
            for data in anomaly_data:
                anomalies.append(Anomaly(
                    timestamp=data['timestamp'],
                    value=data['value'],
                    expected_value=data['expected_value'],
                    severity=data['severity'],
                    confidence=data['confidence']
                ))
            
            return anomalies
            
        except Exception as e:
            logger.warning(f"Ensemble anomaly detection failed: {e}")
            return []
    
    def _detect_with_isolation_forest(
        self,
        resource_id: str,
        timestamps: List[datetime],
        values: List[float]
    ) -> List[Anomaly]:
        """Detect anomalies using Isolation Forest"""
        try:
            values_array = np.array(values).reshape(-1, 1)
            
            # Get or train Isolation Forest
            iso_forest = self._get_or_create_isolation_forest(resource_id)
            iso_forest.fit(values_array)
            
            # Predict anomalies (-1 for outliers, 1 for inliers)
            predictions = iso_forest.predict(values_array)
            anomaly_scores = iso_forest.score_samples(values_array)
            
            # Convert to Anomaly objects
            anomalies = []
            mean_value = np.mean(values)
            
            for i, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
                if pred == -1:  # Outlier detected
                    # Calculate severity based on anomaly score
                    # Lower scores indicate more anomalous points
                    normalized_score = abs(score)
                    severity = 'high' if normalized_score > 0.5 else 'medium'
                    confidence = min(normalized_score, 1.0)
                    
                    anomalies.append(Anomaly(
                        timestamp=timestamps[i],
                        value=float(values[i]),
                        expected_value=float(mean_value),
                        severity=severity,
                        confidence=confidence
                    ))
            
            return anomalies
            
        except Exception as e:
            logger.warning(f"Isolation Forest detection failed: {e}")
            return []
    
    def _combine_anomalies(
        self,
        ensemble_anomalies: List[Anomaly],
        isolation_anomalies: List[Anomaly]
    ) -> List[Anomaly]:
        """Combine anomalies from different methods"""
        # Create dictionary keyed by timestamp
        anomaly_dict = {}
        
        # Add ensemble anomalies
        for anomaly in ensemble_anomalies:
            ts = anomaly.timestamp
            anomaly_dict[ts] = anomaly
        
        # Merge isolation forest anomalies
        for anomaly in isolation_anomalies:
            ts = anomaly.timestamp
            if ts in anomaly_dict:
                # Both methods detected anomaly - increase confidence
                existing = anomaly_dict[ts]
                existing.confidence = min(
                    (existing.confidence + anomaly.confidence) / 2 * 1.3,
                    1.0
                )
                # Upgrade severity if necessary
                if existing.severity == 'medium' and anomaly.severity == 'high':
                    existing.severity = 'high'
            else:
                # Only isolation forest detected this
                anomaly_dict[ts] = anomaly
        
        # Sort by timestamp and return
        return sorted(anomaly_dict.values(), key=lambda x: x.timestamp)
    
    def _calculate_risk_score(
        self,
        anomalies: List[Anomaly],
        values: List[float]
    ) -> float:
        """Calculate overall risk score based on anomalies"""
        if not anomalies:
            return 0.0
        
        # Factors contributing to risk score
        anomaly_ratio = len(anomalies) / len(values)
        
        # Average confidence of detected anomalies
        avg_confidence = np.mean([a.confidence for a in anomalies])
        
        # Severity weights
        severity_weights = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        weighted_severity = np.mean([
            severity_weights.get(a.severity, 0.5)
            for a in anomalies
        ])
        
        # Recent anomaly factor (more weight to recent anomalies)
        recent_anomalies = sum(
            1 for a in anomalies[-10:]
            if a.severity in ['medium', 'high']
        )
        recency_factor = min(recent_anomalies / 10, 1.0)
        
        # Combine factors
        risk_score = (
            anomaly_ratio * 0.3 +
            avg_confidence * 0.3 +
            weighted_severity * 0.3 +
            recency_factor * 0.1
        )
        
        return float(min(risk_score, 1.0))
    
    def _determine_overall_severity(self, anomalies: List[Anomaly]) -> str:
        """Determine overall severity from list of anomalies"""
        if not anomalies:
            return 'low'
        
        high_count = sum(1 for a in anomalies if a.severity == 'high')
        medium_count = sum(1 for a in anomalies if a.severity == 'medium')
        
        # If more than 30% are high severity, overall is high
        if high_count / len(anomalies) > 0.3:
            return 'high'
        
        # If any high severity or more than 50% medium, overall is medium
        if high_count > 0 or medium_count / len(anomalies) > 0.5:
            return 'medium'
        
        return 'low'
    
    async def predict_failure_probability(
        self,
        resource_id: str,
        current_metrics: List[MetricPoint],
        lookback_hours: int = 24
    ) -> Dict:
        """Predict probability of resource failure"""
        try:
            timestamps = [point.timestamp for point in current_metrics]
            values = [point.value for point in current_metrics]
            
            # Detect current anomalies
            anomaly_request = AnomalyDetectionRequest(
                resource_id=resource_id,
                resource_type=current_metrics[0].resource_type,
                current_metrics=current_metrics,
                threshold=2.5
            )
            
            anomaly_response = await self.detect_anomalies(anomaly_request)
            
            # Calculate trend
            trend = self._calculate_trend(values)
            
            # Calculate volatility
            volatility = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
            
            # Failure probability factors
            anomaly_factor = anomaly_response.risk_score * 0.4
            trend_factor = abs(trend) * 0.3
            volatility_factor = min(volatility, 1.0) * 0.3
            
            failure_probability = min(
                anomaly_factor + trend_factor + volatility_factor,
                1.0
            )
            
            # Determine failure risk level
            if failure_probability > 0.7:
                risk_level = 'critical'
            elif failure_probability > 0.5:
                risk_level = 'high'
            elif failure_probability > 0.3:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                anomaly_response,
                trend,
                volatility,
                risk_level
            )
            
            return {
                'resource_id': resource_id,
                'failure_probability': float(failure_probability),
                'risk_level': risk_level,
                'contributing_factors': {
                    'anomaly_score': float(anomaly_factor),
                    'trend_score': float(trend_factor),
                    'volatility_score': float(volatility_factor)
                },
                'anomaly_count': len(anomaly_response.anomalies),
                'trend': 'increasing' if trend > 0.1 else 'decreasing' if trend < -0.1 else 'stable',
                'recommendations': recommendations,
                'analyzed_at': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failure probability prediction failed: {e}")
            raise
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]
        
        # Normalize by mean to get percentage change
        mean_value = np.mean(values)
        if mean_value > 0:
            normalized_slope = slope / mean_value
        else:
            normalized_slope = 0.0
        
        return float(normalized_slope)
    
    def _generate_recommendations(
        self,
        anomaly_response: AnomalyDetectionResponse,
        trend: float,
        volatility: float,
        risk_level: str
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if risk_level in ['critical', 'high']:
            recommendations.append(
                "Immediate action required: Consider scaling resources or investigating root cause"
            )
        
        if anomaly_response.is_anomalous:
            high_severity_count = sum(
                1 for a in anomaly_response.anomalies
                if a.severity == 'high'
            )
            if high_severity_count > 0:
                recommendations.append(
                    f"Detected {high_severity_count} high-severity anomalies - investigate system behavior"
                )
        
        if trend > 0.2:
            recommendations.append(
                "Resource usage is rapidly increasing - consider proactive scaling"
            )
        elif trend < -0.2:
            recommendations.append(
                "Resource usage is rapidly decreasing - consider scaling down to optimize costs"
            )
        
        if volatility > 0.5:
            recommendations.append(
                "High volatility detected - consider implementing auto-scaling policies"
            )
        
        if not recommendations:
            recommendations.append(
                "Resource metrics are within normal parameters - continue monitoring"
            )
        
        return recommendations
    
    async def batch_detect_anomalies(
        self,
        requests: List[AnomalyDetectionRequest]
    ) -> List[AnomalyDetectionResponse]:
        """Detect anomalies for multiple resources"""
        responses = []
        
        for request in requests:
            try:
                response = await self.detect_anomalies(request)
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch anomaly detection failed for {request.resource_id}: {e}")
                # Continue with other detections
        
        return responses
    
    def clear_model_cache(self, resource_id: Optional[str] = None):
        """Clear anomaly detection model cache"""
        if resource_id:
            if resource_id in self.anomaly_models:
                del self.anomaly_models[resource_id]
            if resource_id in self.isolation_forests:
                del self.isolation_forests[resource_id]
            logger.info(f"Cleared anomaly detection cache for resource {resource_id}")
        else:
            self.anomaly_models = {}
            self.isolation_forests = {}
            logger.info("Cleared all anomaly detection caches")
    
    def get_statistics(self) -> Dict:
        """Get service statistics"""
        return {
            'total_ensemble_models': len(self.anomaly_models),
            'total_isolation_forests': len(self.isolation_forests),
            'trained_models': sum(
                1 for model in self.anomaly_models.values()
                if model.is_trained
            )
        }


# Global failure detector service instance
failure_detector_service = FailureDetectorService()