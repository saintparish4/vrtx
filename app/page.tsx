'use client';

import { useState, useEffect } from 'react';
import { Suspense } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  Server, 
  AlertTriangle, 
  DollarSign,
  Zap,
  Shield,
  Clock
} from 'lucide-react';

interface DashboardData {
  traffic_predictions: Array<{
    timestamp: number;
    predicted_load: number;
    confidence: number;
    spike_probability: number;
  }>;
  failure_predictions: Array<{
    resource_id: string;
    failure_probability: number;
    failure_type: string;
    recommended_action: string;
  }>;
  cost_optimization: {
    current_cost_per_hour: number;
    optimized_cost_per_hour: number;
    savings_percentage: number;
    recommendations: string[];
  };
  resources: Array<{
    id: string;
    provider: string;
    current_instances: number;
    min_instances: number;
    max_instances: number;
    cost_per_hour: number;
  }>;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedResource, setSelectedResource] = useState('web-server-cluster-1');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    fetchDashboardData();
    
    if (autoRefresh) {
      const interval = setInterval(fetchDashboardData, 30000); // 30 seconds
      return () => clearInterval(interval);
    }
  }, [selectedResource, autoRefresh]);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      
      // Fetch predictions
      const predictionsResponse = await fetch('/api/predictions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          resource_id: selectedResource,
          hours_ahead: 24,
          include_failures: true,
          include_cost_optimization: true
        })
      });

      // Fetch resources
      const resourcesResponse = await fetch('/api/resources');

      if (predictionsResponse.ok && resourcesResponse.ok) {
        const predictionsData = await predictionsResponse.json();
        const resourcesData = await resourcesResponse.json();

        setData({
          ...predictionsData.data,
          resources: resourcesData.data
        });
      }
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  const trafficData = data?.traffic_predictions?.map(p => ({
    time: formatTimestamp(p.timestamp),
    load: Math.round(p.predicted_load),
    confidence: Math.round(p.confidence * 100),
    spike_risk: Math.round(p.spike_probability * 100)
  })) || [];

  const currentResource = data?.resources?.find(r => r.id === selectedResource);
  const totalCost = data?.resources?.reduce((sum, r) => sum + (r.current_instances * r.cost_per_hour), 0) || 0;
  const criticalFailures = data?.failure_predictions?.filter(f => f.failure_probability > 0.7) || [];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                <Zap className="h-8 w-8 text-blue-600" />
                Predictive Auto-Scaler
              </h1>
              <p className="text-gray-600">AI-powered infrastructure optimization</p>
            </div>
            
            <div className="flex items-center gap-4">
              <select 
                value={selectedResource} 
                onChange={(e) => setSelectedResource(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 bg-white"
              >
                {data?.resources?.map(resource => (
                  <option key={resource.id} value={resource.id}>
                    {resource.id} ({resource.provider})
                  </option>
                ))}
              </select>
              
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`flex items-center gap-2 px-4 py-2 rounded-md ${
                  autoRefresh 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-gray-100 text-gray-600'
                }`}
              >
                <Activity className="h-4 w-4" />
                {autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Current Instances</p>
                <p className="text-3xl font-bold text-gray-900">
                  {currentResource?.current_instances || 0}
                </p>
              </div>
              <Server className="h-8 w-8 text-blue-600" />
            </div>
            <div className="mt-2 flex items-center text-sm">
              <span className="text-gray-500">
                Min: {currentResource?.min_instances} / Max: {currentResource?.max_instances}
              </span>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Hourly Cost</p>
                <p className="text-3xl font-bold text-gray-900">
                  ${totalCost.toFixed(2)}
                </p>
              </div>
              <DollarSign className="h-8 w-8 text-green-600" />
            </div>
            <div className="mt-2 flex items-center text-sm">
              {data?.cost_optimization && (
                <span className={`${
                  data.cost_optimization.savings_percentage > 0 
                    ? 'text-green-600' 
                    : 'text-gray-500'
                }`}>
                  {data.cost_optimization.savings_percentage > 0 ? '↓' : '→'} 
                  {Math.abs(data.cost_optimization.savings_percentage).toFixed(1)}% potential savings
                </span>
              )}
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Critical Alerts</p>
                <p className="text-3xl font-bold text-gray-900">
                  {criticalFailures.length}
                </p>
              </div>
              <AlertTriangle className={`h-8 w-8 ${
                criticalFailures.length > 0 ? 'text-red-600' : 'text-gray-400'
              }`} />
            </div>
            <div className="mt-2 flex items-center text-sm">
              <span className="text-gray-500">
                {criticalFailures.length > 0 ? 'Requires attention' : 'All systems healthy'}
              </span>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Prediction Confidence</p>
                <p className="text-3xl font-bold text-gray-900">
                  {trafficData.length > 0 
                    ? Math.round(trafficData.reduce((sum, d) => sum + d.confidence, 0) / trafficData.length)
                    : 0
                  }%
                </p>
              </div>
              <Shield className="h-8 w-8 text-purple-600" />
            </div>
            <div className="mt-2 flex items-center text-sm">
              <span className="text-gray-500">
                Next 24 hours average
              </span>
            </div>
          </div>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Traffic Predictions */}
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-blue-600" />
              Traffic Predictions (24h)
            </h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trafficData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area 
                    type="monotone" 
                    dataKey="load" 
                    stroke="#3b82f6" 
                    fill="#3b82f6" 
                    fillOpacity={0.3}
                    name="Predicted Load"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Spike Risk & Confidence */}
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-orange-600" />
              Spike Risk & Confidence
            </h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trafficData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="spike_risk" 
                    stroke="#f59e0b" 
                    strokeWidth={2}
                    name="Spike Risk %"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="confidence" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    name="Confidence %"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Resource Distribution & Alerts */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Resource Distribution */}
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Server className="h-5 w-5 text-blue-600" />
              Resource Distribution
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={data?.resources?.map(r => ({
                      name: r.id,
                      value: r.current_instances,
                      cost: r.cost_per_hour * r.current_instances
                    }))}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {data?.resources?.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Failure Predictions */}
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Shield className="h-5 w-5 text-red-600" />
              Failure Predictions
            </h3>
            <div className="space-y-4">
              {data?.failure_predictions?.length ? (
                data.failure_predictions.map((failure, index) => (
                  <div 
                    key={index}
                    className={`p-4 rounded-lg border-l-4 ${
                      failure.failure_probability > 0.7 
                        ? 'border-red-500 bg-red-50' 
                        : 'border-yellow-500 bg-yellow-50'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-gray-900">
                        {failure.failure_type.replace('_', ' ').toUpperCase()}
                      </span>
                      <span className={`text-sm font-bold ${
                        failure.failure_probability > 0.7 ? 'text-red-600' : 'text-yellow-600'
                      }`}>
                        {Math.round(failure.failure_probability * 100)}%
                      </span>
                    </div>
                    <p className="text-sm text-gray-600">
                      {failure.recommended_action}
                    </p>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Shield className="h-12 w-12 mx-auto mb-4 text-green-500" />
                  <p>No failure risks detected</p>
                </div>
              )}
            </div>
          </div>

          {/* Cost Optimization */}
          <div className="bg-white p-6 rounded-lg shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <DollarSign className="h-5 w-5 text-green-600" />
              Cost Optimization
            </h3>
            <div className="space-y-4">
              {data?.cost_optimization && (
                <>
                  <div className="bg-green-50 p-4 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-600">Potential Savings</span>
                      <span className="text-lg font-bold text-green-600">
                        {data.cost_optimization.savings_percentage.toFixed(1)}%
                      </span>
                    </div>
                    <div className="text-sm text-gray-600">
                      ${(data.cost_optimization.current_cost_per_hour - data.cost_optimization.optimized_cost_per_hour).toFixed(2)}/hour
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium text-gray-900">Recommendations:</h4>
                    {data.cost_optimization.recommendations.map((rec, index) => (
                      <div key={index} className="text-sm text-gray-600 flex items-start gap-2">
                        <span className="text-green-600 mt-1">•</span>
                        <span>{rec}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Real-time Status */}
        <div className="mt-8 bg-white p-4 rounded-lg shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <Clock className="h-4 w-4" />
              Last updated: {new Date().toLocaleTimeString()}
            </div>
            <button
              onClick={fetchDashboardData}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              <Activity className="h-4 w-4" />
              Refresh Now
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
