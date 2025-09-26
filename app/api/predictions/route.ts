import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

const PredictionRequestSchema = z.object({
  resource_id: z.string(),
  hours_ahead: z.number().min(1).max(168).default(24),
  include_failures: z.boolean().default(true),
  include_cost_optimization: z.boolean().default(true)
});

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { resource_id, hours_ahead, include_failures, include_cost_optimization } = 
      PredictionRequestSchema.parse(body);

    // Call Python prediction engine
    const pythonScript = `
import sys
sys.path.append('lib')
from prediction_engine import TrafficPredictor, FailurePredictor, CostOptimizer
import json

config = {'redis_host': 'localhost', 'redis_port': 6379}
predictor = TrafficPredictor(config)

# Generate predictions
predictions = predictor.predict_traffic('${resource_id}', hours_ahead=${hours_ahead})

result = {
  'traffic_predictions': predictions,
  'resource_id': '${resource_id}',
  'timestamp': $(Date.now())
}

${include_failures ? `
failure_predictor = FailurePredictor()
sample_data = predictor.get_recent_metrics('${resource_id}', hours=24)
health_analysis = failure_predictor.analyze_resource_health('${resource_id}', sample_data)
result['failure_predictions'] = health_analysis['failure_predictions']
` : ''}

${include_cost_optimization ? `
cost_optimizer = CostOptimizer()
current_config = {
  'provider': 'aws',
  'instance_type': 't3.medium', 
  'current_instances': 3
}
cost_optimization = cost_optimizer.optimize_resource_allocation(predictions, current_config)
result['cost_optimization'] = cost_optimization
` : ''}

print(json.dumps(result))
`;

    const { stdout, stderr } = await execAsync(`python3 -c "${pythonScript}"`);
    
    if (stderr) {
      console.error('Python script error:', stderr);
      return NextResponse.json(
        { error: 'Prediction engine error', details: stderr },
        { status: 500 }
      );
    }

    const predictions = JSON.parse(stdout);
    
    return NextResponse.json({
      success: true,
      data: predictions
    });

  } catch (error) {
    console.error('Prediction API error:', error);
    return NextResponse.json(
      { error: 'Failed to generate predictions', details: error },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const resourceId = searchParams.get('resource_id');
  
  if (!resourceId) {
    return NextResponse.json(
      { error: 'resource_id parameter is required' },
      { status: 400 }
    );
  }

  try {
    // Return cached predictions from Redis
    const Redis = require('ioredis');
    const redis = new Redis();
    
    const cachedData = await redis.get(`predictions:${resourceId}`);
    
    if (cachedData) {
      return NextResponse.json({
        success: true,
        data: JSON.parse(cachedData),
        cached: true
      });
    }

    return NextResponse.json({
      success: false,
      message: 'No cached predictions found'
    });

  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to retrieve predictions' },
      { status: 500 }
    );
  }
}
