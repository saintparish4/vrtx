import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { AutoScaler } from '@/lib/auto-scaler';
import { ResourceConfig } from '@/types';

const ScalingRequestSchema = z.object({
  resource_id: z.string(),
  action: z.enum(['scale_up', 'scale_down', 'auto']),
  target_instances: z.number().optional(),
  force: z.boolean().default(false)
});

const ResourceConfigSchema = z.object({
  id: z.string(),
  type: z.enum(['ec2', 'gce', 'azure_vm', 'kubernetes']),
  provider: z.enum(['aws', 'gcp', 'azure', 'k8s']),
  min_instances: z.number(),
  max_instances: z.number(),
  current_instances: z.number(),
  target_cpu: z.number(),
  target_memory: z.number(),
  cost_per_hour: z.number()
});

let autoScaler: AutoScaler;

// Initialize AutoScaler singleton
if (!autoScaler) {
  autoScaler = new AutoScaler();
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { resource_id, action, target_instances, force } = 
      ScalingRequestSchema.parse(body);

    // Get resource configuration (in production, this would come from database)
    const resourceConfig: ResourceConfig = {
      id: resource_id,
      type: 'ec2',
      provider: 'aws',
      min_instances: 1,
      max_instances: 10,
      current_instances: 3,
      target_cpu: 70,
      target_memory: 80,
      cost_per_hour: 0.0416
    };

    if (action === 'auto') {
      // Get latest predictions
      const predictionsResponse = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL}/api/predictions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resource_id })
      });

      if (!predictionsResponse.ok) {
        return NextResponse.json(
          { error: 'Failed to get predictions for auto-scaling' },
          { status: 500 }
        );
      }

      const predictionsData = await predictionsResponse.json();
      const { traffic_predictions, failure_predictions, cost_optimization } = predictionsData.data;

      // Process scaling decision
      const decisions = await autoScaler.processScalingDecision(
        resourceConfig,
        traffic_predictions || [],
        failure_predictions || [],
        cost_optimization || { savings_percentage: 0, recommendations: [] }
      );

      return NextResponse.json({
        success: true,
        decisions,
        message: `Processed ${decisions.length} scaling decisions`
      });

    } else {
      // Manual scaling
      if (!target_instances) {
        return NextResponse.json(
          { error: 'target_instances required for manual scaling' },
          { status: 400 }
        );
      }

      const decision = {
        resource_id,
        action,
        target_instances,
        reason: 'Manual scaling request',
        confidence: 1.0,
        estimated_cost_impact: (target_instances - resourceConfig.current_instances) * resourceConfig.cost_per_hour,
        timestamp: Date.now()
      };

      // Execute manual scaling
      // Implementation would call cloud provider APIs directly
      
      return NextResponse.json({
        success: true,
        decision,
        message: `Manual ${action} to ${target_instances} instances initiated`
      });
    }

  } catch (error) {
    console.error('Scaling API error:', error);
    return NextResponse.json(
      { error: 'Failed to process scaling request', details: error },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const resourceId = searchParams.get('resource_id');
  const hours = parseInt(searchParams.get('hours') || '24');

  if (!resourceId) {
    return NextResponse.json(
      { error: 'resource_id parameter is required' },
      { status: 400 }
    );
  }

  try {
    const history = await autoScaler.getScalingHistory(resourceId, hours);
    
    return NextResponse.json({
      success: true,
      data: {
        resource_id: resourceId,
        history,
        total_actions: history.length
      }
    });

  } catch (error) {
    console.error('Scaling history error:', error);
    return NextResponse.json(
      { error: 'Failed to retrieve scaling history' },
      { status: 500 }
    );
  }
}
