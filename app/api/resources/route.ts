import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { ResourceConfig } from '@/types';

const CreateResourceSchema = z.object({
  id: z.string(),
  type: z.enum(['ec2', 'gce', 'azure_vm', 'kubernetes']),
  provider: z.enum(['aws', 'gcp', 'azure', 'k8s']),
  min_instances: z.number().min(1),
  max_instances: z.number().min(1),
  current_instances: z.number().min(1),
  target_cpu: z.number().min(10).max(100),
  target_memory: z.number().min(10).max(100),
  cost_per_hour: z.number().min(0)
});

// In-memory storage for demo (use database in production)
let resources: ResourceConfig[] = [
  {
    id: 'web-server-cluster-1',
    type: 'ec2',
    provider: 'aws',
    min_instances: 2,
    max_instances: 20,
    current_instances: 5,
    target_cpu: 70,
    target_memory: 80,
    cost_per_hour: 0.0416
  },
  {
    id: 'api-gateway-cluster',
    type: 'kubernetes',
    provider: 'k8s',
    min_instances: 3,
    max_instances: 15,
    current_instances: 6,
    target_cpu: 65,
    target_memory: 75,
    cost_per_hour: 0.02
  },
  {
    id: 'database-cluster',
    type: 'gce',
    provider: 'gcp',
    min_instances: 2,
    max_instances: 8,
    current_instances: 3,
    target_cpu: 80,
    target_memory: 85,
    cost_per_hour: 0.0504
  }
];

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const resourceId = searchParams.get('resource_id');
  const provider = searchParams.get('provider');

  try {
    let filteredResources = resources;

    if (resourceId) {
      filteredResources = resources.filter(r => r.id === resourceId);
    }

    if (provider) {
      filteredResources = filteredResources.filter(r => r.provider === provider);
    }

    return NextResponse.json({
      success: true,
      data: filteredResources,
      total: filteredResources.length
    });

  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to retrieve resources' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const resourceConfig = CreateResourceSchema.parse(body);

    // Check if resource already exists
    const existingResource = resources.find(r => r.id === resourceConfig.id);
    if (existingResource) {
      return NextResponse.json(
        { error: 'Resource with this ID already exists' },
        { status: 409 }
      );
    }

    // Validate instance constraints
    if (resourceConfig.min_instances > resourceConfig.max_instances) {
      return NextResponse.json(
        { error: 'min_instances cannot be greater than max_instances' },
        { status: 400 }
      );
    }

    if (resourceConfig.current_instances < resourceConfig.min_instances ||
        resourceConfig.current_instances > resourceConfig.max_instances) {
      return NextResponse.json(
        { error: 'current_instances must be between min_instances and max_instances' },
        { status: 400 }
      );
    }

    // Add resource
    resources.push(resourceConfig);

    return NextResponse.json({
      success: true,
      data: resourceConfig,
      message: 'Resource created successfully'
    }, { status: 201 });

  } catch (error) {
    console.error('Create resource error:', error);
    return NextResponse.json(
      { error: 'Failed to create resource', details: error },
      { status: 500 }
    );
  }
}

export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const resourceConfig = CreateResourceSchema.parse(body);

    const resourceIndex = resources.findIndex(r => r.id === resourceConfig.id);
    if (resourceIndex === -1) {
      return NextResponse.json(
        { error: 'Resource not found' },
        { status: 404 }
      );
    }

    // Update resource
    resources[resourceIndex] = resourceConfig;

    return NextResponse.json({
      success: true,
      data: resourceConfig,
      message: 'Resource updated successfully'
    });

  } catch (error) {
    console.error('Update resource error:', error);
    return NextResponse.json(
      { error: 'Failed to update resource', details: error },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const resourceId = searchParams.get('resource_id');

  if (!resourceId) {
    return NextResponse.json(
      { error: 'resource_id parameter is required' },
      { status: 400 }
    );
  }

  try {
    const resourceIndex = resources.findIndex(r => r.id === resourceId);
    if (resourceIndex === -1) {
      return NextResponse.json(
        { error: 'Resource not found' },
        { status: 404 }
      );
    }

    const deletedResource = resources.splice(resourceIndex, 1)[0];

    return NextResponse.json({
      success: true,
      data: deletedResource,
      message: 'Resource deleted successfully'
    });

  } catch (error) {
    console.error('Delete resource error:', error);
    return NextResponse.json(
      { error: 'Failed to delete resource' },
      { status: 500 }
    );
  }
}
