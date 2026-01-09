'use client';

/**
 * Query with Error Boundary Example
 *
 * Demonstrates TanStack Query + Error Boundary composition with:
 * - QueryErrorResetBoundary for coordinated retry
 * - Typed error handling
 * - Loading skeleton
 * - Recovery mechanism
 */

import { Component, type ReactNode, type ErrorInfo, Suspense } from 'react';
import {
  useQuery,
  useSuspenseQuery,
  QueryErrorResetBoundary,
  useQueryClient,
} from '@tanstack/react-query';

// Types
interface Market {
  id: string;
  name: string;
  price: number;
  volume: number;
}

interface ApiError {
  message: string;
  status: number;
}

// Error Boundary Component
interface ErrorBoundaryProps {
  children: ReactNode;
  fallback: (error: Error, reset: () => void) => ReactNode;
  onError?: (error: Error, info: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  error: Error | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary]', {
      error: error.message,
      stack: error.stack,
      componentStack: info.componentStack,
      timestamp: new Date().toISOString(),
    });
    this.props.onError?.(error, info);
  }

  reset = () => {
    this.setState({ error: null });
  };

  render() {
    if (this.state.error) {
      return this.props.fallback(this.state.error, this.reset);
    }
    return this.props.children;
  }
}

// API mock
async function fetchMarkets(): Promise<Market[]> {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 1000));

  // Simulate random failure (30% chance)
  if (Math.random() < 0.3) {
    const error = new Error('Failed to fetch markets') as Error & { status: number };
    error.status = 500;
    throw error;
  }

  return [
    { id: '1', name: 'BTC Price', price: 45000, volume: 1000000 },
    { id: '2', name: 'ETH Price', price: 3000, volume: 500000 },
    { id: '3', name: 'Election 2024', price: 0.55, volume: 2000000 },
  ];
}

// Loading Skeleton
function MarketListSkeleton() {
  return (
    <div className="space-y-4">
      {[1, 2, 3].map((i) => (
        <div key={i} className="p-4 border rounded-lg animate-pulse">
          <div className="h-5 bg-gray-200 rounded w-1/3 mb-2" />
          <div className="h-4 bg-gray-200 rounded w-1/4" />
        </div>
      ))}
    </div>
  );
}

// Error Fallback
interface ErrorFallbackProps {
  error: Error;
  onRetry: () => void;
}

function ErrorFallback({ error, onRetry }: ErrorFallbackProps) {
  const status = (error as Error & { status?: number }).status;

  return (
    <div className="p-6 bg-red-50 border border-red-200 rounded-lg">
      <h3 className="text-lg font-semibold text-red-800">Something went wrong</h3>
      <p className="text-red-600 mt-1">{error.message}</p>
      {status && <p className="text-red-500 text-sm mt-1">Status: {status}</p>}
      <button
        onClick={onRetry}
        className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
      >
        Try Again
      </button>
    </div>
  );
}

// Market List Component (uses Suspense query)
function MarketList() {
  const { data: markets } = useSuspenseQuery({
    queryKey: ['markets'],
    queryFn: fetchMarkets,
  });

  return (
    <div className="space-y-4">
      {markets.map((market) => (
        <div key={market.id} className="p-4 border rounded-lg hover:bg-gray-50">
          <h3 className="font-semibold">{market.name}</h3>
          <p className="text-gray-600">
            Price: ${market.price.toLocaleString()} | Volume: ${market.volume.toLocaleString()}
          </p>
        </div>
      ))}
    </div>
  );
}

// Pattern 1: With QueryErrorResetBoundary + Custom ErrorBoundary
export function MarketsWithErrorBoundary() {
  return (
    <div className="p-6">
      <h2 className="text-xl font-bold mb-4">Markets (with Error Boundary)</h2>

      <QueryErrorResetBoundary>
        {({ reset }) => (
          <ErrorBoundary
            fallback={(error, localReset) => (
              <ErrorFallback
                error={error}
                onRetry={() => {
                  reset(); // Reset query state
                  localReset(); // Reset error boundary state
                }}
              />
            )}
            onError={(error) => {
              // Send to error tracking service
              console.log('Reporting error to tracking service:', error);
            }}
          >
            <Suspense fallback={<MarketListSkeleton />}>
              <MarketList />
            </Suspense>
          </ErrorBoundary>
        )}
      </QueryErrorResetBoundary>
    </div>
  );
}

// Pattern 2: Without Suspense (manual loading/error states)
function MarketListWithStates() {
  const queryClient = useQueryClient();

  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['markets-no-suspense'],
    queryFn: fetchMarkets,
    retry: 2,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 10000),
  });

  if (isLoading) {
    return <MarketListSkeleton />;
  }

  if (isError) {
    return (
      <ErrorFallback
        error={error}
        onRetry={() => {
          // Clear error state and refetch
          queryClient.resetQueries({ queryKey: ['markets-no-suspense'] });
          refetch();
        }}
      />
    );
  }

  return (
    <div className="space-y-4">
      {data?.map((market) => (
        <div key={market.id} className="p-4 border rounded-lg hover:bg-gray-50">
          <h3 className="font-semibold">{market.name}</h3>
          <p className="text-gray-600">
            Price: ${market.price.toLocaleString()} | Volume: ${market.volume.toLocaleString()}
          </p>
        </div>
      ))}
    </div>
  );
}

export function MarketsWithManualStates() {
  return (
    <div className="p-6">
      <h2 className="text-xl font-bold mb-4">Markets (Manual States)</h2>
      <MarketListWithStates />
    </div>
  );
}

// Demo page showing both patterns
export default function ErrorBoundaryDemo() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold mb-2">Query Error Handling Patterns</h1>
        <p className="text-gray-600">
          Refresh to see random failures (30% chance). Both patterns handle errors gracefully.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <MarketsWithErrorBoundary />
        <MarketsWithManualStates />
      </div>
    </div>
  );
}
