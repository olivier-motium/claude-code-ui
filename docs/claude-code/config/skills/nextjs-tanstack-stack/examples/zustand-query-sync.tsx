'use client';

/**
 * Zustand + Query Sync Example
 *
 * Demonstrates synchronizing Zustand filter state with TanStack Query:
 * - Zustand store for filter state
 * - Atomic selectors to prevent re-renders
 * - Query key derived from store values
 * - Debounced search input
 * - URL sync with shallow routing
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { useQuery } from '@tanstack/react-query';
import { useDeferredValue, useEffect, useMemo } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

// Types
interface Market {
  id: string;
  name: string;
  category: string;
  volume: number;
  price: number;
}

type SortField = 'volume' | 'price' | 'name';
type SortDirection = 'asc' | 'desc';

// Store Types
interface MarketFiltersState {
  search: string;
  category: string | null;
  sortField: SortField;
  sortDirection: SortDirection;
  actions: {
    setSearch: (search: string) => void;
    setCategory: (category: string | null) => void;
    setSort: (field: SortField, direction?: SortDirection) => void;
    reset: () => void;
  };
}

// Initial state
const initialState = {
  search: '',
  category: null,
  sortField: 'volume' as SortField,
  sortDirection: 'desc' as SortDirection,
};

// Zustand Store
export const useMarketFilters = create<MarketFiltersState>()(
  devtools(
    persist(
      (set) => ({
        ...initialState,
        actions: {
          setSearch: (search) => set({ search }, false, 'setSearch'),
          setCategory: (category) => set({ category }, false, 'setCategory'),
          setSort: (field, direction) =>
            set(
              (state) => ({
                sortField: field,
                sortDirection:
                  direction ??
                  (state.sortField === field
                    ? state.sortDirection === 'asc'
                      ? 'desc'
                      : 'asc'
                    : 'desc'),
              }),
              false,
              'setSort'
            ),
          reset: () => set(initialState, false, 'reset'),
        },
      }),
      {
        name: 'market-filters',
        partialize: (state) => ({
          // Only persist sort preferences
          sortField: state.sortField,
          sortDirection: state.sortDirection,
        }),
      }
    ),
    { name: 'MarketFilters' }
  )
);

// Atomic Selectors (prevent unnecessary re-renders)
export const selectSearch = (state: MarketFiltersState) => state.search;
export const selectCategory = (state: MarketFiltersState) => state.category;
export const selectSortField = (state: MarketFiltersState) => state.sortField;
export const selectSortDirection = (state: MarketFiltersState) => state.sortDirection;
export const selectActions = (state: MarketFiltersState) => state.actions;

// Mock API
const mockMarkets: Market[] = [
  { id: '1', name: 'Bitcoin Price', category: 'crypto', volume: 1500000, price: 45000 },
  { id: '2', name: 'Election 2024', category: 'politics', volume: 2000000, price: 0.55 },
  { id: '3', name: 'Super Bowl Winner', category: 'sports', volume: 800000, price: 0.32 },
  { id: '4', name: 'Ethereum Merge', category: 'crypto', volume: 1200000, price: 3200 },
  { id: '5', name: 'Oscar Best Picture', category: 'entertainment', volume: 400000, price: 0.18 },
  { id: '6', name: 'Fed Rate Decision', category: 'politics', volume: 900000, price: 0.72 },
  { id: '7', name: 'World Cup Final', category: 'sports', volume: 1800000, price: 0.41 },
  { id: '8', name: 'Solana Price', category: 'crypto', volume: 600000, price: 120 },
];

interface FetchParams {
  search: string;
  category: string | null;
  sortField: SortField;
  sortDirection: SortDirection;
}

async function fetchMarkets(params: FetchParams): Promise<Market[]> {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 500));

  let result = [...mockMarkets];

  // Filter by search
  if (params.search) {
    const searchLower = params.search.toLowerCase();
    result = result.filter((m) => m.name.toLowerCase().includes(searchLower));
  }

  // Filter by category
  if (params.category) {
    result = result.filter((m) => m.category === params.category);
  }

  // Sort
  result.sort((a, b) => {
    const aVal = a[params.sortField];
    const bVal = b[params.sortField];
    const comparison = typeof aVal === 'string' ? aVal.localeCompare(bVal as string) : (aVal as number) - (bVal as number);
    return params.sortDirection === 'asc' ? comparison : -comparison;
  });

  return result;
}

// Hook: Filtered Markets with Query
export function useFilteredMarkets() {
  const search = useMarketFilters(selectSearch);
  const category = useMarketFilters(selectCategory);
  const sortField = useMarketFilters(selectSortField);
  const sortDirection = useMarketFilters(selectSortDirection);

  // Defer search to prevent UI jank
  const deferredSearch = useDeferredValue(search);
  const isSearching = search !== deferredSearch;

  // Query with all filter values in key
  const query = useQuery({
    queryKey: ['markets', { search: deferredSearch, category, sortField, sortDirection }],
    queryFn: () => fetchMarkets({ search: deferredSearch, category, sortField, sortDirection }),
    placeholderData: (previousData) => previousData,
  });

  return { ...query, isSearching };
}

// Hook: URL Sync
function useUrlSync() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const category = useMarketFilters(selectCategory);
  const { setCategory } = useMarketFilters(selectActions);

  // Sync URL → Store on mount
  useEffect(() => {
    const urlCategory = searchParams.get('category');
    if (urlCategory && urlCategory !== category) {
      setCategory(urlCategory);
    }
  }, []); // Only on mount

  // Sync Store → URL on change
  useEffect(() => {
    const params = new URLSearchParams(searchParams.toString());
    if (category) {
      params.set('category', category);
    } else {
      params.delete('category');
    }
    const newUrl = params.toString() ? `?${params.toString()}` : window.location.pathname;
    router.replace(newUrl, { scroll: false });
  }, [category, router, searchParams]);
}

// Components
function SearchInput() {
  const search = useMarketFilters(selectSearch);
  const { setSearch } = useMarketFilters(selectActions);

  return (
    <input
      type="text"
      value={search}
      onChange={(e) => setSearch(e.target.value)}
      placeholder="Search markets..."
      className="w-full px-4 py-2 border rounded-lg"
    />
  );
}

const categories = ['crypto', 'politics', 'sports', 'entertainment'];

function CategoryFilter() {
  const category = useMarketFilters(selectCategory);
  const { setCategory } = useMarketFilters(selectActions);

  return (
    <div className="flex gap-2">
      <button
        onClick={() => setCategory(null)}
        className={`px-3 py-1 rounded ${!category ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}
      >
        All
      </button>
      {categories.map((cat) => (
        <button
          key={cat}
          onClick={() => setCategory(cat)}
          className={`px-3 py-1 rounded capitalize ${
            category === cat ? 'bg-blue-600 text-white' : 'bg-gray-100'
          }`}
        >
          {cat}
        </button>
      ))}
    </div>
  );
}

function SortControls() {
  const sortField = useMarketFilters(selectSortField);
  const sortDirection = useMarketFilters(selectSortDirection);
  const { setSort } = useMarketFilters(selectActions);

  const fields: { value: SortField; label: string }[] = [
    { value: 'volume', label: 'Volume' },
    { value: 'price', label: 'Price' },
    { value: 'name', label: 'Name' },
  ];

  return (
    <div className="flex gap-2">
      <span className="text-sm text-gray-500">Sort by:</span>
      {fields.map((field) => (
        <button
          key={field.value}
          onClick={() => setSort(field.value)}
          className={`px-3 py-1 rounded text-sm ${
            sortField === field.value ? 'bg-blue-600 text-white' : 'bg-gray-100'
          }`}
        >
          {field.label}
          {sortField === field.value && (sortDirection === 'asc' ? ' ↑' : ' ↓')}
        </button>
      ))}
    </div>
  );
}

function ResetButton() {
  const { reset } = useMarketFilters(selectActions);

  return (
    <button
      onClick={reset}
      className="px-3 py-1 text-sm text-gray-600 hover:text-gray-900"
    >
      Reset Filters
    </button>
  );
}

function MarketList() {
  const { data, isLoading, isFetching, isSearching } = useFilteredMarkets();

  if (isLoading) {
    return (
      <div className="space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-16 bg-gray-100 animate-pulse rounded" />
        ))}
      </div>
    );
  }

  return (
    <div className={`space-y-2 ${(isFetching || isSearching) ? 'opacity-70' : ''}`}>
      {data?.length === 0 ? (
        <p className="text-gray-500 text-center py-8">No markets found</p>
      ) : (
        data?.map((market) => (
          <div
            key={market.id}
            className="p-4 border rounded-lg hover:bg-gray-50 flex justify-between items-center"
          >
            <div>
              <h3 className="font-medium">{market.name}</h3>
              <span className="text-sm text-gray-500 capitalize">{market.category}</span>
            </div>
            <div className="text-right">
              <p className="font-medium">${market.price.toLocaleString()}</p>
              <p className="text-sm text-gray-500">Vol: ${market.volume.toLocaleString()}</p>
            </div>
          </div>
        ))
      )}
    </div>
  );
}

// Main Component
export function ZustandQuerySync() {
  useUrlSync();

  return (
    <div className="max-w-2xl mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Markets</h1>
        <ResetButton />
      </div>

      <div className="space-y-4">
        <SearchInput />
        <CategoryFilter />
        <SortControls />
      </div>

      <MarketList />
    </div>
  );
}

export default ZustandQuerySync;
