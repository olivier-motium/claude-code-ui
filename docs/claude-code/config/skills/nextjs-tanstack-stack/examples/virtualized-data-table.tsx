'use client';

/**
 * Virtualized Data Table Example
 *
 * Demonstrates TanStack Table + TanStack Virtual integration for handling
 * 10,000+ rows with smooth scrolling performance.
 *
 * Features:
 * - Row virtualization with overscan
 * - Sortable columns
 * - Column visibility toggle
 * - Type-safe column definitions
 * - Performance metrics logging
 */

import { useRef, useState, useMemo, useEffect } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  type ColumnDef,
  type SortingState,
  type VisibilityState,
} from '@tanstack/react-table';

// Types
interface Market {
  id: string;
  name: string;
  category: string;
  volume: number;
  price: number;
  change24h: number;
  liquidity: number;
  updatedAt: Date;
}

// Generate mock data
function generateMockData(count: number): Market[] {
  const categories = ['Politics', 'Sports', 'Crypto', 'Entertainment', 'Science'];

  return Array.from({ length: count }, (_, i) => ({
    id: `market-${i}`,
    name: `Market ${i + 1}`,
    category: categories[i % categories.length],
    volume: Math.random() * 1000000,
    price: Math.random() * 100,
    change24h: (Math.random() - 0.5) * 20,
    liquidity: Math.random() * 500000,
    updatedAt: new Date(Date.now() - Math.random() * 86400000 * 7),
  }));
}

// Column definitions
const columnHelper = createColumnHelper<Market>();

const columns: ColumnDef<Market>[] = [
  columnHelper.accessor('name', {
    header: 'Market',
    cell: (info) => (
      <span className="font-medium">{info.getValue()}</span>
    ),
    size: 200,
  }),
  columnHelper.accessor('category', {
    header: 'Category',
    cell: (info) => (
      <span className="px-2 py-1 bg-gray-100 rounded text-sm">
        {info.getValue()}
      </span>
    ),
    size: 120,
  }),
  columnHelper.accessor('volume', {
    header: 'Volume',
    cell: (info) => `$${info.getValue().toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
    sortingFn: 'basic',
    size: 120,
  }),
  columnHelper.accessor('price', {
    header: 'Price',
    cell: (info) => `$${info.getValue().toFixed(2)}`,
    size: 100,
  }),
  columnHelper.accessor('change24h', {
    header: '24h Change',
    cell: (info) => {
      const value = info.getValue();
      const isPositive = value >= 0;
      return (
        <span className={isPositive ? 'text-green-600' : 'text-red-600'}>
          {isPositive ? '+' : ''}{value.toFixed(2)}%
        </span>
      );
    },
    size: 100,
  }),
  columnHelper.accessor('liquidity', {
    header: 'Liquidity',
    cell: (info) => `$${info.getValue().toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
    size: 120,
  }),
  columnHelper.accessor('updatedAt', {
    header: 'Updated',
    cell: (info) => {
      const date = info.getValue();
      const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });
      const diffMs = date.getTime() - Date.now();
      const diffHours = Math.round(diffMs / (1000 * 60 * 60));
      return rtf.format(diffHours, 'hour');
    },
    sortingFn: 'datetime',
    size: 120,
  }),
];

// Component
interface VirtualizedDataTableProps {
  rowCount?: number;
}

export function VirtualizedDataTable({ rowCount = 10000 }: VirtualizedDataTableProps) {
  const parentRef = useRef<HTMLDivElement>(null);
  const [sorting, setSorting] = useState<SortingState>([{ id: 'volume', desc: true }]);
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({});

  // Performance tracking
  const [renderMetrics, setRenderMetrics] = useState({ renderCount: 0, lastRenderMs: 0 });

  // Generate data once
  const data = useMemo(() => generateMockData(rowCount), [rowCount]);

  // Table instance
  const table = useReactTable({
    data,
    columns,
    state: { sorting, columnVisibility },
    onSortingChange: setSorting,
    onColumnVisibilityChange: setColumnVisibility,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const { rows } = table.getRowModel();

  // Virtualizer
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 48,
    overscan: 10,
  });

  const virtualRows = virtualizer.getVirtualItems();

  // Track render performance
  useEffect(() => {
    const start = performance.now();
    return () => {
      const duration = performance.now() - start;
      setRenderMetrics((prev) => ({
        renderCount: prev.renderCount + 1,
        lastRenderMs: duration,
      }));
    };
  });

  return (
    <div className="space-y-4">
      {/* Metrics */}
      <div className="flex justify-between items-center text-sm text-gray-500">
        <span>
          Showing {virtualRows.length} of {rows.length} rows (virtualized)
        </span>
        <span>
          Renders: {renderMetrics.renderCount} | Last: {renderMetrics.lastRenderMs.toFixed(1)}ms
        </span>
      </div>

      {/* Column visibility toggle */}
      <div className="flex gap-2 flex-wrap">
        {table.getAllLeafColumns().map((column) => (
          <label key={column.id} className="flex items-center gap-1 text-sm">
            <input
              type="checkbox"
              checked={column.getIsVisible()}
              onChange={column.getToggleVisibilityHandler()}
            />
            {column.id}
          </label>
        ))}
      </div>

      {/* Table container */}
      <div
        ref={parentRef}
        className="h-[600px] overflow-auto border rounded-lg"
      >
        <table className="w-full border-collapse">
          {/* Header */}
          <thead className="sticky top-0 bg-gray-50 z-10">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    onClick={header.column.getToggleSortingHandler()}
                    className="px-4 py-3 text-left text-sm font-semibold text-gray-900 cursor-pointer select-none border-b"
                    style={{ width: header.getSize() }}
                  >
                    <div className="flex items-center gap-1">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {{
                        asc: ' ↑',
                        desc: ' ↓',
                      }[header.column.getIsSorted() as string] ?? null}
                    </div>
                  </th>
                ))}
              </tr>
            ))}
          </thead>

          {/* Body with virtualization */}
          <tbody>
            {/* Spacer for virtualization */}
            <tr style={{ height: `${virtualizer.getTotalSize()}px` }}>
              <td colSpan={columns.length} className="relative p-0">
                {virtualRows.map((virtualRow) => {
                  const row = rows[virtualRow.index];
                  return (
                    <div
                      key={row.id}
                      className="absolute left-0 right-0 flex border-b hover:bg-gray-50"
                      style={{
                        top: virtualRow.start,
                        height: `${virtualRow.size}px`,
                      }}
                    >
                      {row.getVisibleCells().map((cell) => (
                        <div
                          key={cell.id}
                          className="px-4 py-3 text-sm text-gray-700 flex items-center"
                          style={{ width: cell.column.getSize() }}
                        >
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </div>
                      ))}
                    </div>
                  );
                })}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Usage example
export default function VirtualizedTableDemo() {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Virtualized Data Table</h1>
      <VirtualizedDataTable rowCount={10000} />
    </div>
  );
}
