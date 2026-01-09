# Next.js App Router Patterns

Production patterns for Next.js 14+ App Router architecture.

## Server vs Client Components

### Decision Tree

```
Start with Server Component (default)
       │
       ├─ Need event handlers? (onClick, onChange, onSubmit)
       │     └─ YES → Client Component
       │
       ├─ Need browser APIs? (localStorage, window, navigator)
       │     └─ YES → Client Component
       │
       ├─ Need React hooks? (useState, useEffect, useRef, useContext)
       │     └─ YES → Client Component
       │
       ├─ Need TanStack hooks? (useQuery, useForm, useReactTable)
       │     └─ YES → Client Component
       │
       ├─ Need Zustand store?
       │     └─ YES → Client Component
       │
       └─ Otherwise → Keep as Server Component
```

### Server Component Benefits

```tsx
// app/markets/page.tsx (Server Component - default)
import { db } from '@/lib/db';
import { MarketCard } from './market-card';

// ✅ Direct database access (no API layer needed)
// ✅ Access to secrets and env vars
// ✅ Zero client-side JavaScript
// ✅ SEO-friendly content
export default async function MarketsPage() {
  const markets = await db.market.findMany({
    where: { status: 'active' },
    orderBy: { volume: 'desc' },
  });

  return (
    <div className="grid grid-cols-3 gap-4">
      {markets.map((market) => (
        <MarketCard key={market.id} market={market} />
      ))}
    </div>
  );
}
```

### Client Component Pattern

```tsx
// components/market-card.tsx
'use client'; // ← Only add when necessary

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';

interface Props {
  market: Market;
}

export function MarketCard({ market }: Props) {
  const [expanded, setExpanded] = useState(false);

  const favoriteMutation = useMutation({
    mutationFn: () => toggleFavorite(market.id),
  });

  return (
    <div>
      <h3>{market.name}</h3>
      <button onClick={() => setExpanded(!expanded)}>
        {expanded ? 'Collapse' : 'Expand'}
      </button>
      <button onClick={() => favoriteMutation.mutate()}>
        {market.isFavorite ? 'Unfavorite' : 'Favorite'}
      </button>
    </div>
  );
}
```

---

## Data Fetching

### Server Component Fetch

```tsx
// app/markets/[id]/page.tsx
interface Props {
  params: Promise<{ id: string }>;
}

export default async function MarketPage({ params }: Props) {
  const { id } = await params;

  const market = await fetch(`${process.env.API_URL}/markets/${id}`, {
    next: { revalidate: 60 }, // Revalidate every 60 seconds
  }).then((res) => res.json());

  return <MarketDetails market={market} />;
}

// Generate static params for static generation
export async function generateStaticParams() {
  const markets = await fetch(`${process.env.API_URL}/markets`).then((r) => r.json());
  return markets.map((m: Market) => ({ id: m.id }));
}
```

### Parallel Data Fetching

```tsx
// app/dashboard/page.tsx
export default async function DashboardPage() {
  // Parallel fetch - all start immediately
  const [markets, trades, user] = await Promise.all([
    getMarkets(),
    getRecentTrades(),
    getCurrentUser(),
  ]);

  return (
    <div>
      <UserHeader user={user} />
      <MarketOverview markets={markets} />
      <TradeHistory trades={trades} />
    </div>
  );
}
```

### Server to Client Handoff

```tsx
// app/markets/page.tsx (Server)
import { getMarkets } from '@/lib/api';
import { MarketTable } from './market-table';

export default async function MarketsPage() {
  const initialData = await getMarkets();

  return (
    <main>
      <h1>Markets</h1>
      {/* Pass server data to client component */}
      <MarketTable initialData={initialData} />
    </main>
  );
}

// app/markets/market-table.tsx (Client)
'use client';

import { useQuery } from '@tanstack/react-query';

interface Props {
  initialData: Market[];
}

export function MarketTable({ initialData }: Props) {
  const { data } = useQuery({
    queryKey: ['markets'],
    queryFn: getMarkets,
    initialData, // Use server data as initial
    staleTime: 60 * 1000,
  });

  return <Table data={data} columns={columns} />;
}
```

---

## Layouts and Templates

### Root Layout with Providers

```tsx
// app/layout.tsx
import { Providers } from '@/lib/providers';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });

export const metadata = {
  title: 'My App',
  description: 'Description',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}
```

### Nested Layout

```tsx
// app/dashboard/layout.tsx
import { Sidebar } from '@/components/sidebar';
import { getCurrentUser } from '@/lib/auth';
import { redirect } from 'next/navigation';

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const user = await getCurrentUser();

  if (!user) {
    redirect('/login');
  }

  return (
    <div className="flex">
      <Sidebar user={user} />
      <main className="flex-1 p-6">{children}</main>
    </div>
  );
}
```

---

## Loading and Error States

### Loading UI

```tsx
// app/markets/loading.tsx
export default function Loading() {
  return (
    <div className="grid grid-cols-3 gap-4">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="h-48 bg-gray-200 animate-pulse rounded" />
      ))}
    </div>
  );
}
```

### Error Boundary

```tsx
// app/markets/error.tsx
'use client';

interface Props {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function Error({ error, reset }: Props) {
  return (
    <div className="p-6 bg-red-50 rounded">
      <h2 className="text-red-800 font-bold">Something went wrong!</h2>
      <p className="text-red-600">{error.message}</p>
      <button
        onClick={reset}
        className="mt-4 px-4 py-2 bg-red-600 text-white rounded"
      >
        Try again
      </button>
    </div>
  );
}
```

### Not Found

```tsx
// app/markets/[id]/not-found.tsx
import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="text-center py-12">
      <h2 className="text-2xl font-bold">Market Not Found</h2>
      <p className="text-gray-600 mt-2">The market you're looking for doesn't exist.</p>
      <Link href="/markets" className="mt-4 text-blue-600 underline">
        Back to Markets
      </Link>
    </div>
  );
}

// Trigger from page
import { notFound } from 'next/navigation';

export default async function MarketPage({ params }: Props) {
  const market = await getMarket(params.id);

  if (!market) {
    notFound();
  }

  return <MarketDetails market={market} />;
}
```

---

## Streaming with Suspense

```tsx
// app/dashboard/page.tsx
import { Suspense } from 'react';

export default function DashboardPage() {
  return (
    <div>
      <h1>Dashboard</h1>

      {/* Streams immediately */}
      <Suspense fallback={<MarketsSkeleton />}>
        <MarketsSection />
      </Suspense>

      {/* Streams when ready */}
      <Suspense fallback={<TradesSkeleton />}>
        <RecentTradesSection />
      </Suspense>
    </div>
  );
}

// Each section can be async
async function MarketsSection() {
  const markets = await getMarkets(); // Can take time
  return <MarketList markets={markets} />;
}

async function RecentTradesSection() {
  const trades = await getTrades(); // Slower endpoint
  return <TradeList trades={trades} />;
}
```

---

## Route Handlers (API Routes)

```tsx
// app/api/markets/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';

const querySchema = z.object({
  page: z.coerce.number().min(1).default(1),
  limit: z.coerce.number().min(1).max(100).default(20),
  category: z.string().optional(),
});

export async function GET(request: NextRequest) {
  const searchParams = Object.fromEntries(request.nextUrl.searchParams);

  const parsed = querySchema.safeParse(searchParams);
  if (!parsed.success) {
    return NextResponse.json(
      { error: 'Invalid parameters', details: parsed.error.flatten() },
      { status: 400 }
    );
  }

  const { page, limit, category } = parsed.data;

  const markets = await db.market.findMany({
    where: category ? { category } : undefined,
    skip: (page - 1) * limit,
    take: limit,
  });

  return NextResponse.json({ items: markets, page, limit });
}

// app/api/markets/[id]/route.ts
interface Context {
  params: Promise<{ id: string }>;
}

export async function GET(request: NextRequest, { params }: Context) {
  const { id } = await params;

  const market = await db.market.findUnique({ where: { id } });

  if (!market) {
    return NextResponse.json({ error: 'Not found' }, { status: 404 });
  }

  return NextResponse.json(market);
}
```

---

## Middleware

```tsx
// middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const token = request.cookies.get('auth-token');
  const { pathname } = request.nextUrl;

  // Protect dashboard routes
  if (pathname.startsWith('/dashboard') && !token) {
    return NextResponse.redirect(new URL('/login', request.url));
  }

  // Redirect logged-in users away from auth pages
  if ((pathname === '/login' || pathname === '/register') && token) {
    return NextResponse.redirect(new URL('/dashboard', request.url));
  }

  // Add request ID header for logging
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set('x-request-id', crypto.randomUUID());

  return NextResponse.next({
    request: { headers: requestHeaders },
  });
}

export const config = {
  matcher: ['/dashboard/:path*', '/login', '/register'],
};
```

---

## Metadata and SEO

```tsx
// app/markets/[id]/page.tsx
import type { Metadata } from 'next';

interface Props {
  params: Promise<{ id: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  const market = await getMarket(id);

  return {
    title: market?.name ?? 'Market Not Found',
    description: market?.description ?? 'Market details',
    openGraph: {
      title: market?.name,
      description: market?.description,
      images: [{ url: market?.imageUrl ?? '/default-og.png' }],
    },
  };
}

export default async function MarketPage({ params }: Props) {
  // ...
}
```

---

## Caching and Revalidation

### Fetch Cache Options

```tsx
// Cached indefinitely (default for static)
const data = await fetch(url);

// Revalidate every 60 seconds
const data = await fetch(url, { next: { revalidate: 60 } });

// No cache (always fresh)
const data = await fetch(url, { cache: 'no-store' });

// Force cache even for dynamic
const data = await fetch(url, { cache: 'force-cache' });
```

### On-Demand Revalidation

```tsx
// app/api/revalidate/route.ts
import { revalidatePath, revalidateTag } from 'next/cache';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  const { path, tag, secret } = await request.json();

  if (secret !== process.env.REVALIDATION_SECRET) {
    return NextResponse.json({ error: 'Invalid secret' }, { status: 401 });
  }

  if (path) {
    revalidatePath(path);
  }

  if (tag) {
    revalidateTag(tag);
  }

  return NextResponse.json({ revalidated: true });
}

// Usage with tags
const markets = await fetch(url, { next: { tags: ['markets'] } });
// Later: revalidateTag('markets')
```

---

## Dynamic vs Static Rendering

```tsx
// Force dynamic rendering
export const dynamic = 'force-dynamic';

// Force static rendering
export const dynamic = 'force-static';

// Revalidate interval
export const revalidate = 60;

// Generate static pages for dynamic routes
export async function generateStaticParams() {
  const markets = await getMarkets();
  return markets.map((m) => ({ id: m.id }));
}
```
