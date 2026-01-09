import { useLiveQuery } from "@tanstack/react-db";
import { getSessionsDbSync } from "../data/sessionsDb";
import type { Session } from "../types/schema";

/**
 * Hook to get all sessions from the StreamDB.
 * Returns reactive data that updates when sessions change.
 *
 * NOTE: This must only be called after the root loader has run,
 * which initializes the db via getSessionsDb().
 */
export function useSessions() {
  const db = getSessionsDbSync();

  const query = useLiveQuery(
    (q) => q.from({ sessions: db.collections.sessions }),
    [db]
  );

  // Transform to array of sessions
  // The query.data is a Map where values are the session objects directly
  const allSessions: Session[] = query?.data
    ? Array.from(query.data.values())
    : [];

  // Filter to only sessions with status files (hook system installed)
  // This hides old sessions from before the hook system was added
  const sessions = allSessions.filter((session) => session.fileStatus !== null);

  return {
    sessions,
    isLoading: query?.isLoading ?? false,
  };
}
