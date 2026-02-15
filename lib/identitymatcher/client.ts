/**
 * Server-to-server GraphQL client for Identity Matcher.
 *
 * Uses the API key stored on the OAuth client to authenticate.
 * All user-scoped operations require passing the identitymatcher userId
 * (retrieved from the `account` table via `getExternalUserId`).
 */

import { db } from "@/lib/db/drizzle";
import { account } from "@/lib/models/auth/schema";
import { eq, and } from "drizzle-orm";

const IDENTITYMATCHER_GRAPHQL_URL = `${process.env.OAUTH_SERVER_URL}/api/graphql`;
const IDENTITYMATCHER_API_KEY = process.env.IDENTITYMATCHER_API_KEY!;

// ============================================
// GENERIC GRAPHQL FETCH
// ============================================

interface GraphQLResponse<T = unknown> {
  data?: T;
  errors?: Array<{ message: string; extensions?: Record<string, unknown> }>;
}

/**
 * Execute a GraphQL query/mutation against identitymatcher's API.
 * Authenticates via API key (server-to-server).
 */
export async function idmGraphQL<T = unknown>(
  query: string,
  variables?: Record<string, unknown>,
): Promise<T> {
  const res = await fetch(IDENTITYMATCHER_GRAPHQL_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": IDENTITYMATCHER_API_KEY,
    },
    body: JSON.stringify({ query, variables }),
  });

  if (!res.ok) {
    throw new Error(
      `identitymatcher GraphQL error: ${res.status} ${res.statusText}`,
    );
  }

  const json: GraphQLResponse<T> = await res.json();

  if (json.errors?.length) {
    throw new Error(json.errors[0].message);
  }

  if (!json.data) {
    throw new Error("No data returned from identitymatcher");
  }

  return json.data;
}

// ============================================
// USER ID MAPPING
// ============================================

/**
 * Get the identitymatcher (external) user ID from the local matcher user ID.
 *
 * better-auth stores the provider's user ID in `account.accountId`
 * with `providerId = "identitymatcher"`.
 */
export async function getExternalUserId(
  localUserId: string,
): Promise<string | null> {
  const result = await db.query.account.findFirst({
    where: and(
      eq(account.userId, localUserId),
      eq(account.providerId, "identitymatcher"),
    ),
    columns: { accountId: true },
  });

  return result?.accountId ?? null;
}

/**
 * Get the external user ID, throwing if not found.
 */
export async function requireExternalUserId(
  localUserId: string,
): Promise<string> {
  const externalId = await getExternalUserId(localUserId);
  if (!externalId) {
    throw new Error(
      "Identity Matcher account not linked. Please sign in again.",
    );
  }
  return externalId;
}
