import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";
import { genericOAuth } from "better-auth/plugins";
import { nextCookies } from "better-auth/next-js";
import { db } from "./db/drizzle";
import * as schema from "./db/schemas";

const authServerUrl = process.env.OAUTH_SERVER_URL;
const appUrl = process.env.NEXT_PUBLIC_APP_URL;

/**
 * Matcher — better-auth configuration
 *
 * Authenticates users via IdentityMatcher (external OAuth 2.1 provider).
 * Local session management using better-auth with Drizzle/PostgreSQL.
 */
export const auth = betterAuth({
  baseURL: appUrl,

  database: drizzleAdapter(db, {
    provider: "pg",
    schema,
  }),

  basePath: "/api/auth",

  advanced: {
    database: {
      // Generate UUIDs at app level (auth tables use text PK without DB default)
      generateId: () => crypto.randomUUID(),
    },
  },

  account: {
    accountLinking: {
      enabled: true,
      // Identity Matcher is our trusted auth provider; allow implicit relinking
      // when users recreate their upstream account with the same verified email.
      trustedProviders: ["identitymatcher"],
      updateUserInfoOnLink: true,
    },
  },

  user: {
    modelName: "users",
    additionalFields: {
      givenName: {
        type: "string",
        required: false,
        input: true,
      },
      familyName: {
        type: "string",
        required: false,
        input: true,
      },
      birthdate: {
        type: "string",
        required: false,
        input: true,
      },
      gender: {
        type: "string",
        required: false,
        input: true,
      },
    },
  },

  session: {
    expiresIn: 7 * 24 * 60 * 60, // 7 days
    updateAge: 24 * 60 * 60, // 1 day
  },

  plugins: [
    genericOAuth({
      config: [
        {
          providerId: "identitymatcher",
          discoveryUrl: `${authServerUrl}/api/auth/.well-known/openid-configuration`,
          clientId: process.env.OAUTH_CLIENT_ID!,
          clientSecret: process.env.OAUTH_CLIENT_SECRET!,
          pkce: true,
          prompt: "consent",
          scopes: ["openid", "profile", "email", "location"],
        },
      ],
    }),
    nextCookies(),
  ],
});

export type Session = typeof auth.$Infer.Session;
