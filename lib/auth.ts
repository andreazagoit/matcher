import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";
import { emailOTP } from "better-auth/plugins";
import { expo } from "@better-auth/expo";
import { nextCookies } from "better-auth/next-js";
import { db } from "./db/drizzle";
import * as schema from "./db/schemas";
import { sendOTPEmail } from "./email";
import { userInterests } from "./models/interests/schema";

/**
 * Temporary in-memory bridge: stores initial interests between
 * user.create.before (where we strip them) and user.create.after
 * (where we insert them into userInterests).
 * Keyed by email; entries are cleaned up immediately after use.
 */
const pendingInterests = new Map<string, string[]>();

const appUrl = process.env.NEXT_PUBLIC_APP_URL;

/**
 * Matcher — better-auth configuration
 *
 * Standalone auth with email OTP and email+password registration.
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
      generateId: () => crypto.randomUUID(),
    },
  },

  user: {
    modelName: "users",
    additionalFields: {
      username: {
        type: "string",
        required: false,
        input: true,
        unique: true,
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
      sexualOrientation: {
        type: "string",
        required: false,
        input: true,
      },
      heightCm: {
        type: "number",
        required: false,
        input: true,
      },
      relationshipIntent: {
        type: "string",
        required: false,
        input: true,
      },
      relationshipStyle: {
        type: "string",
        required: false,
        input: true,
      },
      hasChildren: {
        type: "string",
        required: false,
        input: true,
      },
      wantsChildren: {
        type: "string",
        required: false,
        input: true,
      },
      smoking: {
        type: "string",
        required: false,
        input: true,
      },
      drinking: {
        type: "string",
        required: false,
        input: true,
      },
      activityLevel: {
        type: "string",
        required: false,
        input: true,
      },
      religion: {
        type: "string",
        required: false,
        input: true,
      },
      // Transient field: accepted from client but stripped before DB write.
      // Interests are inserted into userInterests via databaseHooks below.
      initialInterests: {
        type: "string[]",
        required: false,
        input: true,
      },
    },
  },

  databaseHooks: {
    user: {
      create: {
        before: async (userData) => {
          const data = userData as Record<string, unknown>;
          const tags = data.initialInterests as string[] | undefined;
          if (tags?.length && data.email) {
            pendingInterests.set(data.email as string, tags);
          }
          // Strip initialInterests so it never reaches the DB
          const { initialInterests: _omit, ...rest } = data;
          return { data: rest as typeof userData };
        },
        after: async (user) => {
          const tags = pendingInterests.get(user.email);
          if (!tags?.length) return;
          pendingInterests.delete(user.email);
          await db
            .insert(userInterests)
            .values(tags.map((tag) => ({ userId: user.id, tag })))
            .onConflictDoNothing();
        },
      },
    },
  },

  emailAndPassword: {
    enabled: true,
    requireEmailVerification: true,
  },

  session: {
    expiresIn: 7 * 24 * 60 * 60, // 7 days
    updateAge: 24 * 60 * 60, // 1 day
    cookieCache: {
      enabled: true,
      strategy: "jwe",
      maxAge: 300, // 5 minutes
      refreshCache: false,
    },
  },

  plugins: [
    emailOTP({
      async sendVerificationOTP({ email, otp, type }) {
        try {
          await sendOTPEmail(email, otp, type);
          console.log(`[OTP] Email sent to ${email} (type: ${type})`);
        } catch (err) {
          console.error(`[OTP] Failed to send email to ${email}:`, err);
          throw err;
        }
      },
      otpLength: 6,
      expiresIn: 300, // 5 minutes
      disableSignUp: true, // signup handled separately (needs profile fields)
      overrideDefaultEmailVerification: true, // use OTP instead of magic link
    }),
    expo(),
    nextCookies(),
  ],
});

export type Session = typeof auth.$Infer.Session;
