import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";
import { emailOTP } from "better-auth/plugins";
import { expo } from "@better-auth/expo";
import { nextCookies } from "better-auth/next-js";
import { db } from "./db/drizzle";
import * as schema from "./db/schemas";
import { sendOTPEmail } from "./email";

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
        await sendOTPEmail(email, otp, type);
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
