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
    },
  },

  emailAndPassword: {
    enabled: true,
    requireEmailVerification: true,
  },

  emailVerification: {
    autoSignInAfterVerification: true,
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
        if (process.env.NODE_ENV === "development") {
          console.log(`\n\n=========================================`);
          console.log(`[OTP BYPASS] 🚀 DEVELOPMENT MODE`);
          console.log(`Login OTP for ${email} is: ${otp}`);
          console.log(`=========================================\n\n`);
          // We can skip actual email sending in dev to save resend limits
          return;
        }

        try {
          await sendOTPEmail(email, otp, type);
          console.log(`[OTP] Email sent to ${email} (type: ${type})`);
        } catch (err) {
          console.error(`[OTP] Failed to send email to ${email}:`, err);
          throw err;
        }
      },
      generateOTP: () => {
        if (process.env.NODE_ENV === "development") {
          return "000000";
        }
        // Fallback to default random generation if not in dev
        // This is safe because better-auth uses a secure random generator internally
        // when generateOTP is strictly NOT provided, but since we provide it we must 
        // fallback to standard JS crypto
        const array = new Uint32Array(6);
        crypto.getRandomValues(array);
        return Array.from(array, (num) => (num % 10).toString()).join("");
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
