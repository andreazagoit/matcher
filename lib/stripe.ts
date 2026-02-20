import Stripe from "stripe";

const stripeSecretKey = process.env.STRIPE_SECRET_KEY;

if (!stripeSecretKey) {
  // Helpful startup error instead of a generic SDK exception.
  throw new Error(
    "Missing STRIPE_SECRET_KEY. Add STRIPE_SECRET_KEY=sk_test_... to your .env and restart the dev server.",
  );
}

// Use a single Stripe client instance for all requests.
// API version is managed by the SDK account default.
export const stripeClient = new Stripe(stripeSecretKey);

// Backward-compatible alias used across the existing codebase.
export const stripe = stripeClient;

export const PLATFORM_FEE_PERCENT = 0.1;
