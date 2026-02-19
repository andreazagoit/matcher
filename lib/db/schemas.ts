// Central export for all Drizzle schemas
// Importance notice: Order is critical to avoid circular dependency issues.

// 1. Users (base, no dependencies)
export * from "../models/users/schema";

// 2. Better-Auth (session, account, verification — depends on users)
export * from "../models/auth/schema";

// 3. Spaces & Community
export * from "../models/spaces/schema";
export * from "../models/tiers/schema";
export * from "../models/members/schema";
export * from "../models/posts/schema";

// 4. Profiles
export * from "../models/profiles/schema";

// 5. Events
export * from "../models/events/schema";

// 6. Interests
export * from "../models/interests/schema";

// 7. Chat System
export * from "../models/conversations/schema";
export * from "../models/messages/schema";

// 8. Impressions (ML training data)
export * from "../models/impressions/schema";

// 9. Embeddings (64-dim vectors for recommendations)
export * from "../models/embeddings/schema";
