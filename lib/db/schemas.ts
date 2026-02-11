// Central export for all Drizzle schemas
// Importance notice: Order is critical to avoid circular dependency issues.

// 1. Users (base, no dependencies)
export * from "../models/users/schema";

// 2. Assessments (depends on users) - sessions with JSONB response data
export * from "../models/assessments/schema";

// 3. Profiles (depends on users) - standard profiles with vector embeddings
export * from "../models/profiles/schema";

// 4. Spaces & Community (ex Apps)
export * from "../models/spaces/schema";
export * from "../models/tiers/schema";
export * from "../models/members/schema";
export * from "../models/posts/schema";

// 5. OAuth 2.0 (depends on users and spaces)
export * from "../models/authorization-codes/schema";
export * from "../models/tokens/schema";

// 6. Connections & Matches
export * from "../models/connections/schema";
export * from "../models/matches/schema";

// 7. Chat System
export * from "../models/conversations/schema";
export * from "../models/messages/schema";
