// Central export for all Drizzle schemas
// Importance notice: Order is critical to avoid circular dependency issues.

// 1. Users (base, no dependencies)
export * from "../models/users/schema";

// 2. Better-Auth (session, account, verification — depends on users)
export * from "../models/auth/schema";

// 3. Assessments (depends on users) - sessions with JSONB response data
export * from "../models/assessments/schema";

// 4. Profiles (depends on users) - standard profiles with vector embeddings
export * from "../models/profiles/schema";

// 5. Spaces & Community (ex Apps)
export * from "../models/spaces/schema";
export * from "../models/tiers/schema";
export * from "../models/members/schema";
export * from "../models/posts/schema";

// 6. Connections & Matches
export * from "../models/connections/schema";
export * from "../models/matches/schema";

// 7. Chat System
export * from "../models/conversations/schema";
export * from "../models/messages/schema";
