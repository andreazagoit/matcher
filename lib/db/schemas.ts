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

// 4. Connections
export * from "../models/connections/schema";

// 5. Chat System
export * from "../models/conversations/schema";
export * from "../models/messages/schema";
