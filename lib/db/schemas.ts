// Central export for all Drizzle schemas
// Ordine importante per evitare circular dependencies

// 1. Users (base, no dependencies)
export * from "../models/users/schema";

// 2. Assessments (dipende da users) - sessions con answers JSONB
export * from "../models/assessments/schema";

// 3. Profiles (dipende da users) - profilo standard con 4 embeddings
export * from "../models/profiles/schema";

// 4. Spaces & Community (ex Apps)
export * from "../models/spaces/schema";
export * from "../models/tiers/schema";
export * from "../models/members/schema";
export * from "../models/posts/schema";

// 5. OAuth 2.0 (dipende da users e spaces)
export * from "../models/authorization-codes/schema";
export * from "../models/tokens/schema";
