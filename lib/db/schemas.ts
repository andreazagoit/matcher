// Central export for all Drizzle schemas
// Ordine importante per evitare circular dependencies

// 1. Users (base, no dependencies)
export * from "../models/users/schema";

// 2. Assessments (dipende da users) - sessions con answers JSONB
export * from "../models/assessments/schema";

// 3. Profiles (dipende da users) - profilo standard con 4 embeddings
export * from "../models/profiles/schema";

// 4. OAuth 2.0 (dipende da users)
export * from "../models/apps/schema";
export * from "../models/authorization-codes/schema";
export * from "../models/tokens/schema";

