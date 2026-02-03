// Central export for all Drizzle schemas
// Ordine importante per evitare circular dependencies

// 1. Users (base, no dependencies)
export * from "../models/users/schema";

// 2. Tests (dipende da users) - sessions con answers JSONB
export * from "../models/tests/schema";

// 3. Profiles (dipende da users) - profilo standard con 4 embeddings
export * from "../models/profiles/schema";

