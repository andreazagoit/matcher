import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "./models/users/schema";

const connectionString = process.env.DATABASE_URL!;

const client = postgres(connectionString, { prepare: false });

export const db = drizzle(client, { schema });

export * from "./models/users";
export * from "./models/values/data";
export * from "./models/values/queries";
export * from "./models/interests/data";
export * from "./models/interests/queries";
