import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "./users/schema";

const connectionString = process.env.DATABASE_URL!;

const client = postgres(connectionString, { prepare: false });

export const db = drizzle(client, { schema });

export * from "./users";
export * from "./constants";
