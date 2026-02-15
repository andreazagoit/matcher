import { NextRequest } from "next/server";
import type { AuthContext } from "@/lib/auth/utils";
import type { DataLoaders } from "./dataloaders";

export interface GraphQLContext {
  req: NextRequest;
  auth: AuthContext;
  loaders: DataLoaders;
}
