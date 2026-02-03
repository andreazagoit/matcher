import { ApolloServer } from "@apollo/server";
import { startServerAndCreateNextHandler } from "@as-integrations/next";
import { NextRequest, NextResponse } from "next/server";
import { typeDefs } from "@/lib/graphql/typedefs";
import { resolvers } from "@/lib/graphql/resolvers";
import { getAuthContext, type AuthContext } from "@/lib/auth/middleware";

export interface GraphQLContext {
  req: NextRequest;
  auth: AuthContext;
}

const server = new ApolloServer<GraphQLContext>({
  typeDefs,
  resolvers,
});

const handler = startServerAndCreateNextHandler<NextRequest, GraphQLContext>(server, {
  context: async (req) => {
    const auth = await getAuthContext(req);
    return { req, auth };
  },
});

export async function GET(request: NextRequest) {
  return handler(request) as Promise<NextResponse>;
}

export async function POST(request: NextRequest) {
  return handler(request) as Promise<NextResponse>;
}
