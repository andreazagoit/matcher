import { ApolloServer } from "@apollo/server";
import { startServerAndCreateNextHandler } from "@as-integrations/next";
import { NextRequest, NextResponse } from "next/server";
import { typeDefs } from "@/lib/graphql/typedefs";
import { resolvers } from "@/lib/graphql/resolvers";

const server = new ApolloServer({
  typeDefs,
  resolvers,
});

const handler = startServerAndCreateNextHandler<NextRequest>(server, {
  context: async (req) => ({ req }),
});

export async function GET(request: NextRequest) {
  return handler(request) as Promise<NextResponse>;
}

export async function POST(request: NextRequest) {
  return handler(request) as Promise<NextResponse>;
}
