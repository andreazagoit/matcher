import { ApolloServer } from "@apollo/server";
import { startServerAndCreateNextHandler } from "@as-integrations/next";
import { NextRequest } from "next/server";
import { typeDefs, resolvers } from "@/graphql";

const server = new ApolloServer({
  typeDefs,
  resolvers,
});

const handler = startServerAndCreateNextHandler<NextRequest>(server, {
  context: async (req) => ({
    req,
    // In produzione, aggiungere qui l'autenticazione
    // user: await getUser(req),
  }),
});

export async function GET(
  request: NextRequest,
  context: { params: Promise<{}> }
) {
  return handler(request);
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{}> }
) {
  return handler(request);
}

