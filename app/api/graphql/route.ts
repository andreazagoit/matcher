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

export { handler as GET, handler as POST };

