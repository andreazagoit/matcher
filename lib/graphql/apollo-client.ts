import { HttpLink } from "@apollo/client";
import {
  registerApolloClient,
  ApolloClient,
  InMemoryCache,
} from "@apollo/client-integration-nextjs";
import { cookies } from "next/headers";

// RSC Apollo Client — uses absolute URL and forwards cookies automatically
export const { getClient, query, PreloadQuery } = registerApolloClient(() => {
  return new ApolloClient({
    cache: new InMemoryCache(),
    link: new HttpLink({
      uri: `${process.env.NEXT_PUBLIC_APP_URL}/api/client/v1/graphql`,
      // Custom fetch that forwards request cookies for server-side auth
      fetch: async (uri, options) => {
        const cookieStore = await cookies();
        const opts = { ...options } as RequestInit;
        opts.headers = {
          ...(opts.headers as Record<string, string>),
          cookie: cookieStore.toString(),
        };
        return fetch(uri, opts);
      },
    }),
  });
});
