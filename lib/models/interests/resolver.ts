import { GraphQLError } from "graphql";
import type { GraphQLContext } from "@/lib/graphql/context";
import { getUserInterests, setDeclaredInterests } from "./operations";
import { isValidTag } from "@/lib/models/tags/data";
import { embedUser } from "@/lib/models/embeddings/operations";
import { db } from "@/lib/db/drizzle";
import { users } from "@/lib/models/users/schema";
import { eq } from "drizzle-orm";

function requireAuth(context: GraphQLContext) {
  if (!context.auth.user) {
    throw new GraphQLError("Authentication required", {
      extensions: { code: "UNAUTHENTICATED" },
    });
  }
  return context.auth.user;
}

export const interestResolvers = {
  Query: {
    myInterests: async (
      _: unknown,
      __: unknown,
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return getUserInterests(user.id);
    },
  },

  Mutation: {
    updateMyInterests: async (
      _: unknown,
      { tags }: { tags: string[] },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);

      const invalidTags = tags.filter((t) => !isValidTag(t));
      if (invalidTags.length > 0) {
        throw new GraphQLError(`Invalid tags: ${invalidTags.join(", ")}`);
      }

      const result = await setDeclaredInterests(user.id, tags);

      // Regenerate embedding in background with updated interests
      (async () => {
        const userData = await db.query.users.findFirst({ where: eq(users.id, user.id) });
        await embedUser(user.id, {
          tags,
          birthdate: userData?.birthdate ?? null,
        });
      })().catch(() => {});

      return result;
    },
  },
};
