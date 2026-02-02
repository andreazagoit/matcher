import { GraphQLError } from "graphql";
import { eq } from "drizzle-orm";
import { db, users } from "@/db";
import type { User, NewUser } from "@/db/schemas";

export const userResolvers = {
  Query: {
    user: async (_: unknown, { id }: { id: string }) => {
      const result = await db.query.users.findFirst({
        where: eq(users.id, id),
      });
      return result || null;
    },

    users: async () => {
      return await db.query.users.findMany();
    },

    me: async (_: unknown, __: unknown, context: { userId?: string }) => {
      const userId = context.userId;
      if (!userId) return null;

      return await db.query.users.findFirst({
        where: eq(users.id, userId),
      });
    },
  },

  Mutation: {
    createUser: async (
      _: unknown,
      {
        input,
      }: {
        input: Omit<NewUser, "id" | "createdAt" | "updatedAt">;
      }
    ) => {
      const [newUser] = await db
        .insert(users)
        .values({ ...input })
        .returning();

      return newUser;
    },

    updateUser: async (
      _: unknown,
      {
        id,
        input,
      }: {
        id: string;
        input: Partial<Omit<User, "id" | "createdAt" | "updatedAt">>;
      }
    ) => {
      const [updatedUser] = await db
        .update(users)
        .set({ ...input, updatedAt: new Date() })
        .where(eq(users.id, id))
        .returning();

      if (!updatedUser) {
        throw new GraphQLError("User not found", {
          extensions: { code: "NOT_FOUND" },
        });
      }

      return updatedUser;
    },

    deleteUser: async (_: unknown, { id }: { id: string }) => {
      const [deleted] = await db
        .delete(users)
        .where(eq(users.id, id))
        .returning();

      return !!deleted;
    },
  },
};
