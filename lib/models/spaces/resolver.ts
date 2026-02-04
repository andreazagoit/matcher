import { db } from "@/lib/db/drizzle";
import { eq, or } from "drizzle-orm";
import { spaces } from "./schema";
import { users } from "@/lib/models/users/schema";
import { createSpace, updateSpace, deleteSpace } from "./operations";
import { GraphQLError } from "graphql";

export const spaceResolvers = {
    Query: {
        space: async (_: any, { id, slug }: { id?: string; slug?: string }) => {
            if (!id && !slug) throw new GraphQLError("ID or Slug required");

            const result = await db.query.spaces.findFirst({
                where: id ? eq(spaces.id, id) : eq(spaces.slug, slug!),
            });
            return result;
        },

        spaces: async () => {
            return db.query.spaces.findMany({
                where: eq(spaces.isPublic, true),
                orderBy: (spaces, { desc }) => [desc(spaces.createdAt)],
            });
        },

        mySpaces: async (_: any, __: any, context: { auth: { user: { id: string } } }) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            // Spaces created by me
            return db.query.spaces.findMany({
                where: eq(spaces.ownerId, context.auth.user.id),
                orderBy: (spaces, { desc }) => [desc(spaces.createdAt)],
            });
        },
    },

    Mutation: {
        createSpace: async (_: any, { input }: { input: any }, context: { auth: { user: { id: string } } }) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            const result = await createSpace({
                ...input,
                ownerId: context.auth.user.id,
            });

            return result.space;
        },

        updateSpace: async (_: any, { id, input }: { id: string; input: any }, context: { auth: { user: { id: string } } }) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            const space = await db.query.spaces.findFirst({ where: eq(spaces.id, id) });
            if (!space) throw new GraphQLError("Space not found");
            if (space.ownerId !== context.auth.user.id) throw new GraphQLError("Forbidden");

            return updateSpace(id, input);
        },

        deleteSpace: async (_: any, { id }: { id: string }, context: { auth: { user: { id: string } } }) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            const space = await db.query.spaces.findFirst({ where: eq(spaces.id, id) });
            if (!space) throw new GraphQLError("Space not found");
            if (space.ownerId !== context.auth.user.id) throw new GraphQLError("Forbidden");

            return deleteSpace(id);
        },
    },

    Space: {
        owner: async (parent: any) => {
            const result = await db.query.users.findFirst({
                where: eq(users.id, parent.ownerId),
            });
            return result;
        },

        createdAt: (parent: any) => parent.createdAt.toISOString(),
    },
};
