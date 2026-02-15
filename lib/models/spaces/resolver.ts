import { db } from "@/lib/db/drizzle";
import { eq, and } from "drizzle-orm";
import { spaces, type Space } from "./schema";
import { members } from "@/lib/models/members/schema";
import { createSpace, updateSpace, deleteSpace } from "./operations";
import { GraphQLError } from "graphql";
import type { GraphQLContext } from "@/lib/graphql/context";

interface CreateSpaceInput {
    name: string;
    slug?: string;
    description?: string;
    visibility?: "public" | "private" | "hidden";
    joinPolicy?: "open" | "apply" | "invite_only";
    image?: string;
}

type UpdateSpaceInput = Partial<CreateSpaceInput>;

export const spaceResolvers = {
    Query: {
        space: async (_: unknown, { id, slug }: { id?: string; slug?: string }) => {
            if (!id && !slug) throw new GraphQLError("ID or Slug required");

            const result = await db.query.spaces.findFirst({
                where: id ? eq(spaces.id, id) : (slug ? eq(spaces.slug, slug) : undefined),
            });

            return result ?? null;
        },

        spaces: async () => {
            return db.query.spaces.findMany({
                where: eq(spaces.visibility, "public"),
                orderBy: (spaces, { desc }) => [desc(spaces.createdAt)],
            });
        },

        mySpaces: async (_: unknown, __: unknown, { auth }: GraphQLContext) => {
            if (!auth.user) return [];

            const userMemberships = await db.query.members.findMany({
                where: eq(members.userId, auth.user.id),
                with: {
                    space: true
                }
            });

            return userMemberships
                .map(m => m.space)
                .filter((s): s is Space => !!s)
                .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
        },
    },

    Mutation: {
        createSpace: async (_: unknown, { input }: { input: CreateSpaceInput }, { auth }: GraphQLContext) => {
            if (!auth.user) throw new GraphQLError("Unauthorized");

            const result = await createSpace({
                ...input,
                creatorId: auth.user.id,
            });

            return result.space;
        },

        updateSpace: async (_: unknown, { id, input }: { id: string; input: UpdateSpaceInput }, { auth }: GraphQLContext) => {
            if (!auth.user) throw new GraphQLError("Unauthorized");

            const memberRecord = await db.query.members.findFirst({
                where: and(
                    eq(members.spaceId, id),
                    eq(members.userId, auth.user.id),
                    eq(members.role, "admin")
                )
            });

            if (!memberRecord) throw new GraphQLError("Forbidden: Not an admin of this space");

            return updateSpace(id, input);
        },

        deleteSpace: async (_: unknown, { id }: { id: string }, { auth }: GraphQLContext) => {
            if (!auth.user) throw new GraphQLError("Unauthorized");

            const memberRecord = await db.query.members.findFirst({
                where: and(
                    eq(members.spaceId, id),
                    eq(members.userId, auth.user.id),
                    eq(members.role, "admin")
                )
            });

            if (!memberRecord) throw new GraphQLError("Forbidden: Only admins can delete spaces");

            return deleteSpace(id);
        },
    },

    Space: {
        membersCount: async (parent: Space, _: unknown, { loaders }: GraphQLContext) => {
            return loaders.membersCountLoader.load(parent.id);
        },
    },
};
