import { db } from "@/lib/db/drizzle";
import { eq } from "drizzle-orm";
import { spaces, type Space } from "./schema";
import { members } from "@/lib/models/members/schema";
import { createSpace, updateSpace, deleteSpace, getSpaceRecommendedEvents } from "./operations";
import { GraphQLError } from "graphql";
import type { GraphQLContext } from "@/lib/graphql/context";

interface CreateSpaceInput {
    name: string;
    slug?: string;
    description?: string;
    visibility?: "public" | "private" | "hidden";
    joinPolicy?: "open" | "apply" | "invite_only";
    image?: string;
    categories?: string[];
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
                ownerId: auth.user.id,
            });

            return result.space;
        },

        updateSpace: async (_: unknown, { id, input }: { id: string; input: UpdateSpaceInput }, { auth }: GraphQLContext) => {
            if (!auth.user) throw new GraphQLError("Unauthorized");

            const space = await db.query.spaces.findFirst({
                where: eq(spaces.id, id),
                columns: { ownerId: true },
            });
            if (!space) throw new GraphQLError("Space not found");
            if (space.ownerId !== auth.user.id) {
                throw new GraphQLError("Forbidden: Only the owner can update this space");
            }

            return updateSpace(id, input);
        },

        deleteSpace: async (_: unknown, { id }: { id: string }, { auth }: GraphQLContext) => {
            if (!auth.user) throw new GraphQLError("Unauthorized");

            const space = await db.query.spaces.findFirst({
                where: eq(spaces.id, id),
                columns: { ownerId: true },
            });
            if (!space) throw new GraphQLError("Space not found");
            if (space.ownerId !== auth.user.id) {
                throw new GraphQLError("Forbidden: Only the owner can delete this space");
            }

            return deleteSpace(id);
        },
    },

    Space: {
        membersCount: async (parent: Space, _: unknown, { loaders }: GraphQLContext) => {
            return loaders.membersCountLoader.load(parent.id);
        },

        events: async (
            parent: Space,
            { limit = 50, offset = 0 }: { limit?: number; offset?: number },
            context: GraphQLContext,
        ) => {
            const { getSpaceEvents } = await import("@/lib/models/events/operations");
            return getSpaceEvents(parent.id, context.auth.user?.id, limit, offset);
        },

        recommendedEvents: async (
            parent: Space,
            { limit = 6 }: { limit?: number },
        ) => {
            return getSpaceRecommendedEvents(parent.id, limit);
        },
    },
};
