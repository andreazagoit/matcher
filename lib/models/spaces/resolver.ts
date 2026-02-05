import { db } from "@/lib/db/drizzle";
import { eq, and } from "drizzle-orm";
import { spaces, type Space } from "./schema";
import { members } from "@/lib/models/members/schema";
import { createSpace, updateSpace, deleteSpace } from "./operations";
import { GraphQLError } from "graphql";

interface ResolverContext {
    auth?: {
        user: {
            id: string;
        };
    };
}

interface CreateSpaceInput {
    name: string;
    slug?: string;
    description?: string;
    visibility?: "public" | "private" | "hidden";
    joinPolicy?: "open" | "apply" | "invite_only";
}

type UpdateSpaceInput = Partial<CreateSpaceInput>;

export const spaceResolvers = {
    Query: {
        space: async (_: unknown, { id, slug }: { id?: string; slug?: string }) => {
            if (!id && !slug) throw new GraphQLError("ID or Slug required");

            const result = await db.query.spaces.findFirst({
                where: id ? eq(spaces.id, id) : eq(spaces.slug, slug!),
            });
            return result;
        },

        spaces: async () => {
            return db.query.spaces.findMany({
                where: eq(spaces.visibility, "public"),
                orderBy: (spaces, { desc }) => [desc(spaces.createdAt)],
            });
        },

        mySpaces: async (_: unknown, __: unknown, context: ResolverContext) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            const userMemberships = await db.query.members.findMany({
                where: eq(members.userId, context.auth.user.id),
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
        createSpace: async (_: unknown, { input }: { input: CreateSpaceInput }, context: ResolverContext) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            const result = await createSpace({
                ...input,
                creatorId: context.auth.user.id,
            });

            return result.space;
        },

        updateSpace: async (_: unknown, { id, input }: { id: string; input: UpdateSpaceInput }, context: ResolverContext) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            // Check if user is admin of the space
            const memberRecord = await db.query.members.findFirst({
                where: and(
                    eq(members.spaceId, id),
                    eq(members.userId, context.auth.user.id),
                    eq(members.role, "admin")
                )
            });

            if (!memberRecord) throw new GraphQLError("Forbidden: Not an admin of this space");

            return updateSpace(id, input);
        },

        deleteSpace: async (_: unknown, { id }: { id: string }, context: ResolverContext) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            // Only admins can delete spaces
            const memberRecord = await db.query.members.findFirst({
                where: and(
                    eq(members.spaceId, id),
                    eq(members.userId, context.auth.user.id),
                    eq(members.role, "admin")
                )
            });

            if (!memberRecord) throw new GraphQLError("Forbidden: Only admins can delete spaces");

            return deleteSpace(id);
        },
    },

    Space: {
        createdAt: (parent: Space) => parent.createdAt?.toISOString(),
        membersCount: async (parent: Space) => {
            const result = await db
                .select({ count: db.$count(members, eq(members.spaceId, parent.id)) })
                .from(members);
            return result[0]?.count ?? 0;
        },
    },
};
