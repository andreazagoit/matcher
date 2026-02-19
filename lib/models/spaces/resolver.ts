import { db } from "@/lib/db/drizzle";
import { eq, and, sql } from "drizzle-orm";
import { spaces, type Space } from "./schema";
import { members } from "@/lib/models/members/schema";
import { profiles } from "@/lib/models/profiles/schema";
import { getUserInterestTags } from "@/lib/models/interests/operations";
import { createSpace, updateSpace, deleteSpace, getSpacesByTags } from "./operations";
import { GraphQLError } from "graphql";
import type { GraphQLContext } from "@/lib/graphql/context";

interface CreateSpaceInput {
    name: string;
    slug?: string;
    description?: string;
    visibility?: "public" | "private" | "hidden";
    joinPolicy?: "open" | "apply" | "invite_only";
    image?: string;
    tags?: string[];
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

        spacesByTags: async (
            _: unknown,
            { tags, matchAll }: { tags: string[]; matchAll?: boolean },
            { auth }: GraphQLContext,
        ) => {
            if (!auth.user) throw new GraphQLError("Unauthorized");
            return await getSpacesByTags(tags, matchAll ?? false);
        },

        recommendedSpaces: async (
            _: unknown,
            { limit }: { limit?: number },
            { auth }: GraphQLContext,
        ) => {
            if (!auth.user) throw new GraphQLError("Unauthorized");
            const maxResults = limit ?? 10;

            const profile = await db.query.profiles.findFirst({
                where: eq(profiles.userId, auth.user.id),
            });
            const userTags = await getUserInterestTags(auth.user.id);

            // Exclude spaces user is already a member of
            const myMemberships = await db
                .select({ spaceId: members.spaceId })
                .from(members)
                .where(eq(members.userId, auth.user.id));
            const mySpaceIds = myMemberships.map((m) => m.spaceId);

            const excludeMine = (results: Space[]) =>
                results.filter((s) => !mySpaceIds.includes(s.id)).slice(0, maxResults);

            // Strategy 1: OpenAI behavioral embedding
            if (profile?.behaviorEmbedding) {
                const embeddingStr = `[${profile.behaviorEmbedding.join(",")}]`;
                const results = await db
                    .select()
                    .from(spaces)
                    .where(
                        and(
                            eq(spaces.visibility, "public"),
                            eq(spaces.isActive, true),
                            sql`${spaces.embedding} IS NOT NULL`,
                        ),
                    )
                    .orderBy(sql`${spaces.embedding} <=> ${embeddingStr}::vector`)
                    .limit(maxResults + mySpaceIds.length);
                return excludeMine(results);
            }

            // Strategy 2: tag overlap
            if (userTags.length > 0) {
                const tagArray = `{${userTags.join(",")}}`;
                const results = await db
                    .select()
                    .from(spaces)
                    .where(
                        and(
                            eq(spaces.visibility, "public"),
                            eq(spaces.isActive, true),
                            sql`${spaces.tags} && ${tagArray}::text[]`,
                        ),
                    )
                    .limit(maxResults + mySpaceIds.length);
                const filtered = excludeMine(results);
                if (filtered.length > 0) return filtered;
            }

            // Strategy 3: fallback
            const results = await db.query.spaces.findMany({
                where: and(
                    eq(spaces.visibility, "public"),
                    eq(spaces.isActive, true),
                ),
                limit: maxResults + mySpaceIds.length,
            });
            return excludeMine(results);
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
