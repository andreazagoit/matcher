import { db } from "@/lib/db/drizzle";
import { eq, and } from "drizzle-orm";
import { members, type Member } from "./schema";
import { spaces, type Space } from "@/lib/models/spaces/schema";
import { membershipTiers } from "@/lib/models/tiers/schema";
import { boostInterestsFromTags, getUserInterests } from "@/lib/models/interests/operations";
import { embedUser } from "@/lib/models/embeddings/operations";
import { users } from "@/lib/models/users/schema";
import { GraphQLError } from "graphql";
import type { GraphQLContext } from "@/lib/graphql/context";

export const memberResolvers = {
    Member: {
        user: async (parent: Member, _: unknown, { loaders }: GraphQLContext) => {
            return loaders.userLoader.load(parent.userId);
        },
        tier: async (parent: Member) => {
            if (!parent.tierId) return null;
            return db.query.membershipTiers.findFirst({
                where: eq(membershipTiers.id, parent.tierId),
            });
        },
    },

    Space: {
        members: async (parent: Space, { limit = 20, offset = 0 }: { limit?: number; offset?: number }) => {
            return db.query.members.findMany({
                where: eq(members.spaceId, parent.id),
                limit,
                offset,
                orderBy: (members, { desc }) => [desc(members.joinedAt)],
            });
        },

        myMembership: async (parent: Space, _: unknown, { loaders }: GraphQLContext) => {
            return loaders.myMembershipLoader.load(parent.id);
        },
    },

    Mutation: {
        joinSpace: async (_: unknown, { spaceSlug, tierId }: { spaceSlug: string, tierId?: string }, { auth }: GraphQLContext) => {
            if (!auth.user) throw new GraphQLError("Unauthorized");

            const space = await db.query.spaces.findFirst({ where: eq(spaces.slug, spaceSlug) });
            if (!space) throw new GraphQLError("Space not found");

            const spaceId = space.id;

            const existing = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, auth.user.id)),
            });

            if (existing) throw new GraphQLError("Already a member");

            let status: "active" | "pending" | "waiting_payment" | "suspended" = "active";

            if (space.joinPolicy === "apply") {
                status = "pending";
            }

            if (tierId) {
                const tier = await db.query.membershipTiers.findFirst({
                    where: eq(membershipTiers.id, tierId as string)
                });

                if (tier && tier.price > 0) {
                    status = "waiting_payment";
                }
            }

            const [newMember] = await db.insert(members).values({
                spaceId,
                userId: auth.user.id,
                tierId: tierId || null,
                role: "member",
                status,
            }).returning();

            if (!newMember) {
                throw new GraphQLError("Failed to create member record", {
                    extensions: { code: "INTERNAL_SERVER_ERROR" }
                });
            }

            // Boost interests and regenerate embedding in background
            if (space.tags?.length && status === "active") {
                const userId = auth.user.id;
                (async () => {
                    await boostInterestsFromTags(userId, space.tags!, 0.1);
                    const interests = await getUserInterests(userId);
                    const userData = await db.query.users.findFirst({ where: eq(users.id, userId) });
                    await embedUser(userId, {
                        tags: interests.map((i) => ({ tag: i.tag, weight: i.weight })),
                        birthdate: userData?.birthdate ?? null,
                        gender: userData?.gender ?? null,
                        relationshipIntent: userData?.relationshipIntent ?? null,
                        jobTitle: userData?.jobTitle ?? null,
                        educationLevel: userData?.educationLevel ?? null,
                        smoking: userData?.smoking ?? null,
                        drinking: userData?.drinking ?? null,
                        activityLevel: userData?.activityLevel ?? null,
                        religion: userData?.religion ?? null,
                    });
                })().catch(() => {});
            }

            return newMember;
        },

        approveMember: async (
            _: unknown,
            { spaceId, userId }: { spaceId: string, userId: string },
            { auth }: GraphQLContext
        ) => {
            if (!auth.user) throw new GraphQLError("Unauthorized");

            const requester = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, auth.user.id)),
            });

            if (!requester || !["admin", "owner"].includes(requester.role)) {
                throw new GraphQLError("Forbidden - Admin access required", {
                    extensions: { code: "FORBIDDEN" }
                });
            }

            const [updated] = await db.update(members)
                .set({ status: "active" })
                .where(and(eq(members.spaceId, spaceId), eq(members.userId, userId)))
                .returning();

            if (!updated) {
                throw new GraphQLError("Member request not found", {
                    extensions: { code: "NOT_FOUND" }
                });
            }

            return updated;
        },

        leaveSpace: async (_: unknown, { spaceId }: { spaceId: string }, { auth }: GraphQLContext) => {
            if (!auth.user) throw new GraphQLError("Unauthorized");

            const result = await db.delete(members)
                .where(and(
                    eq(members.spaceId, spaceId),
                    eq(members.userId, auth.user.id)
                ))
                .returning();

            return result.length > 0;
        },

        updateMemberRole: async (
            _: unknown,
            { spaceId, userId, role }: { spaceId: string, userId: string, role: "admin" | "member" },
            { auth }: GraphQLContext
        ) => {
            if (!auth.user) throw new GraphQLError("Unauthorized");

            const requester = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, auth.user.id)),
            });

            if (!requester || !["admin", "owner"].includes(requester.role)) {
                throw new GraphQLError("Forbidden");
            }

            const [updated] = await db.update(members)
                .set({ role })
                .where(and(eq(members.spaceId, spaceId), eq(members.userId, userId)))
                .returning();

            return updated;
        },

        removeMember: async (
            _: unknown,
            { spaceId, userId }: { spaceId: string, userId: string },
            { auth }: GraphQLContext
        ) => {
            if (!auth.user) throw new GraphQLError("Unauthorized");

            const requester = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, auth.user.id)),
            });

            if (!requester || !["admin", "owner"].includes(requester.role)) {
                throw new GraphQLError("Forbidden");
            }

            const result = await db.delete(members)
                .where(and(eq(members.spaceId, spaceId), eq(members.userId, userId)))
                .returning();

            return result.length > 0;
        }
    }
};
