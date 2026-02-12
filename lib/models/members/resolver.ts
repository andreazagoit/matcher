import { db } from "@/lib/db/drizzle";
import { eq, and } from "drizzle-orm";
import { members, type Member } from "./schema";
import { spaces, type Space } from "@/lib/models/spaces/schema";
import { users } from "@/lib/models/users/schema";
import { membershipTiers } from "@/lib/models/tiers/schema";
import { GraphQLError } from "graphql";

interface ResolverContext {
    user?: { id: string } | null;
}

export const memberResolvers = {
    Member: {
        user: async (parent: Member) => {
            const user = await db.query.users.findFirst({
                where: eq(users.id, parent.userId),
            });
            if (!user) {
                throw new GraphQLError(`User ${parent.userId} not found for member ${parent.id}`, {
                    extensions: { code: "INTERNAL_SERVER_ERROR" }
                });
            }
            return user;
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

        myMembership: async (parent: Space, _: unknown, context: ResolverContext) => {
            if (!context.user) return null;
            return db.query.members.findFirst({
                where: and(
                    eq(members.spaceId, parent.id),
                    eq(members.userId, context.user.id)
                ),
            });
        },
    },

    Mutation: {
        joinSpace: async (_: unknown, { spaceSlug, tierId }: { spaceSlug: string, tierId?: string }, context: ResolverContext) => {
            if (!context.user) throw new GraphQLError("Unauthorized");

            const space = await db.query.spaces.findFirst({ where: eq(spaces.slug, spaceSlug) });
            if (!space) throw new GraphQLError("Space not found");

            const spaceId = space.id;

            const existing = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, context.user.id)),
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
                userId: context.user.id,
                tierId: tierId || null,
                role: "member",
                status,
            }).returning();

            if (!newMember) {
                throw new GraphQLError("Failed to create member record", {
                    extensions: { code: "INTERNAL_SERVER_ERROR" }
                });
            }

            return newMember;
        },

        approveMember: async (
            _: unknown,
            { spaceId, userId }: { spaceId: string, userId: string },
            context: ResolverContext
        ) => {
            if (!context.user) throw new GraphQLError("Unauthorized");

            const requester = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, context.user.id)),
            });

            if (!requester || requester.role !== "admin") {
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

        leaveSpace: async (_: unknown, { spaceId }: { spaceId: string }, context: ResolverContext) => {
            if (!context.user) throw new GraphQLError("Unauthorized");

            const result = await db.delete(members)
                .where(and(
                    eq(members.spaceId, spaceId),
                    eq(members.userId, context.user.id)
                ))
                .returning();

            return result.length > 0;
        },

        updateMemberRole: async (
            _: unknown,
            { spaceId, userId, role }: { spaceId: string, userId: string, role: "admin" | "member" },
            context: ResolverContext
        ) => {
            if (!context.user) throw new GraphQLError("Unauthorized");

            const requester = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, context.user.id)),
            });

            if (!requester || requester.role !== "admin") {
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
            context: ResolverContext
        ) => {
            if (!context.user) throw new GraphQLError("Unauthorized");

            const requester = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, context.user.id)),
            });

            if (!requester || requester.role !== "admin") {
                throw new GraphQLError("Forbidden");
            }

            const result = await db.delete(members)
                .where(and(eq(members.spaceId, spaceId), eq(members.userId, userId)))
                .returning();

            return result.length > 0;
        }
    }
};
