import { db } from "@/lib/db/drizzle";
import { eq, and } from "drizzle-orm";
import { members, type Member } from "./schema";
import { spaces, type Space } from "@/lib/models/spaces/schema";
import { users } from "@/lib/models/users/schema";
import { membershipTiers } from "@/lib/models/tiers/schema";
import { GraphQLError } from "graphql";

interface ResolverContext {
    auth?: {
        user: {
            id: string;
        };
    };
}

export const memberResolvers = {
    Member: {
        user: async (parent: Member) => {
            return db.query.users.findFirst({
                where: eq(users.id, parent.userId),
            });
        },
        joinedAt: (parent: Member) => parent.joinedAt.toISOString(),
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
            if (!context.auth?.user) return null;
            return db.query.members.findFirst({
                where: and(
                    eq(members.spaceId, parent.id),
                    eq(members.userId, context.auth.user.id)
                ),
            });
        },
    },

    Mutation: {
        joinSpace: async (_: unknown, { spaceId, tierId }: { spaceId: string, tierId?: string }, context: ResolverContext) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            const space = await db.query.spaces.findFirst({ where: eq(spaces.id, spaceId) });
            if (!space) throw new GraphQLError("Space not found");

            const existing = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, context.auth.user.id)),
            });

            if (existing) throw new GraphQLError("Already a member");

            let status: "active" | "pending" | "waiting_payment" | "suspended" = "active"; // Default for public/free

            // 1. Check Space Logic
            if (space.joinPolicy === "apply") {
                status = "pending";
            }

            // 2. Check Tier Logic
            if (tierId) {
                const tier = await db.query.membershipTiers.findFirst({
                    where: eq(membershipTiers.id, tierId as string) // Ensure proper type assertion or check
                });

                if (tier && tier.price > 0) {
                    status = "waiting_payment";
                }
            }

            // TODO: If space.type is tiered but no tierId provided, should we enforce default tier?

            const [newMember] = await db.insert(members).values({
                spaceId,
                userId: context.auth.user.id,
                tierId: tierId || null,
                role: "member",
                status,
            }).returning();

            return newMember;
        },

        leaveSpace: async (_: unknown, { spaceId }: { spaceId: string }, context: ResolverContext) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            const result = await db.delete(members)
                .where(and(
                    eq(members.spaceId, spaceId),
                    eq(members.userId, context.auth.user.id)
                ))
                .returning();

            return result.length > 0;
        },

        updateMemberRole: async (
            _: unknown,
            { spaceId, userId, role }: { spaceId: string, userId: string, role: "admin" | "member" },
            context: ResolverContext
        ) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            const requester = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, context.auth.user.id)),
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
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            const requester = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, context.auth.user.id)),
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
