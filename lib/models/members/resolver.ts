import { db } from "@/lib/db/drizzle";
import { eq, and } from "drizzle-orm";
import { members } from "./schema";
import { spaces } from "@/lib/models/spaces/schema";
import { users } from "@/lib/models/users/schema";
import { GraphQLError } from "graphql";

export const memberResolvers = {
    Member: {
        user: async (parent: any) => {
            // Drizzle might define this relation, but manual fetch is safe
            return db.query.users.findFirst({
                where: eq(users.id, parent.userId),
            });
        },
        joinedAt: (parent: any) => parent.joinedAt.toISOString(),
    },

    Space: {
        members: async (parent: any, { limit = 20, offset = 0 }) => {
            return db.query.members.findMany({
                where: eq(members.spaceId, parent.id),
                limit,
                offset,
                orderBy: (members, { desc }) => [desc(members.joinedAt)],
            });
        },

        myMembership: async (parent: any, _: any, context: { auth: { user: { id: string } } }) => {
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
        joinSpace: async (_: any, { spaceId }: { spaceId: string }, context: { auth: { user: { id: string } } }) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            // Check if space exists and if it requires approval
            const space = await db.query.spaces.findFirst({ where: eq(spaces.id, spaceId) });
            if (!space) throw new GraphQLError("Space not found");

            // Check if already member
            const existing = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, context.auth.user.id)),
            });

            if (existing) throw new GraphQLError("Already a member");

            const status = space.requiresApproval ? "pending" : "active";

            const [newMember] = await db.insert(members).values({
                spaceId,
                userId: context.auth.user.id,
                role: "member",
                status,
            }).returning();

            return newMember;
        },

        leaveSpace: async (_: any, { spaceId }: { spaceId: string }, context: { auth: { user: { id: string } } }) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            const result = await db.delete(members)
                .where(and(
                    eq(members.spaceId, spaceId),
                    eq(members.userId, context.auth.user.id)
                ))
                .returning();

            return result.length > 0;
        },

        updateMemberRole: async (_: any, { spaceId, userId, role }: { spaceId: string, userId: string, role: "admin" | "member" }, context: { auth: { user: { id: string } } }) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            // Verify requester is owner or admin
            const requester = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, context.auth.user.id)),
            });

            if (!requester || (requester.role !== "owner" && requester.role !== "admin")) {
                throw new GraphQLError("Forbidden");
            }

            const [updated] = await db.update(members)
                .set({ role })
                .where(and(eq(members.spaceId, spaceId), eq(members.userId, userId)))
                .returning();

            return updated;
        },

        removeMember: async (_: any, { spaceId, userId }: { spaceId: string, userId: string }, context: { auth: { user: { id: string } } }) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            // Verify requester is owner or admin
            const requester = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, context.auth.user.id)),
            });

            if (!requester || (requester.role !== "owner" && requester.role !== "admin")) {
                throw new GraphQLError("Forbidden");
            }

            const result = await db.delete(members)
                .where(and(eq(members.spaceId, spaceId), eq(members.userId, userId)))
                .returning();

            return result.length > 0;
        }
    }
};
