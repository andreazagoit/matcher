import DataLoader from "dataloader";
import { db } from "@/lib/db/drizzle";
import { users, type User } from "@/lib/models/users/schema";
import { members, type Member } from "@/lib/models/members/schema";
import { eq, inArray, count, and } from "drizzle-orm";

/**
 * Creates a new set of data loaders for a single request.
 * Loaders must be per-request to avoid cross-user data leakage and stale cache.
 */
export function createDataLoaders(currentUserId?: string | null) {
    return {
        /**
         * Loads users by their IDs in a single batch.
         */
        userLoader: new DataLoader<string, User | null>(async (userIds) => {
            const results = await db.query.users.findMany({
                where: inArray(users.id, Array.from(userIds)),
            });

            // Map results back to the order of userIds
            const userMap = new Map(results.map((u) => [u.id, u]));
            return userIds.map((id) => userMap.get(id) || null);
        }),

        /**
         * Loads member counts for multiple spaces in a single batch.
         */
        membersCountLoader: new DataLoader<string, number>(async (spaceIds) => {
            const results = await db
                .select({
                    spaceId: members.spaceId,
                    count: count(),
                })
                .from(members)
                .where(inArray(members.spaceId, Array.from(spaceIds)))
                .groupBy(members.spaceId);

            const countMap = new Map(results.map((r) => [r.spaceId, r.count]));
            return spaceIds.map((id) => countMap.get(id) || 0);
        }),

        /**
         * Loads the current user's membership for multiple spaces in a single batch.
         */
        myMembershipLoader: new DataLoader<string, Member | null>(async (spaceIds) => {
            if (!currentUserId) {
                return spaceIds.map(() => null);
            }

            const results = await db.query.members.findMany({
                where: and(
                    eq(members.userId, currentUserId),
                    inArray(members.spaceId, Array.from(spaceIds))
                ),
            });

            const membershipMap = new Map(results.map((m) => [m.spaceId, m]));
            return spaceIds.map((id) => membershipMap.get(id) || null);
        }),
    };
}

export type DataLoaders = ReturnType<typeof createDataLoaders>;
