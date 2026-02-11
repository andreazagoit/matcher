import { db } from "@/lib/db/drizzle";
import { membershipTiers } from "@/lib/models/tiers/schema";
import { eq, and } from "drizzle-orm";

interface ResolverContext {
    user?: { id: string } | null;
}

export const tierResolvers = {
    Space: {
        tiers: async (parent: { id: string }) => {
            return db
                .select()
                .from(membershipTiers)
                .where(
                    and(
                        eq(membershipTiers.spaceId, parent.id),
                        eq(membershipTiers.isActive, true)
                    )
                );
        },
    },
    Member: {
        tier: async (parent: { tierId?: string }) => {
            if (!parent.tierId) return null;
            const [tier] = await db
                .select()
                .from(membershipTiers)
                .where(eq(membershipTiers.id, parent.tierId));
            return tier;
        },
    },
    Mutation: {
        createTier: async (_: unknown, { spaceId, input }: {
            spaceId: string,
            input: {
                name: string;
                description?: string;
                price: number;
                interval: "month" | "year" | "one_time";
            }
        }, context: ResolverContext) => {
            if (!context.user) throw new Error("Unauthorized");

            const [newTier] = await db.insert(membershipTiers).values({
                spaceId: spaceId,
                name: input.name,
                description: input.description,
                price: input.price,
                interval: input.interval,
                isActive: true,
            }).returning();
            return newTier;
        },
        updateTier: async (_: unknown, { id, input }: {
            id: string,
            input: {
                name?: string;
                description?: string;
                price?: number;
                interval?: "month" | "year" | "one_time";
                isActive?: boolean;
            }
        }) => {
            const [updated] = await db.update(membershipTiers)
                .set(input)
                .where(eq(membershipTiers.id, id))
                .returning();
            return updated;
        },
        archiveTier: async (_: unknown, { id }: { id: string }) => {
            const [updated] = await db.update(membershipTiers)
                .set({ isActive: false })
                .where(eq(membershipTiers.id, id))
                .returning();
            return !!updated;
        }
    },
};
