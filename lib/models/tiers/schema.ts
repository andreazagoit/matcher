import {
    pgTable,
    uuid,
    text,
    integer,
    boolean,
    pgEnum,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { spaces } from "@/lib/models/spaces/schema";
import { members } from "@/lib/models/members/schema";

/**
 * Membership Tiers Schema
 * 
 * Defines the subscription levels available in a Space.
 */

export const tierIntervalEnum = pgEnum("tier_interval", ["month", "year", "one_time"]);

export const membershipTiers = pgTable(
    "membership_tiers",
    {
        id: uuid("id").primaryKey().defaultRandom(),
        spaceId: uuid("space_id")
            .notNull()
            .references(() => spaces.id, { onDelete: "cascade" }),

        name: text("name").notNull(), // e.g., "Silver", "Gold"
        description: text("description"), // e.g., "Access to exclusive content"

        // Price in cents (integer to avoid floating point issues)
        // 1000 = 10.00
        price: integer("price").default(0).notNull(),
        currency: text("currency").default("EUR").notNull(),

        interval: tierIntervalEnum("interval").default("month").notNull(),

        isActive: boolean("is_active").default(true).notNull(),
    }
);

export const membershipTiersRelations = relations(membershipTiers, ({ one, many }) => ({
    space: one(spaces, {
        fields: [membershipTiers.spaceId],
        references: [spaces.id],
    }),
    members: many(members),
}));

export type MembershipTier = typeof membershipTiers.$inferSelect;
export type NewMembershipTier = typeof membershipTiers.$inferInsert;
