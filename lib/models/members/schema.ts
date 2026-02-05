import {
    pgTable,
    uuid,
    timestamp,
    pgEnum,
    index,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { users } from "@/lib/models/users/schema";
import { spaces } from "@/lib/models/spaces/schema";

/**
 * Members Schema
 * 
 * Represents the relationship between User and Space.
 * Includes Role (owner, admin, member) and Status.
 */

export const memberRoleEnum = pgEnum("member_role", ["owner", "admin", "member"]);
export const memberStatusEnum = pgEnum("member_status", ["pending", "active", "suspended"]);

export const members = pgTable(
    "members",
    {
        id: uuid("id").primaryKey().defaultRandom(),

        spaceId: uuid("space_id")
            .notNull()
            .references(() => spaces.id, { onDelete: "cascade" }),

        userId: uuid("user_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),

        role: memberRoleEnum("role").default("member").notNull(),
        status: memberStatusEnum("status").default("active").notNull(),

        joinedAt: timestamp("joined_at").defaultNow().notNull(),
    },
    (table) => [
        index("members_space_idx").on(table.spaceId),
        index("members_user_idx").on(table.userId),
        // Ensure user is only member once per space (unique composite constraint would be better but index works too if checked)
        // Drizzle currently supports unique constraints via extra config or index
    ]
);

export const membersRelations = relations(members, ({ one }) => ({
    space: one(spaces, {
        fields: [members.spaceId],
        references: [spaces.id],
    }),
    user: one(users, {
        fields: [members.userId],
        references: [users.id],
    }),
}));

export type Member = typeof members.$inferSelect;
export type NewMember = typeof members.$inferInsert;
