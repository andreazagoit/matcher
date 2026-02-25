import { pgTable, uuid, timestamp, text, index, uniqueIndex } from "drizzle-orm/pg-core";
import { users } from "@/lib/models/users/schema";
import { userItems } from "@/lib/models/useritems/schema"; // Ensure useritems exists
import { relations } from "drizzle-orm";
import { messages } from "../messages/schema";

export const connectionStatusEnum = ["pending", "accepted", "declined"] as const;

export const connections = pgTable(
    "connections",
    {
        id: uuid("id").primaryKey().defaultRandom(),

        initiatorId: uuid("initiator_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),

        recipientId: uuid("recipient_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),

        targetUserItemId: uuid("target_user_item_id")
            .notNull()
            .references(() => userItems.id, { onDelete: "cascade" }),

        initialMessage: text("initial_message"),

        status: text("status", { enum: connectionStatusEnum })
            .default("pending")
            .notNull(),

        lastMessageAt: timestamp("last_message_at"),

        createdAt: timestamp("created_at").defaultNow().notNull(),
        updatedAt: timestamp("updated_at").defaultNow().notNull(),
    },
    (table) => [
        index("connections_initiator_idx").on(table.initiatorId),
        index("connections_recipient_idx").on(table.recipientId),
        index("connections_status_idx").on(table.status),
        // A user can only have one pending/accepted connection with another user
        uniqueIndex("connections_participants_uidx").on(table.initiatorId, table.recipientId),
    ],
);

export const connectionsRelations = relations(connections, ({ one, many }) => ({
    initiator: one(users, {
        fields: [connections.initiatorId],
        references: [users.id],
        relationName: "connections_initiated",
    }),
    recipient: one(users, {
        fields: [connections.recipientId],
        references: [users.id],
        relationName: "connections_received",
    }),
    targetUserItem: one(userItems, {
        fields: [connections.targetUserItemId],
        references: [userItems.id],
    }),
    messages: many(messages),
}));

export type Connection = typeof connections.$inferSelect;
export type NewConnection = typeof connections.$inferInsert;
