import { pgTable, uuid, timestamp, text, index, primaryKey } from "drizzle-orm/pg-core";
import { users } from "@/lib/models/users/schema";
import { relations } from "drizzle-orm";

export const connections = pgTable(
    "connections",
    {
        requesterId: uuid("requester_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),

        targetId: uuid("target_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),

        status: text("status", { enum: ["interested", "rejected", "matched"] })
            .notNull()
            .default("interested"),

        createdAt: timestamp("created_at").defaultNow().notNull(),
        updatedAt: timestamp("updated_at").defaultNow().notNull(),
    },
    (table) => [
        primaryKey({ columns: [table.requesterId, table.targetId] }),
        index("connections_requester_idx").on(table.requesterId),
        index("connections_target_idx").on(table.targetId),
        index("connections_status_idx").on(table.status),
    ]
);

export const connectionsRelations = relations(connections, ({ one }) => ({
    requester: one(users, {
        fields: [connections.requesterId],
        references: [users.id],
        relationName: "sent_connections",
    }),
    target: one(users, {
        fields: [connections.targetId],
        references: [users.id],
        relationName: "received_connections",
    }),
}));

export type Connection = typeof connections.$inferSelect;
export type NewConnection = typeof connections.$inferInsert;
