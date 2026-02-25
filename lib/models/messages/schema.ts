import { pgTable, uuid, timestamp, text, index } from "drizzle-orm/pg-core";
import { users } from "@/lib/models/users/schema";
import { connections } from "@/lib/models/connections/schema";
import { relations } from "drizzle-orm";

export const messages = pgTable(
    "messages",
    {
        id: uuid("id").primaryKey().defaultRandom(),

        connectionId: uuid("connection_id")
            .notNull()
            .references(() => connections.id, { onDelete: "cascade" }),

        senderId: uuid("sender_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),

        content: text("content").notNull(),

        readAt: timestamp("read_at"),

        createdAt: timestamp("created_at").defaultNow().notNull(),
    },
    (table) => [
        index("messages_connection_idx").on(table.connectionId),
        index("messages_created_at_idx").on(table.createdAt),
    ]
);

export const messagesRelations = relations(messages, ({ one }) => ({
    connection: one(connections, {
        fields: [messages.connectionId],
        references: [connections.id],
    }),
    sender: one(users, {
        fields: [messages.senderId],
        references: [users.id],
    }),
}));

export type Message = typeof messages.$inferSelect;
export type NewMessage = typeof messages.$inferInsert;
