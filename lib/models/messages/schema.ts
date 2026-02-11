import { pgTable, uuid, timestamp, text, boolean, index } from "drizzle-orm/pg-core";
import { users } from "@/lib/models/users/schema";
import { conversations } from "@/lib/models/conversations/schema";
import { relations } from "drizzle-orm";

export const messages = pgTable(
    "messages",
    {
        id: uuid("id").primaryKey().defaultRandom(),

        conversationId: uuid("conversation_id")
            .notNull()
            .references(() => conversations.id, { onDelete: "cascade" }),

        senderId: uuid("sender_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),

        content: text("content").notNull(),

        readAt: timestamp("read_at"),

        createdAt: timestamp("created_at").defaultNow().notNull(),
    },
    (table) => [
        index("messages_conversation_idx").on(table.conversationId),
        index("messages_created_at_idx").on(table.createdAt),
    ]
);

export const messagesRelations = relations(messages, ({ one }) => ({
    conversation: one(conversations, {
        fields: [messages.conversationId],
        references: [conversations.id],
    }),
    sender: one(users, {
        fields: [messages.senderId],
        references: [users.id],
    }),
}));

export type Message = typeof messages.$inferSelect;
export type NewMessage = typeof messages.$inferInsert;
