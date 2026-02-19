import { pgTable, uuid, timestamp, text, index } from "drizzle-orm/pg-core";
import { users } from "@/lib/models/users/schema";
import { relations } from "drizzle-orm";
import { messages } from "../messages/schema";

export const conversationStatusEnum = ["request", "active", "declined"] as const;
export const conversationSourceEnum = ["discovery", "event", "space"] as const;

export const conversations = pgTable(
  "conversations",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    initiatorId: uuid("initiator_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),

    recipientId: uuid("recipient_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),

    status: text("status", { enum: conversationStatusEnum })
      .default("request")
      .notNull(),

    source: text("source", { enum: conversationSourceEnum })
      .default("discovery")
      .notNull(),

    lastMessageAt: timestamp("last_message_at").defaultNow(),

    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    index("conversations_initiator_idx").on(table.initiatorId),
    index("conversations_recipient_idx").on(table.recipientId),
    index("conversations_status_idx").on(table.status),
  ],
);

export const conversationsRelations = relations(conversations, ({ one, many }) => ({
  initiator: one(users, {
    fields: [conversations.initiatorId],
    references: [users.id],
    relationName: "conversations_initiated",
  }),
  recipient: one(users, {
    fields: [conversations.recipientId],
    references: [users.id],
    relationName: "conversations_received",
  }),
  messages: many(messages),
}));

export type Conversation = typeof conversations.$inferSelect;
export type NewConversation = typeof conversations.$inferInsert;
