import { pgTable, uuid, timestamp } from "drizzle-orm/pg-core";
import { users } from "@/lib/models/users/schema";
import { relations } from "drizzle-orm";

export const conversations = pgTable(
    "conversations",
    {
        id: uuid("id").primaryKey().defaultRandom(),

        participant1Id: uuid("participant1_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),

        participant2Id: uuid("participant2_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),

        lastMessageAt: timestamp("last_message_at").defaultNow(),

        createdAt: timestamp("created_at").defaultNow().notNull(),
        updatedAt: timestamp("updated_at").defaultNow().notNull(),
    },
    () => [
        // Ensure unique pair of participants. 
        // We might enforce p1 < p2 in application logic to avoid duplicates (A,B) and (B,A)
        // or just use a unique index if we always sort ids.
    ]
);

export const conversationsRelations = relations(conversations, ({ one, many }) => ({
    participant1: one(users, {
        fields: [conversations.participant1Id],
        references: [users.id],
        relationName: "conversations_as_p1",
    }),
    participant2: one(users, {
        fields: [conversations.participant2Id],
        references: [users.id],
        relationName: "conversations_as_p2",
    }),
    messages: many(messages),
}));

// We need to import messages here to define relation, but circular dependency concern?
// Usually relations are defined in relations file or strictly ordered. 
// Given the pattern in this project, relations are often in the schema file.
// I will define relations after creating messages schema.
import { messages } from "../messages/schema";

export type Conversation = typeof conversations.$inferSelect;
export type NewConversation = typeof conversations.$inferInsert;
