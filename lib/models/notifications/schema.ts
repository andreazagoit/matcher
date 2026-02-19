import {
  pgTable,
  uuid,
  text,
  boolean,
  timestamp,
  index,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { users } from "@/lib/models/users/schema";

export const notificationTypeEnum = [
  "new_match",
  "match_mutual",
  "new_message",
  "space_joined",
  "event_reminder",
  "event_rsvp",
  "generic",
] as const;

export const notifications = pgTable(
  "notifications",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    userId: uuid("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),

    type: text("type", { enum: notificationTypeEnum })
      .notNull()
      .default("generic"),

    text: text("text").notNull(),

    image: text("image"),

    href: text("href"),

    read: boolean("read").notNull().default(false),

    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    index("notifications_user_idx").on(table.userId),
    index("notifications_user_read_idx").on(table.userId, table.read),
    index("notifications_created_at_idx").on(table.createdAt),
  ],
);

export const notificationsRelations = relations(notifications, ({ one }) => ({
  user: one(users, {
    fields: [notifications.userId],
    references: [users.id],
  }),
}));

export type Notification = typeof notifications.$inferSelect;
export type NewNotification = typeof notifications.$inferInsert;
