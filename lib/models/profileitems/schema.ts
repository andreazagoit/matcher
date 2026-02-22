import {
  pgTable,
  uuid,
  text,
  integer,
  timestamp,
  index,
  pgEnum,
  unique,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { users } from "@/lib/models/users/schema";

export const profileItemTypeEnum = pgEnum("profile_item_type", ["photo", "prompt"]);

export const profileItems = pgTable(
  "profile_items",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    userId: uuid("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),

    type: profileItemTypeEnum("type").notNull(),

    /**
     * For type=prompt: the prompt key from the predefined list (e.g. "controversial_opinion").
     * For type=photo: null.
     */
    promptKey: text("prompt_key"),

    /**
     * For type=prompt: the user's text answer.
     * For type=photo: the image URL.
     */
    content: text("content").notNull(),

    /** Position in the profile (0-based). Unique per user. */
    displayOrder: integer("display_order").notNull(),

    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    index("profile_items_user_idx").on(table.userId),
    index("profile_items_order_idx").on(table.userId, table.displayOrder),
    unique("profile_items_user_order_unique").on(table.userId, table.displayOrder),
  ]
);

export const profileItemsRelations = relations(profileItems, ({ one }) => ({
  user: one(users, {
    fields: [profileItems.userId],
    references: [users.id],
  }),
}));

export type ProfileItem = typeof profileItems.$inferSelect;
export type NewProfileItem = typeof profileItems.$inferInsert;
