import {
    pgTable,
    uuid,
    text,
    timestamp,
    integer,
    index,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { users } from "@/lib/models/users/schema";
import { spaces } from "@/lib/models/spaces/schema";

/**
 * Posts Schema
 * 
 * Content created within a Space.
 */

export const posts = pgTable(
    "posts",
    {
        id: uuid("id").primaryKey().defaultRandom(),

        spaceId: uuid("space_id")
            .notNull()
            .references(() => spaces.id, { onDelete: "cascade" }),

        authorId: uuid("author_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),

        content: text("content").notNull(),
        mediaUrls: text("media_urls").array(),

        likesCount: integer("likes_count").default(0),
        commentsCount: integer("comments_count").default(0),

        createdAt: timestamp("created_at").defaultNow().notNull(),
        updatedAt: timestamp("updated_at").defaultNow().notNull(),
    },
    (table) => [
        index("posts_space_idx").on(table.spaceId),
        index("posts_author_idx").on(table.authorId),
        index("posts_created_at_idx").on(table.createdAt),
    ]
);

export const postsRelations = relations(posts, ({ one }) => ({
    space: one(spaces, {
        fields: [posts.spaceId],
        references: [spaces.id],
    }),
    author: one(users, {
        fields: [posts.authorId],
        references: [users.id],
    }),
}));

export type Post = typeof posts.$inferSelect;
export type NewPost = typeof posts.$inferInsert;
