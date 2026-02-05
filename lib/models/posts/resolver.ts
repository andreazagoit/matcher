import { db } from "@/lib/db/drizzle";
import { eq, and } from "drizzle-orm";
import { posts, type Post } from "./schema";
import { members } from "@/lib/models/members/schema";
import { users } from "@/lib/models/users/schema";
import { type Space } from "@/lib/models/spaces/schema";
import { GraphQLError } from "graphql";

interface ResolverContext {
    auth?: {
        user: {
            id: string;
        };
    };
}

export const postResolvers = {
    Query: {
        globalFeed: async (_: unknown, { limit = 20, offset = 0 }: { limit?: number; offset?: number }) => {
            return db.query.posts.findMany({
                limit,
                offset,
                orderBy: (posts, { desc }) => [desc(posts.createdAt)],
            });
        },
    },

    Post: {
        author: async (parent: Post) => {
            return db.query.users.findFirst({
                where: eq(users.id, parent.authorId),
            });
        },
        createdAt: (parent: Post) => parent.createdAt.toISOString(),
    },

    Space: {
        feed: async (parent: Space, { limit = 20, offset = 0 }: { limit?: number; offset?: number }) => {
            return db.query.posts.findMany({
                where: eq(posts.spaceId, parent.id),
                limit,
                offset,
                orderBy: (posts, { desc }) => [desc(posts.createdAt)],
            });
        },
    },

    Mutation: {
        createPost: async (
            _: unknown,
            { spaceId, content, mediaUrls }: { spaceId: string, content: string, mediaUrls?: string[] },
            context: ResolverContext
        ) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            // Verify membership
            const membership = await db.query.members.findFirst({
                where: and(eq(members.spaceId, spaceId), eq(members.userId, context.auth.user.id)),
            });

            if (!membership || membership.status !== "active") {
                throw new GraphQLError("Must be an active member to post");
            }

            const [newPost] = await db.insert(posts).values({
                spaceId,
                authorId: context.auth.user.id,
                content,
                mediaUrls: mediaUrls || [],
            }).returning();

            return newPost;
        },

        deletePost: async (_: unknown, { postId }: { postId: string }, context: ResolverContext) => {
            if (!context.auth?.user) throw new GraphQLError("Unauthorized");

            const post = await db.query.posts.findFirst({ where: eq(posts.id, postId) });
            if (!post) throw new GraphQLError("Post not found");

            if (post.authorId !== context.auth.user.id) {
                // Check if admin
                const membership = await db.query.members.findFirst({
                    where: and(eq(members.spaceId, post.spaceId), eq(members.userId, context.auth.user.id)),
                });

                if (!membership || membership.role !== "admin") {
                    throw new GraphQLError("Forbidden");
                }
            }

            const result = await db.delete(posts).where(eq(posts.id, postId)).returning();
            return result.length > 0;
        },
    }
};
