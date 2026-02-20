"use client"

import { useEffect, useState } from "react"
import {
    Loader2Icon
} from "lucide-react"
import { useSession } from "@/lib/auth-client"
import { PostCard } from "@/components/feed/post-card"
import { useQuery, useMutation } from "@apollo/client/react"
import { GET_SPACE_FEED, DELETE_POST } from "@/lib/models/posts/gql"
import type {
    GetSpaceFeedQuery,
    GetSpaceFeedQueryVariables,
    DeletePostMutation,
    DeletePostMutationVariables
} from "@/lib/graphql/__generated__/graphql"

interface PostListProps {
    spaceId: string
    isAdmin: boolean
    refreshTrigger: number // Simple way to trigger refresh from parent
}

export function PostList({ spaceId, isAdmin, refreshTrigger }: PostListProps) {
    const { data: session } = useSession()
    const [deletingId, setDeletingId] = useState<string | null>(null)

    const { data, loading, refetch } = useQuery<GetSpaceFeedQuery, GetSpaceFeedQueryVariables>(GET_SPACE_FEED, {
        variables: { spaceId },
        skip: !spaceId,
        fetchPolicy: "network-only" // Ensure freshness on mount or when key changes, though cache-and-network might be better for cache
    });

    const [deletePost] = useMutation<DeletePostMutation, DeletePostMutationVariables>(DELETE_POST);

    useEffect(() => {
        if (refreshTrigger > 0) {
            refetch();
        }
    }, [refreshTrigger, refetch]);

    const handleDelete = async (postId: string) => {
        if (!confirm("Are you sure you want to delete this post?")) return

        setDeletingId(postId)
        try {
            await deletePost({
                variables: { postId }
            });

            await refetch();
        } catch (error) {
            console.error("Failed to delete post:", error)
            alert("Failed to delete post")
        } finally {
            setDeletingId(null)
        }
    }

    const posts = data?.space?.feed || [];

    if (loading && posts.length === 0) {
        return (
            <div className="flex justify-center py-10">
                <Loader2Icon className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
        )
    }

    if (posts.length === 0) {
        return (
            <div className="text-center py-10 text-muted-foreground">
                No posts yet. Be the first to share something!
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {posts.map((post) => (
                <PostCard
                    key={post.id}
                    post={post}
                    currentUserId={session?.user?.id}
                    isAdmin={isAdmin}
                    onDelete={handleDelete}
                    isDeleting={deletingId === post.id}
                />
            ))}
        </div>
    )
}
