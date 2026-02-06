"use client"

import { useEffect, useState, useCallback } from "react"
import { graphql } from "@/lib/graphql/client"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
    HeartIcon,
    MessageCircleIcon,
    MoreVerticalIcon,
    TrashIcon,
    Loader2Icon
} from "lucide-react"
import { useSession } from "next-auth/react"
import { PostCard, type Post } from "@/components/feed/post-card"



interface PostListProps {
    spaceId: string
    isAdmin: boolean
    refreshTrigger: number // Simple way to trigger refresh from parent
}

export function PostList({ spaceId, isAdmin, refreshTrigger }: PostListProps) {
    const { data: session } = useSession()
    const [posts, setPosts] = useState<Post[]>([])
    const [loading, setLoading] = useState(true)
    const [deletingId, setDeletingId] = useState<string | null>(null)

    const fetchPosts = useCallback(async () => {
        setLoading(true)
        try {
            const data = await graphql<{ space: { feed: Post[] } }>(`
                query GetSpaceFeed($spaceId: ID!) {
                    space(id: $spaceId) {
                        feed(limit: 50) {
                            id
                            content
                            createdAt
                            author {
                                id
                                firstName
                                lastName
                            }
                            likesCount
                            commentsCount
                        }
                    }
                }
            `, { spaceId })

            if (data.space?.feed) {
                setPosts(data.space.feed)
            }
        } catch (error) {
            console.error("Failed to fetch posts:", error)
        } finally {
            setLoading(false)
        }
    }, [spaceId])

    useEffect(() => {
        fetchPosts()
    }, [fetchPosts, refreshTrigger])

    const handleDelete = async (postId: string) => {
        if (!confirm("Are you sure you want to delete this post?")) return

        setDeletingId(postId)
        try {
            await graphql(`
                mutation DeletePost($postId: ID!) {
                    deletePost(postId: $postId)
                }
            `, { postId })

            setPosts(posts.filter(p => p.id !== postId))
        } catch (error) {
            console.error("Failed to delete post:", error)
            alert("Failed to delete post")
        } finally {
            setDeletingId(null)
        }
    }

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
