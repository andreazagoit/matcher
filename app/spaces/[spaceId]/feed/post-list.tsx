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

interface Author {
    id: string
    firstName: string
    lastName: string
}

interface Post {
    id: string
    content: string
    createdAt: string
    author: Author
    likesCount: number
    commentsCount: number
}

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
        <div className="space-y-4">
            {posts.map((post) => (
                <Card key={post.id}>
                    <CardHeader className="flex flex-row items-start gap-4 space-y-0 pb-2">
                        <Avatar>
                            <AvatarFallback>
                                {post.author.firstName[0]}{post.author.lastName[0]}
                            </AvatarFallback>
                        </Avatar>
                        <div className="flex-1">
                            <div className="flex items-center justify-between">
                                <p className="text-sm font-medium leading-none">
                                    {post.author.firstName} {post.author.lastName}
                                </p>
                                <span className="text-xs text-muted-foreground bg-transparent">
                                    {new Date(post.createdAt).toLocaleDateString()}
                                    {/* Using JS date since I don't want to rely on external lib if not present */}
                                </span>
                            </div>
                        </div>
                        {(isAdmin || session?.user?.id === post.author.id) && (
                            <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                    <Button variant="ghost" size="icon" className="-mr-2 h-8 w-8">
                                        <MoreVerticalIcon className="h-4 w-4" />
                                        <span className="sr-only">More</span>
                                    </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end">
                                    <DropdownMenuItem
                                        className="text-destructive focus:text-destructive"
                                        onClick={() => handleDelete(post.id)}
                                        disabled={deletingId === post.id}
                                    >
                                        <TrashIcon className="mr-2 h-4 w-4" />
                                        Delete
                                    </DropdownMenuItem>
                                </DropdownMenuContent>
                            </DropdownMenu>
                        )}
                    </CardHeader>
                    <CardContent className="pb-2">
                        <p className="text-sm leading-relaxed whitespace-pre-wrap">{post.content}</p>
                    </CardContent>
                    <CardFooter className="pt-2 pb-4">
                        <div className="flex items-center gap-4 text-muted-foreground">
                            <Button variant="ghost" size="sm" className="h-8 px-2 text-xs gap-1">
                                <HeartIcon className="h-3 w-3" />
                                {post.likesCount}
                            </Button>
                            <Button variant="ghost" size="sm" className="h-8 px-2 text-xs gap-1">
                                <MessageCircleIcon className="h-3 w-3" />
                                {post.commentsCount}
                            </Button>
                        </div>
                    </CardFooter>
                </Card>
            ))}
        </div>
    )
}
