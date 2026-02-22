"use client"

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
    MoreVertical,
    TrashIcon
} from "lucide-react"
import Link from "next/link"
import type { PostFieldsFragment } from "@/lib/graphql/__generated__/graphql"

interface PostCardProps {
    post: PostFieldsFragment
    currentUserId?: string
    isAdmin?: boolean
    onDelete?: (postId: string) => void
    isDeleting?: boolean
}

export function PostCard({
    post,
    currentUserId,
    isAdmin = false,
    onDelete,
    isDeleting = false
}: PostCardProps) {
    const canDelete = onDelete && (isAdmin || currentUserId === post.author.id)

    return (
        <Card className="overflow-hidden border-none shadow-md hover:shadow-xl transition-all py-0">
            <CardHeader className="flex flex-row items-center gap-4 space-y-0 p-4">
                <Avatar className="rounded-full ring-1 ring-border">
                    <AvatarImage src={post.author.image ?? undefined} />
                    <AvatarFallback className="bg-muted text-muted-foreground font-medium text-sm">
                        {post.author.name?.[0]?.toUpperCase()}
                    </AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="font-bold text-base truncate">
                                {post.author.name}
                            </p>
                            <p className="text-xs text-muted-foreground flex items-center gap-1">
                                <Link
                                    href={`/spaces/${post.space.slug}`}
                                    className="hover:text-foreground transition-colors hover:underline"
                                >
                                    {post.space.name}
                                </Link>
                                <span>·</span>
                                {new Date(post.createdAt as string).toLocaleDateString(undefined, {
                                    month: 'short',
                                    day: 'numeric',
                                    year: 'numeric'
                                })}
                            </p>
                        </div>
                        {canDelete && (
                            <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                    <Button variant="ghost" size="icon" className="h-9 w-9 text-muted-foreground hover:text-foreground">
                                        <MoreVertical className="h-5 w-5" />
                                        <span className="sr-only">Options</span>
                                    </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end">
                                    <DropdownMenuItem
                                        className="text-destructive focus:text-destructive"
                                        onClick={() => onDelete(post.id)}
                                        disabled={isDeleting}
                                    >
                                        <TrashIcon className="mr-2 h-4 w-4" />
                                        Delete Post
                                    </DropdownMenuItem>
                                </DropdownMenuContent>
                            </DropdownMenu>
                        )}
                    </div>
                </div>
            </CardHeader>

            <CardContent className="px-4 pb-4 pt-1">
                <p className="text-[15px] leading-relaxed whitespace-pre-wrap text-foreground/90">
                    {post.content}
                </p>
                {post.mediaUrls && post.mediaUrls.length > 0 && (
                    <div className="mt-5 grid gap-3">
                        {post.mediaUrls.map((url, i) => (
                            <div key={i} className="relative group overflow-hidden rounded-xl border border-border/50">
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img
                                    src={url}
                                    alt="Post media"
                                    className="object-cover w-full h-[350px] group-hover:scale-[1.02] transition-transform duration-500"
                                />
                            </div>
                        ))}
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
