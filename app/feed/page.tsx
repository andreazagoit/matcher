"use client";

import { useEffect, useState } from "react";
import { graphql } from "@/lib/graphql/client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { Loader2, Heart, MessageCircle, Share2, MoreVertical } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Author {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
}

interface Post {
    id: string;
    content: string;
    mediaUrls: string[];
    likesCount: number;
    commentsCount: number;
    createdAt: string;
    author: Author;
}

export default function FeedPage() {
    const [posts, setPosts] = useState<Post[]>([]);
    const [loading, setLoading] = useState(true);

    const fetchFeed = async () => {
        try {
            const data = await graphql<{ globalFeed: Post[] }>(`
        query GetGlobalFeed {
          globalFeed {
            id
            content
            mediaUrls
            likesCount
            commentsCount
            createdAt
            author {
              id
              firstName
              lastName
              email
            }
          }
        }
      `);
            setPosts(data.globalFeed || []);
        } catch (error) {
            console.error("Failed to fetch feed:", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchFeed();
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[60vh]">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="max-w-2xl mx-auto py-6 space-y-6">
            <div className="flex items-center justify-between px-4 sm:px-0">
                <h1 className="text-3xl font-bold">Feed</h1>
                <Button variant="outline" size="sm" onClick={fetchFeed}>Refresh</Button>
            </div>

            <Separator />

            {(!posts || posts.length === 0) ? (
                <div className="text-center py-20 bg-muted/20 rounded-lg border-2 border-dashed">
                    <p className="text-muted-foreground">No posts to show yet. Be the first to share something!</p>
                </div>
            ) : (
                <div className="space-y-6">
                    {posts.map((post) => (
                        <Card key={post.id} className="overflow-hidden">
                            <CardHeader className="flex flex-row items-center gap-4 space-y-0 p-4">
                                <Avatar className="h-10 w-10">
                                    <AvatarFallback>
                                        {post.author.firstName[0]}{post.author.lastName[0]}
                                    </AvatarFallback>
                                </Avatar>
                                <div className="flex-1">
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <p className="font-semibold text-sm">
                                                {post.author.firstName} {post.author.lastName}
                                            </p>
                                            <p className="text-xs text-muted-foreground">
                                                {new Date(post.createdAt).toLocaleDateString()}
                                            </p>
                                        </div>
                                        <Button variant="ghost" size="icon" className="h-8 w-8">
                                            <MoreVertical className="h-4 w-4" />
                                        </Button>
                                    </div>
                                </div>
                            </CardHeader>

                            <CardContent className="px-4 pb-4">
                                <p className="text-sm whitespace-pre-wrap">{post.content}</p>
                                {post.mediaUrls && post.mediaUrls.length > 0 && (
                                    <div className="mt-4 grid gap-2">
                                        {post.mediaUrls.map((url, i) => (
                                            <img
                                                key={i}
                                                src={url}
                                                alt="Post media"
                                                className="rounded-lg object-cover w-full max-h-[400px]"
                                            />
                                        ))}
                                    </div>
                                )}
                            </CardContent>

                            <Separator />

                            <CardFooter className="flex justify-between p-2">
                                <div className="flex gap-1">
                                    <Button variant="ghost" size="sm" className="h-9 px-3 gap-2">
                                        <Heart className="h-4 w-4" />
                                        <span className="text-xs">{post.likesCount || 0}</span>
                                    </Button>
                                    <Button variant="ghost" size="sm" className="h-9 px-3 gap-2">
                                        <MessageCircle className="h-4 w-4" />
                                        <span className="text-xs">{post.commentsCount || 0}</span>
                                    </Button>
                                </div>
                                <Button variant="ghost" size="sm" className="h-9 px-3 gap-2">
                                    <Share2 className="h-4 w-4" />
                                    <span className="text-xs text-sr-only">Share</span>
                                </Button>
                            </CardFooter>
                        </Card>
                    ))}
                </div>
            )}
        </div>
    );
}
