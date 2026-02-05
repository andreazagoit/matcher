"use client";

import { useEffect, useState } from "react";
import { graphql } from "@/lib/graphql/client";
import { Card, CardContent, CardHeader, CardFooter } from "@/components/ui/card";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { PageShell } from "@/components/page-shell";
import {
    Loader2,
    Heart,
    MessageCircle,
    Share2,
    MoreVertical,
    RefreshCw,
    Compass
} from "lucide-react";
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

    if (loading && posts.length === 0) { // Only show full page loader if no posts are loaded yet
        return (
            <div className="flex items-center justify-center min-h-[60vh]">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <PageShell
            title="Feed"
            subtitle="Stay updated with the latest posts from all spaces"
            actions={
                <Button variant="outline" size="sm" onClick={fetchFeed} className="gap-2">
                    <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                    Refresh
                </Button>
            }
        >
            <div className="max-w-2xl mx-auto space-y-8">
                {(!posts || posts.length === 0) ? (
                    <div className="text-center py-24 bg-muted/10 rounded-2xl border-2 border-dashed border-muted-foreground/20">
                        <div className="text-5xl mb-4">📭</div>
                        <h3 className="text-xl font-semibold">No posts to show yet</h3>
                        <p className="text-muted-foreground mt-2 max-w-sm mx-auto">
                            Be the first to share something with the community and start the conversation!
                        </p>
                    </div>
                ) : (
                    <div className="space-y-8">
                        {posts.map((post) => (
                            <Card key={post.id} className="overflow-hidden border-none shadow-md hover:shadow-xl transition-all">
                                <CardHeader className="flex flex-row items-center gap-4 space-y-0 p-5 bg-card/50 backdrop-blur-sm">
                                    <Avatar className="h-12 w-12 ring-2 ring-primary/10">
                                        <AvatarFallback className="bg-primary/5 text-primary">
                                            {post.author.firstName[0]}{post.author.lastName[0]}
                                        </AvatarFallback>
                                    </Avatar>
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center justify-between">
                                            <div>
                                                <p className="font-bold text-base truncate">
                                                    {post.author.firstName} {post.author.lastName}
                                                </p>
                                                <p className="text-xs text-muted-foreground flex items-center gap-1">
                                                    <Compass className="h-3 w-3" />
                                                    {new Date(post.createdAt).toLocaleDateString(undefined, {
                                                        month: 'short',
                                                        day: 'numeric',
                                                        year: 'numeric'
                                                    })}
                                                </p>
                                            </div>
                                            <Button variant="ghost" size="icon" className="h-9 w-9 text-muted-foreground hover:text-foreground">
                                                <MoreVertical className="h-5 w-5" />
                                            </Button>
                                        </div>
                                    </div>
                                </CardHeader>

                                <CardContent className="px-5 pb-5 pt-2">
                                    <p className="text-[15px] leading-relaxed whitespace-pre-wrap text-foreground/90">
                                        {post.content}
                                    </p>
                                    {post.mediaUrls && post.mediaUrls.length > 0 && (
                                        <div className="mt-5 grid gap-3">
                                            {post.mediaUrls.map((url, i) => (
                                                <div key={i} className="relative group overflow-hidden rounded-xl border border-border/50">
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

                                <Separator className="opacity-50" />

                                <CardFooter className="flex justify-between p-3 bg-muted/5">
                                    <div className="flex gap-2">
                                        <Button variant="ghost" size="sm" className="h-10 px-4 gap-2.5 rounded-full hover:bg-primary/5 hover:text-primary transition-colors">
                                            <Heart className="h-5 w-5" />
                                            <span className="text-sm font-medium">{post.likesCount || 0}</span>
                                        </Button>
                                        <Button variant="ghost" size="sm" className="h-10 px-4 gap-2.5 rounded-full hover:bg-primary/5 hover:text-primary transition-colors">
                                            <MessageCircle className="h-5 w-5" />
                                            <span className="text-sm font-medium">{post.commentsCount || 0}</span>
                                        </Button>
                                    </div>
                                    <Button variant="ghost" size="sm" className="h-10 w-10 p-0 rounded-full hover:bg-primary/5 hover:text-primary">
                                        <Share2 className="h-5 w-5" />
                                    </Button>
                                </CardFooter>
                            </Card>
                        ))}
                    </div>
                )}
            </div>
        </PageShell>
    );
}
