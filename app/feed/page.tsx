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
import { PostCard, type Post } from "@/components/feed/post-card";



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
            header={
                <div className="space-y-1">
                    <h1 className="text-4xl font-extrabold tracking-tight text-foreground bg-clip-text">Feed</h1>
                    <p className="text-lg text-muted-foreground font-medium">Stay updated with the latest posts from all spaces</p>
                </div>
            }
            actions={
                <Button variant="outline" size="sm" onClick={fetchFeed} className="gap-2">
                    <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                    Refresh
                </Button>
            }
        >
            <div className="space-y-8">
                {(!posts || posts.length === 0) ? (
                    <div className="text-center py-24 bg-muted/10 rounded-2xl border-2 border-dashed border-muted-foreground/20">
                        <div className="text-5xl mb-4">📭</div>
                        <h3 className="text-xl font-semibold">No posts to show yet</h3>
                        <p className="text-muted-foreground mt-2 max-w-sm mx-auto">
                            Be the first to share something with the community and start the conversation!
                        </p>
                    </div>
                ) : (
                    <div className="space-y-6">
                        {posts.map((post) => (
                            <PostCard key={post.id} post={post} />
                        ))}
                    </div>
                )}
            </div>
        </PageShell>
    );
}
