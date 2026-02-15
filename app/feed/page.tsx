"use client";

import { useQuery } from "@apollo/client/react";
import { GET_GLOBAL_FEED } from "@/lib/models/posts/gql";
import type { GetGlobalFeedQuery, GetGlobalFeedQueryVariables } from "@/lib/graphql/__generated__/graphql";
import {
    Loader2,
} from "lucide-react";
import { Page } from "@/components/page";
import { PostCard } from "@/components/feed/post-card";

export default function FeedPage() {
    const { data, loading } = useQuery<GetGlobalFeedQuery, GetGlobalFeedQueryVariables>(GET_GLOBAL_FEED);
    const posts = data?.globalFeed || [];

    if (loading && posts.length === 0) { // Only show full page loader if no posts are loaded yet
        return (
            <div className="flex items-center justify-center min-h-[60vh]">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <Page
            breadcrumbs={[
                { label: "Feed" }
            ]}
            header={
                <div className="space-y-1">
                    <h1 className="text-4xl font-extrabold tracking-tight">Feed</h1>
                    <p className="text-lg text-muted-foreground font-medium">What&apos;s happening in your communities</p>
                </div>
            }
        >
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
        </Page>
    );
}
