"use client";

import { useState } from "react";
import { CreatePost } from "./create-post";
import { PostList } from "./post-list";

interface Props {
  spaceId: string;
  isAdmin: boolean;
}

export function FeedSection({ spaceId, isAdmin }: Props) {
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  return (
    <>
      <CreatePost
        spaceId={spaceId}
        onPostCreated={() => setRefreshTrigger(prev => prev + 1)}
      />
      <PostList
        spaceId={spaceId}
        isAdmin={isAdmin}
        refreshTrigger={refreshTrigger}
      />
    </>
  );
}
