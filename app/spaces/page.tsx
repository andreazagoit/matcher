"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { CreateSpaceDialog } from "@/components/create-space-dialog";
import { graphql } from "@/lib/graphql/client";
import { SpaceCard } from "@/components/spaces/space-card";

interface Space {
  id: string;
  name: string;
  slug: string;
  description?: string;
  clientId: string;
  isActive: boolean;
  visibility: string;
  joinPolicy: string;
  membersCount: number;
  createdAt: string;
  image?: string;
}

export default function DiscoverSpacesPage() {
  const [spaces, setSpaces] = useState<Space[]>([]);
  const [loading, setLoading] = useState(true);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);

  const fetchSpaces = async () => {
    try {
      const data = await graphql<{ spaces: Space[] }>(`
        query GetAllSpaces {
          spaces {
            id
            name
            slug
            description
            clientId
            isActive
            visibility
            joinPolicy
            membersCount
            createdAt
            image
          }
        }
      `);
      setSpaces(data.spaces);
    } catch (error) {
      console.error("Failed to fetch spaces:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSpaces();
  }, []);

  return (
    <>
      <PageShell
        header={
          <div className="space-y-1">
            <h1 className="text-4xl font-extrabold tracking-tight">Discover Spaces</h1>
            <p className="text-lg text-muted-foreground font-medium">Explore and join communities and clubs</p>
          </div>
        }
        actions={
          <Button onClick={() => setCreateDialogOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Create Space
          </Button>
        }
      >
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[1, 2, 3].map((i) => (
              <Card key={i} className="shadow-none border animate-pulse">
                <CardHeader className="p-4 flex flex-row items-center gap-4">
                  <div className="size-10 bg-muted rounded-lg" />
                  <div className="space-y-2 flex-1">
                    <div className="h-4 bg-muted rounded w-1/3" />
                    <div className="h-3 bg-muted rounded w-1/4" />
                  </div>
                </CardHeader>
                <CardContent className="p-4 pt-0">
                  <div className="h-4 bg-muted rounded w-full" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : spaces.length === 0 ? (
          <Card className="text-center py-12 shadow-none border-dashed">
            <CardHeader>
              <div className="text-6xl mb-4">🪐</div>
              <CardTitle className="text-xl">No spaces yet</CardTitle>
              <CardDescription>Create your first space to start building your community</CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={() => setCreateDialogOpen(true)}>Create Space</Button>
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {spaces.map((space) => (
              <SpaceCard key={space.id} space={space} />
            ))}
          </div>
        )}
      </PageShell>

      <CreateSpaceDialog
        open={createDialogOpen}
        onOpenChange={setCreateDialogOpen}
        onCreated={fetchSpaces}
      />
    </>
  );
}
