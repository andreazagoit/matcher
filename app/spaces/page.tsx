"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Plus, Users } from "lucide-react";
import { PageShell } from "@/components/page-shell";
import { Badge } from "@/components/ui/badge";
import { CreateSpaceDialog } from "@/components/create-space-dialog";
import { graphql } from "@/lib/graphql/client";

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
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {spaces.map((space) => (
              <Link key={space.id} href={`/spaces/${space.id}`} className="block h-full group">
                <Card className="h-full flex flex-col shadow-none border transition-all hover:shadow-md hover:border-foreground/10 overflow-hidden">
                  <CardHeader className="flex flex-row items-center gap-4 p-4">
                    <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground ring-1 ring-border group-hover:bg-primary/10 group-hover:text-primary transition-colors">
                      <span className="text-lg font-bold">{space.name.charAt(0).toUpperCase()}</span>
                    </div>
                    <div className="flex flex-col gap-0.5 min-w-0">
                      <h3 className="font-semibold tracking-tight text-base truncate group-hover:text-primary transition-colors">
                        {space.name}
                      </h3>
                      <p className="text-xs text-muted-foreground font-mono truncate">
                        {space.slug}
                      </p>
                    </div>
                  </CardHeader>

                  <CardContent className="p-4 pt-0 flex-1">
                    {space.description ? (
                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {space.description}
                      </p>
                    ) : (
                      <p className="text-sm text-muted-foreground italic">
                        No description provided
                      </p>
                    )}
                  </CardContent>

                  <CardFooter className="p-4 pt-0 flex justify-between items-center mt-auto">
                    <div className="flex gap-2">
                      <Badge variant={space.visibility === "public" ? "outline" : "secondary"} className="text-[10px] h-5 px-1.5 font-normal rounded-md">
                        {space.visibility === "public" ? "Public" : "Private"}
                      </Badge>
                      {space.isActive && (
                        <Badge variant="default" className="text-[10px] h-5 px-1.5 font-normal rounded-md bg-green-500/10 text-green-600 hover:bg-green-500/20 shadow-none border-0">
                          Active
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                      <Users className="size-3.5" />
                      <span>{space.membersCount}</span>
                    </div>
                  </CardFooter>
                </Card>
              </Link>
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
