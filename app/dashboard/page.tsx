"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { CreateSpaceDialog } from "@/components/create-space-dialog";
import { Plus } from "lucide-react";
import { PageShell } from "@/components/page-shell";
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

export default function DashboardPage() {
  const [spaces, setSpaces] = useState<Space[]>([]);
  const [loading, setLoading] = useState(true);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);

  const fetchSpaces = async () => {
    try {
      const data = await graphql<{ mySpaces: Space[] }>(`
        query GetMySpaces {
          mySpaces {
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
      setSpaces(data.mySpaces);
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
    <PageShell
      title="Dashboard"
      subtitle="Manage your communities and clubs"
      actions={
        <Button onClick={() => setCreateDialogOpen(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Create Space
        </Button>
      }
    >
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader>
                <div className="h-6 bg-muted rounded w-2/3 mb-2"></div>
                <div className="h-4 bg-muted rounded w-1/2"></div>
              </CardHeader>
              <CardContent>
                <div className="h-4 bg-muted rounded w-full"></div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : spaces.length === 0 ? (
        <Card className="text-center py-12">
          <CardHeader>
            <div className="text-6xl mb-4">🪐</div>
            <CardTitle className="text-2xl">No spaces yet</CardTitle>
            <CardDescription className="text-lg">Create your first space to start building your community</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => setCreateDialogOpen(true)} size="lg">Create Your First Space</Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {spaces.map((space) => (
            <Link key={space.id} href={`/spaces/${space.id}`}>
              <Card className="hover:border-primary/50 transition-all hover:shadow-lg cursor-pointer h-full group">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center text-3xl font-bold group-hover:scale-110 transition-transform">
                      {space.name.charAt(0).toUpperCase()}
                    </div>
                    <div className="flex gap-2">
                      <Badge variant={space.visibility === "public" ? "outline" : "secondary"}>
                        {space.visibility === "public" ? "Public" : "Private"}
                      </Badge>
                      <Badge variant={space.isActive ? "default" : "secondary"}>
                        {space.isActive ? "Active" : "Inactive"}
                      </Badge>
                    </div>
                  </div>
                  <CardTitle className="mt-5 text-2xl group-hover:text-primary transition-colors">{space.name}</CardTitle>
                  {space.description && (
                    <CardDescription className="line-clamp-2 mt-2 text-base">{space.description}</CardDescription>
                  )}
                </CardHeader>
                <CardContent>
                  <div className="flex justify-between items-center text-sm text-muted-foreground pt-4 border-t border-border/50">
                    <span className="font-mono bg-muted px-2 py-0.5 rounded text-xs">{space.slug}</span>
                    <span className="flex items-center gap-1.5 font-medium">
                      <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                      {space.membersCount} members
                    </span>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}

      <CreateSpaceDialog
        open={createDialogOpen}
        onOpenChange={setCreateDialogOpen}
        onCreated={fetchSpaces}
      />
    </PageShell>
  );
}
