"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
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
  isPublic: boolean;
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
            isPublic
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
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Your Spaces</h1>
          <p className="text-muted-foreground mt-1">Manage your communities and clubs</p>
        </div>
        <Button onClick={() => setCreateDialogOpen(true)}>+ Create Space</Button>
      </div>

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
        <Card className="text-center">
          <CardHeader>
            <div className="text-6xl mb-4">🪐</div>
            <CardTitle>No spaces yet</CardTitle>
            <CardDescription>Create your first space to start building your community</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => setCreateDialogOpen(true)}>Create Your First Space</Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {spaces.map((space) => (
            <Link key={space.id} href={`/spaces/${space.id}`}>
              <Card className="hover:border-primary/50 transition-colors cursor-pointer h-full">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center text-2xl">
                      {/* Would use space logo here */}
                      {space.name.charAt(0).toUpperCase()}
                    </div>
                    <div className="flex gap-2">
                      <Badge variant="outline">
                        {space.isPublic ? "Public" : "Private"}
                      </Badge>
                      <Badge variant={space.isActive ? "default" : "secondary"}>
                        {space.isActive ? "Active" : "Inactive"}
                      </Badge>
                    </div>
                  </div>
                  <CardTitle className="mt-4">{space.name}</CardTitle>
                  {space.description && (
                    <CardDescription className="line-clamp-2">{space.description}</CardDescription>
                  )}
                </CardHeader>
                <CardContent>
                  <div className="flex justify-between items-center text-sm text-muted-foreground">
                    <span className="font-mono">{space.slug}</span>
                    <span>{space.membersCount} members</span>
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
    </div>
  );
}
