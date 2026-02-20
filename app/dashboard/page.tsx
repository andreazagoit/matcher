"use client";

import { useQuery } from "@apollo/client/react";
import { GET_MY_SPACES } from "@/lib/models/spaces/gql";
import { useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { CreateSpaceDialog } from "@/components/create-space-dialog";
import { Plus } from "lucide-react";
import { Page } from "@/components/page";
import type {
  GetMySpacesQuery,
  GetMySpacesQueryVariables
} from "@/lib/graphql/__generated__/graphql";

export default function DashboardPage() {
  const { data: spacesData, loading: spacesLoading, refetch: refetchSpaces } = useQuery<GetMySpacesQuery, GetMySpacesQueryVariables>(GET_MY_SPACES);

  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const spaces = (spacesData?.mySpaces || []).filter(
    (s) => s.myMembership?.role === "owner"
  );

  return (
    <Page
      breadcrumbs={[
        { label: "Dashboard" }
      ]}
      header={
        <div className="space-y-1">
          <h1 className="text-2xl font-bold tracking-tight text-foreground bg-clip-text">Dashboard</h1>
          <p className="text-muted-foreground">Overview of your activity and matches</p>
        </div>
      }
    >
      {/* My Spaces Section */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Plus className="h-5 w-5 text-primary" />
            <h2 className="text-xl font-bold tracking-tight">Spaces creati</h2>
          </div>
          {spaces.length > 0 && (
            <Button onClick={() => setCreateDialogOpen(true)} size="sm" variant="outline">
              <Plus className="h-4 w-4 mr-2" />
              New Space
            </Button>
          )}
        </div>

        {spacesLoading ? (
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
          <Card className="text-center py-12 border-dashed">
            <CardHeader>
              <div className="text-6xl mb-4">🪐</div>
              <CardTitle className="text-2xl">Nessuno space creato</CardTitle>
              <CardDescription className="text-lg">Crea il tuo primo space per costruire la tua community</CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={() => setCreateDialogOpen(true)} size="lg">Crea il tuo primo Space</Button>
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
      </section>

      <CreateSpaceDialog
        open={createDialogOpen}
        onOpenChange={setCreateDialogOpen}
        onCreated={() => refetchSpaces()}
      />
    </Page>
  );
}
