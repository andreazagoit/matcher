"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { CreateAppDialog } from "@/components/create-app-dialog";

interface App {
  id: string;
  name: string;
  description?: string;
  clientId: string;
  isActive: boolean;
  createdAt: string;
}

export default function DashboardPage() {
  const [apps, setApps] = useState<App[]>([]);
  const [loading, setLoading] = useState(true);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);

  const fetchApps = async () => {
    try {
      const res = await fetch("/api/dashboard/apps");
      if (res.ok) {
        const data = await res.json();
        setApps(data.apps);
      }
    } catch (error) {
      console.error("Failed to fetch apps:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchApps();
  }, []);

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Your Apps</h1>
          <p className="text-muted-foreground mt-1">Manage your OAuth applications</p>
        </div>
        <Button onClick={() => setCreateDialogOpen(true)}>+ Create App</Button>
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
      ) : apps.length === 0 ? (
        <Card className="text-center">
          <CardHeader>
            <div className="text-6xl mb-4">🔐</div>
            <CardTitle>No apps yet</CardTitle>
            <CardDescription>Create your first app to get started with OAuth and M2M access</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => setCreateDialogOpen(true)}>Create Your First App</Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {apps.map((app) => (
            <Link key={app.id} href={`/dashboard/${app.id}`}>
              <Card className="hover:border-primary/50 transition-colors cursor-pointer h-full">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="w-12 h-12 rounded-lg bg-primary flex items-center justify-center text-primary-foreground text-xl font-bold">
                      {app.name.charAt(0).toUpperCase()}
                    </div>
                    <Badge variant={app.isActive ? "default" : "secondary"}>
                      {app.isActive ? "Active" : "Inactive"}
                    </Badge>
                  </div>
                  <CardTitle className="mt-4">{app.name}</CardTitle>
                  {app.description && (
                    <CardDescription className="line-clamp-2">{app.description}</CardDescription>
                  )}
                </CardHeader>
                <CardContent>
                  <code className="text-xs text-muted-foreground font-mono">
                    {app.clientId}
                  </code>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}

      <CreateAppDialog
        open={createDialogOpen}
        onOpenChange={setCreateDialogOpen}
        onCreated={fetchApps}
      />
    </div>
  );
}
